"""Fine-tuning d'un Vision Transformer sur les images des tweets de CrisisMMD.

Le modèle ne voit que l'image : aucun texte n'est lu. La recette d'entraînement
est volontairement celle de src/bertweet.py (batch, learning rate, epochs,
patience, loss pondérée, meilleur état sur le F1 macro dev, seed), pour que
l'écart entre modèle texte et modèle image vienne de la modalité et non du
réglage. Protocol A entraîne sur le split officiel ; protocol B tient un
événement à l'écart de train et dev et teste sur lui seul. Les splits viennent
de src/data.py, les prédictions et les métriques de src/evaluate.py.

Usage:
    python src/vit.py informative                          # protocol A
    python src/vit.py humanitarian --event hurricane_maria  # protocol B
"""
import argparse

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup, set_seed

if __package__:
    from .data import EVENTS, LABELS, ROOT, get_split
    from .evaluate import evaluate_file, write_predictions
    from .images import ImageDataset
else:
    from data import EVENTS, LABELS, ROOT, get_split
    from evaluate import evaluate_file, write_predictions
    from images import ImageDataset

PRED_DIR = ROOT / "results" / "predictions"
METRIC_DIR = ROOT / "results" / "metrics"
# Mêmes cibles du protocole B que BERTweet (PROJET.md §4) : un ouragan, type de
# catastrophe vu à l'entraînement, et le seul incendie, type absent.
TARGET_EVENTS = ("hurricane_maria", "california_wildfires")

MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 5
PATIENCE = 2  # early stopping : nombre d'epochs sans progrès du F1 macro dev
SEED = 42
WORKERS = 4  # décodage JPEG en parallèle, sinon le GPU attend les images

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Preprocess:
    """Redimensionnement et normalisation du checkpoint, sous forme picklable.

    Les workers du DataLoader démarrent par spawn sur macOS : une closure ne
    passerait pas, une instance de classe au niveau du module oui.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


def make_loader(df, transform, label2id, shuffle=False, batch_size=BATCH_SIZE, workers=WORKERS):
    """Images dans l'ordre des lignes du DataFrame, pas de déduplication par tweet."""
    dataset = ImageDataset(df["image"], transform, [label2id[label] for label in df["label"]])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0)


def logits_of(model, batch):
    return model(pixel_values=batch["pixel_values"].to(DEVICE)).logits


@torch.no_grad()
def predict(model, loader):
    """Probabilités par classe, dans l'ordre des lignes du loader."""
    model.eval()
    return torch.cat([logits_of(model, batch).softmax(dim=-1).cpu() for batch in loader]).numpy()


def fit(model, train_loader, dev_loader, y_dev, class_weights):
    """Entraîne le modèle et le laisse dans son meilleur état sur dev (F1 macro)."""
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    # AdamW, warmup linéaire sur 10 % des pas puis décroissance linéaire jusqu'à 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * n_steps), n_steps)

    best_f1, best_state, patience = 0.0, None, PATIENCE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            loss = loss_fn(logits_of(model, batch), batch["labels"].to(DEVICE))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        f1 = f1_score(y_dev, predict(model, dev_loader).argmax(axis=1), average="macro")
        print(f"epoch {epoch} · loss train {train_loss / len(train_loader):.4f} · F1 macro dev {f1:.4f}", flush=True)

        if f1 > best_f1:  # meilleur modèle sur dev : on garde une copie de ses poids
            best_f1, patience = f1, PATIENCE
            best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best_state)
    print(f"meilleur F1 macro dev : {best_f1:.4f}")


def save(frame, probas, task, protocol, fold, *, subset="test", folder=None):
    """Write the predictions in the shared format, then their metrics JSON."""
    labels = LABELS[task]
    y_pred = [labels[i] for i in probas.argmax(axis=1)]
    path = write_predictions(frame, y_pred, probas, task=task, model="vit", protocol=protocol,
                             fold=fold, subset=subset, output_dir=PRED_DIR / folder if folder else None)
    evaluate_file(path, subset=subset, output_dir=METRIC_DIR / folder if folder else None)
    print(f"{subset} · {path.relative_to(ROOT)}")
    print(classification_report(frame["label"], y_pred, labels=list(labels), digits=4, zero_division=0))


def main(task, event=None, batch_size=BATCH_SIZE, workers=WORKERS):
    protocol, fold = ("B", event) if event else ("A", "official")
    # text_only=False : l'entrée est l'image, donc un tweet à plusieurs images
    # fournit plusieurs exemples et aucune ligne n'est dédupliquée.
    splits = get_split(task, protocol, fold)
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    labels = LABELS[task]
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"{task} · {protocol}/{fold} · train : {len(train)} paires · dev : {len(dev)} · test : {len(test)} · {DEVICE}")

    transform = Preprocess(AutoImageProcessor.from_pretrained(MODEL_NAME))
    train_loader = make_loader(train, transform, label2id, shuffle=True, batch_size=batch_size, workers=workers)
    dev_loader = make_loader(dev, transform, label2id, batch_size=batch_size, workers=workers)
    test_loader = make_loader(test, transform, label2id, batch_size=batch_size, workers=workers)

    set_seed(SEED)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, num_labels=len(labels)).to(DEVICE)

    # Déséquilibre des classes : poids "balanced" dans la loss, même formule que scikit-learn
    counts = train["label"].value_counts()[list(labels)].to_numpy()
    class_weights = torch.tensor(len(train) / (len(labels) * counts), dtype=torch.float32, device=DEVICE)
    fit(model, train_loader, dev_loader, dev["label"].map(label2id).to_numpy(), class_weights)

    probas = predict(model, test_loader)
    save(test, probas, task, protocol, fold)

    if protocol == "A":
        # In-domain side of the protocol B gap: this same model, restricted to
        # the test pairs of each target event, so A and B score identical rows.
        for target in TARGET_EVENTS:
            mask = (test["event_name"] == target).to_numpy()
            save(test[mask], probas[mask], task, "B", target, folder="in_domain")
    else:
        # Supplementary score on every pair of the unseen event (train, dev and test).
        all_target = splits["test_all_target"]
        save(all_target, predict(model, make_loader(all_target, transform, label2id, batch_size=batch_size, workers=workers)),
             task, protocol, fold, subset="all_target")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Vision Transformer on CrisisMMD images.")
    parser.add_argument("task", nargs="?", default="informative", choices=tuple(LABELS))
    parser.add_argument("--event", choices=EVENTS, help="held-out event for protocol B; protocol A if omitted")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    main(args.task, args.event, args.batch_size, args.workers)
