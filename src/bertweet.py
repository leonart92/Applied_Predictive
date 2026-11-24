"""Fine-tuning de BERTweet sur le texte des tweets de CrisisMMD.

Protocol A trains on the official split. Protocol B holds one event out of
train and dev and tests on that event only. Splits come from src/data.py;
predictions and metrics are written with src/evaluate.py.

Usage:
    python src/bertweet.py informative                          # protocol A
    python src/bertweet.py humanitarian --event hurricane_maria  # protocol B
"""
import argparse

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

if __package__:
    from .data import EVENTS, LABELS, ROOT, get_split
    from .evaluate import evaluate_file, write_predictions
else:
    from data import EVENTS, LABELS, ROOT, get_split
    from evaluate import evaluate_file, write_predictions

PRED_DIR = ROOT / "results" / "predictions"
METRIC_DIR = ROOT / "results" / "metrics"
# Protocol B targets for the deep models (PROJET.md §4): a hurricane, a disaster
# type seen in training, and the only wildfire, a type absent from training.
TARGET_EVENTS = ("hurricane_maria", "california_wildfires")

MODEL_NAME = "vinai/bertweet-base"
MAX_LEN = 128
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 5
PATIENCE = 2  # early stopping : nombre d'epochs sans progrès du F1 macro dev
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def make_loader(df, tokenizer, label2id, shuffle=False):
    encodings = tokenizer(df["tweet_text"].tolist(), truncation=True, max_length=MAX_LEN)
    examples = [
        {"input_ids": ids, "attention_mask": mask, "labels": label2id[label]}
        for ids, mask, label in zip(encodings["input_ids"], encodings["attention_mask"], df["label"])
    ]
    # padding dynamique : chaque batch est complété à la longueur de son tweet le plus long
    return DataLoader(examples, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer))


def logits_of(model, batch):
    return model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE)).logits


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
        print(f"epoch {epoch} · loss train {train_loss / len(train_loader):.4f} · F1 macro dev {f1:.4f}")

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
    path = write_predictions(frame, y_pred, probas, task=task, model="bertweet", protocol=protocol,
                             fold=fold, subset=subset, output_dir=PRED_DIR / folder if folder else None)
    evaluate_file(path, subset=subset, output_dir=METRIC_DIR / folder if folder else None)
    print(f"{subset} · {path.relative_to(ROOT)}")
    print(classification_report(frame["label"], y_pred, labels=list(labels), digits=4, zero_division=0))


def main(task, event=None):
    protocol, fold = ("B", event) if event else ("A", "official")
    splits = get_split(task, protocol, fold, text_only=True)
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    labels = LABELS[task]
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"{task} · {protocol}/{fold} · train : {len(train)} tweets · dev : {len(dev)} · test : {len(test)} · {DEVICE}")

    # normalization=True applique la normalisation de BERTweet : mentions -> @USER, URLs -> HTTPURL, emojis -> texte
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, normalization=True)
    train_loader = make_loader(train, tokenizer, label2id, shuffle=True)
    dev_loader = make_loader(dev, tokenizer, label2id)
    test_loader = make_loader(test, tokenizer, label2id)

    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels)).to(DEVICE)

    # Déséquilibre des classes : poids "balanced" dans la loss, même formule que scikit-learn
    counts = train["label"].value_counts()[list(labels)].to_numpy()
    class_weights = torch.tensor(len(train) / (len(labels) * counts), dtype=torch.float32, device=DEVICE)
    fit(model, train_loader, dev_loader, dev["label"].map(label2id).to_numpy(), class_weights)

    probas = predict(model, test_loader)
    save(test, probas, task, protocol, fold)

    if protocol == "A":
        # In-domain side of the protocol B gap: this same model, restricted to
        # the test tweets of each target event, so A and B score identical rows.
        for target in TARGET_EVENTS:
            mask = (test["event_name"] == target).to_numpy()
            save(test[mask], probas[mask], task, "B", target, folder="in_domain")
    else:
        # Supplementary score on every tweet of the unseen event (train, dev and test).
        all_target = splits["test_all_target"]
        save(all_target, predict(model, make_loader(all_target, tokenizer, label2id)),
             task, protocol, fold, subset="all_target")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERTweet on CrisisMMD tweet text.")
    parser.add_argument("task", nargs="?", default="informative", choices=tuple(LABELS))
    parser.add_argument("--event", choices=EVENTS, help="held-out event for protocol B; protocol A if omitted")
    args = parser.parse_args()
    main(args.task, args.event)
