"""Fine-tuning de BERTweet sur le texte des tweets de CrisisMMD.

Protocole A (split officiel « agreed labels »), à partir des TSV officiels bruts.
Écrit les prédictions du test dans results/predictions/, au format commun du projet.

Usage : python src/bertweet.py [informative|humanitarian]
"""
import csv
import sys
from pathlib import Path

import pandas as pd
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

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "crisismmd_datasplit_agreed_label"
PRED_DIR = ROOT / "results" / "predictions"

MODEL_NAME = "vinai/bertweet-base"
MAX_LEN = 128
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 5
PATIENCE = 2  # early stopping : nombre d'epochs sans progrès du F1 macro dev
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_split(task, split):
    path = DATA_DIR / f"task_{task}_text_img_agreed_lab_{split}.tsv"
    # QUOTE_NONE : ces TSV n'échappent rien, sinon pandas retire les guillemets des tweets qui commencent par "
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)


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


def main(task):
    train, dev, test = (load_split(task, split) for split in ("train", "dev", "test"))
    train = train.drop_duplicates("tweet_id")  # modèle texte : un exemple par tweet, pas par image
    classes = sorted(train["label"].unique())
    label2id = {label: i for i, label in enumerate(classes)}
    print(f"{task} · train : {len(train)} tweets · dev : {len(dev)} · test : {len(test)} · {DEVICE}")

    # normalization=True applique la normalisation de BERTweet : mentions -> @USER, URLs -> HTTPURL, emojis -> texte
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, normalization=True)
    train_loader = make_loader(train, tokenizer, label2id, shuffle=True)
    dev_loader = make_loader(dev, tokenizer, label2id)
    test_loader = make_loader(test, tokenizer, label2id)

    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(classes)).to(DEVICE)

    # Déséquilibre des classes : poids "balanced" dans la loss, même formule que scikit-learn
    counts = train["label"].value_counts()[classes].to_numpy()
    class_weights = torch.tensor(len(train) / (len(classes) * counts), dtype=torch.float32, device=DEVICE)
    fit(model, train_loader, dev_loader, dev["label"].map(label2id).to_numpy(), class_weights)

    probas = predict(model, test_loader)
    predictions = test[["tweet_id", "image_id", "event_name"]].copy()
    predictions["y_true"] = test["label"]
    predictions["y_pred"] = [classes[i] for i in probas.argmax(axis=1)]
    for i, label in enumerate(classes):
        predictions[f"proba_{label}"] = probas[:, i]

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PRED_DIR / f"{task}_bertweet_A_official.csv", index=False, float_format="%.6f")

    # Aperçu rapide : les métriques du rapport sont calculées par src/evaluate.py à partir de ce CSV
    print(classification_report(predictions["y_true"], predictions["y_pred"], digits=4))
    print(pd.crosstab(predictions["y_true"], predictions["y_pred"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "informative")
