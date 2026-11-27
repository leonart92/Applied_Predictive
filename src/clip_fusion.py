"""Fusion texte + image : embeddings CLIP gelés et régression logistique.

CLIP encode séparément le tweet et son image ; les deux vecteurs sont normalisés
L2, mis bout à bout (512 + 512 = 1024) et un classifieur linéaire tranche. Les
encodeurs ne sont jamais entraînés : seule la régression logistique apprend, avec
exactement la configuration du baseline TF-IDF (grille de C sélectionnée sur dev
au F1 macro, class_weight='balanced', mêmes solveur, pénalité et tolérance), de
sorte que la comparaison TF-IDF -> fusion isole le changement de features.

Les embeddings des 12 708 images et de tous les textes des six splits sont
calculés une seule fois puis relus depuis cache/ : les six runs et l'ablation ne
repayent pas CLIP. Le cache n'est pas versionné.

Usage:
    python src/clip_fusion.py informative                           # protocol A
    python src/clip_fusion.py humanitarian --event hurricane_maria   # protocol B
    python src/clip_fusion.py informative --ablation                 # texte / image / fusion
"""
import argparse
import hashlib
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from transformers import AutoProcessor, CLIPModel

if __package__:
    from .data import EVENTS, LABELS, ROOT, SEED, SPLITS, get_split, load_split
    from .evaluate import compute_metrics, evaluate_file, write_predictions
    from .images import ImageDataset, unique_images
else:
    from data import EVENTS, LABELS, ROOT, SEED, SPLITS, get_split, load_split
    from evaluate import compute_metrics, evaluate_file, write_predictions
    from images import ImageDataset, unique_images

PRED_DIR = ROOT / "results" / "predictions"
METRIC_DIR = ROOT / "results" / "metrics"
TABLE_DIR = ROOT / "results" / "tables"
CACHE_DIR = ROOT / "cache"
TARGET_EVENTS = ("hurricane_maria", "california_wildfires")

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
MAX_TEXT_TOKENS = 77  # limite du tube texte de CLIP ; les tweets plus longs sont tronqués
MODES = ("text", "image", "fusion")
# Configuration reprise telle quelle du baseline TF-IDF, pour que les deux
# modèles ne diffèrent que par leurs features.
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
LR_CONFIG = dict(class_weight="balanced", solver="lbfgs", penalty="l2",
                 max_iter=2000, tol=1e-4, random_state=SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def shown(path):
    """Chemin relatif au dépôt quand c'est possible, absolu sinon (cf. evaluate.py)."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def text_key(text):
    """Clé de cache d'un texte : son sha1, pour un npz stable et compact."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_cache(path):
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return dict(zip(data["keys"].tolist(), data["vectors"]))


def save_cache(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(table)
    np.savez(path, keys=np.array(keys), vectors=np.stack([table[key] for key in keys]))


def ensure_table(path, items, embed):
    """Complète le cache pour `items` (clé -> charge utile) et rend ce sous-ensemble.

    `embed` n'est appelé que sur ce qui manque, et doit rendre un vecteur par
    charge utile, dans le même ordre.
    """
    table = load_cache(path)
    missing = [key for key in items if key not in table]
    if missing:
        print(f"  {len(missing)} embeddings à calculer ({path.name})", flush=True)
        vectors = np.asarray(embed([items[key] for key in missing]), dtype=np.float32)
        if vectors.shape[0] != len(missing):
            raise ValueError("L'encodeur a rendu un nombre de vecteurs différent du nombre d'entrées")
        table.update(zip(missing, vectors))
        save_cache(path, table)
    return {key: table[key] for key in items}


def projected(output):
    """Vecteur projeté de CLIP : transformers 5 rend un objet de sortie, 4 un tenseur."""
    return output if torch.is_tensor(output) else output.pooler_output


class ImagePreprocess:
    """Prétraitement CLIP picklable, pour les workers spawn du DataLoader."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


def clip_embedders(batch_size=BATCH_SIZE, root=ROOT, workers=4):
    """Deux fonctions d'encodage, qui ne chargent CLIP que si le cache est incomplet."""
    state = {}

    def ready():
        if not state:
            print(f"chargement de {MODEL_NAME} sur {DEVICE}", flush=True)
            state["model"] = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
            state["processor"] = AutoProcessor.from_pretrained(MODEL_NAME)
        return state["model"], state["processor"]

    @torch.no_grad()
    def embed_texts(texts):
        model, processor = ready()
        vectors = []
        for start in range(0, len(texts), batch_size):
            inputs = processor.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                         max_length=MAX_TEXT_TOKENS, return_tensors="pt").to(DEVICE)
            features = projected(model.get_text_features(**inputs))
            vectors.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
        return np.concatenate(vectors)

    @torch.no_grad()
    def embed_images(paths):
        model, processor = ready()
        transform = ImagePreprocess(processor)
        loader = torch.utils.data.DataLoader(
            ImageDataset(paths, transform, root=root), batch_size=batch_size,
            num_workers=workers, persistent_workers=workers > 0)
        vectors = []
        for done, batch in enumerate(loader, start=1):
            features = projected(model.get_image_features(pixel_values=batch["pixel_values"].to(DEVICE)))
            vectors.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
            if done % 25 == 0:
                print(f"    {done * batch_size}/{len(paths)} images encodées", flush=True)
        return np.concatenate(vectors)

    return embed_texts, embed_images


def clip_tables(batch_size=BATCH_SIZE, cache_dir=CACHE_DIR, root=ROOT, split_dir=None, embedders=None):
    """Tables d'embeddings couvrant les six splits, calculées une fois pour toutes.

    La couverture est volontairement globale : le premier run paye CLIP, les
    suivants (autre tâche, autre protocole, ablation) ne relisent que le cache.
    """
    frames = [load_split(task, split, split_dir=split_dir) for task in LABELS for split in SPLITS]
    images = {path: path for path in unique_images(*frames)}
    texts = {}
    for frame in frames:
        for text in frame["tweet_text"]:
            texts.setdefault(text_key(text), text)
    print(f"cache CLIP : {len(texts)} textes, {len(images)} images", flush=True)
    embed_texts, embed_images = embedders if embedders else clip_embedders(batch_size, root)
    text_table = ensure_table(cache_dir / "clip_vit_b32_text.npz", texts, embed_texts)
    image_table = ensure_table(cache_dir / "clip_vit_b32_image.npz", images, embed_images)
    return text_table, image_table


def build_features(frame, text_table, image_table, mode="fusion"):
    """Matrice (n, d) dans l'ordre des lignes du DataFrame ; fusion = [texte | image]."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    parts = []
    if mode in ("text", "fusion"):
        parts.append(np.stack([text_table[text_key(text)] for text in frame["tweet_text"]]))
    if mode in ("image", "fusion"):
        parts.append(np.stack([image_table[image] for image in frame["image"]]))
    return np.concatenate(parts, axis=1).astype(np.float64)


def select_classifier(x_train, y_train, x_dev, y_dev, labels):
    """Grille de C sélectionnée sur dev au F1 macro, comme le baseline TF-IDF."""
    best = None
    for c in C_VALUES:
        with warnings.catch_warnings():
            # penalty='l2' est déprécié depuis scikit-learn 1.8 mais reste la clé
            # enregistrée dans tfidf_A_run_config.json : on garde la configuration
            # du baseline à l'identique plutôt que de la réécrire de notre côté.
            warnings.filterwarnings('ignore', category=FutureWarning)
            candidate = LogisticRegression(C=c, **LR_CONFIG).fit(x_train, y_train)
        f1 = f1_score(y_dev, candidate.predict(x_dev), labels=list(labels),
                      average="macro", zero_division=0)
        iterations = int(np.max(candidate.n_iter_))
        print(f"  C={c:<6} F1 macro dev {f1:.4f} · {iterations} itérations"
              f"{' (non convergé)' if iterations >= LR_CONFIG['max_iter'] else ''}", flush=True)
        if best is None or f1 > best["f1_dev"]:
            best = dict(f1_dev=f1, C=c, classifier=candidate, n_iter=iterations)
    return best


def probabilities(classifier, features, labels):
    """predict_proba réordonné de classifier.classes_ vers l'ordre de LABELS[task]."""
    index = {label: i for i, label in enumerate(classifier.classes_)}
    absent = [label for label in labels if label not in index]
    if absent:
        raise ValueError(f"Classes absentes de l'entraînement, probabilités non définies : {absent}")
    return classifier.predict_proba(features)[:, [index[label] for label in labels]]


def save(frame, probas, task, protocol, fold, *, subset="test", folder=None):
    """Write the predictions in the shared format, then their metrics JSON."""
    labels = LABELS[task]
    y_pred = [labels[i] for i in probas.argmax(axis=1)]
    path = write_predictions(frame, y_pred, probas, task=task, model="clip_fusion", protocol=protocol,
                             fold=fold, subset=subset, output_dir=PRED_DIR / folder if folder else None)
    evaluate_file(path, subset=subset, output_dir=METRIC_DIR / folder if folder else None)
    print(f"{subset} · {shown(path)}")
    print(classification_report(frame["label"], y_pred, labels=list(labels), digits=4, zero_division=0))


def write_ablation(rows, path=None):
    """Ajoute ou remplace les lignes d'ablation, sans toucher aux autres runs."""
    path = path if path is not None else TABLE_DIR / "clip_fusion_ablation.csv"
    frame = pd.DataFrame(rows)
    keys = ["task", "protocol", "fold", "mode"]
    if path.is_file():
        previous = pd.read_csv(path)
        merged = previous.merge(frame[keys], on=keys, how="left", indicator=True)
        frame = pd.concat([previous[(merged["_merge"] == "left_only").to_numpy()], frame], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(keys).to_csv(path, index=False)
    print(f"ablation · {shown(path)}")
    return path


def main(task, event=None, ablation=False, batch_size=BATCH_SIZE):
    protocol, fold = ("B", event) if event else ("A", "official")
    # text_only=False : l'image fait partie de l'entrée, donc aucune ligne n'est
    # dédupliquée et la population reste celle que src/evaluate.py valide.
    splits = get_split(task, protocol, fold)
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    labels = LABELS[task]
    print(f"{task} · {protocol}/{fold} · train : {len(train)} paires · dev : {len(dev)} · test : {len(test)}")

    text_table, image_table = clip_tables(batch_size)
    features = {mode: {name: build_features(frame, text_table, image_table, mode)
                       for name, frame in (("train", train), ("dev", dev), ("test", test))}
                for mode in (MODES if ablation else ("fusion",))}

    rows = []
    for mode, matrices in features.items():
        print(f"features {mode} : {matrices['train'].shape[1]} dimensions")
        best = select_classifier(matrices["train"], train["label"], matrices["dev"], dev["label"], labels)
        print(f"  retenu : C={best['C']} · F1 macro dev {best['f1_dev']:.4f}")
        probas = probabilities(best["classifier"], matrices["test"], labels)
        if mode == "fusion":
            save(test, probas, task, protocol, fold)
            if protocol == "A":
                # Côté in-domain de l'écart du protocole B : le même modèle,
                # restreint aux paires test de chaque événement cible.
                for target in TARGET_EVENTS:
                    mask = (test["event_name"] == target).to_numpy()
                    save(test[mask], probas[mask], task, "B", target, folder="in_domain")
            else:
                all_target = splits["test_all_target"]
                save(all_target,
                     probabilities(best["classifier"],
                                   build_features(all_target, text_table, image_table, mode), labels),
                     task, protocol, fold, subset="all_target")
        if ablation:
            scores = compute_metrics(test["label"].to_numpy(),
                                     np.asarray(labels)[probas.argmax(axis=1)], task)
            rows.append(dict(task=task, protocol=protocol, fold=fold, mode=mode,
                             dimensions=matrices["train"].shape[1], C=best["C"],
                             f1_macro_dev=round(best["f1_dev"], 6),
                             f1_macro_test=round(scores["f1_macro"], 6),
                             f1_weighted_test=round(scores["f1_weighted"], 6),
                             accuracy_test=round(scores["accuracy"], 6),
                             n_train=len(train), n_test=len(test)))
    if rows:
        write_ablation(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP embedding fusion of tweet text and image.")
    parser.add_argument("task", nargs="?", default="informative", choices=tuple(LABELS))
    parser.add_argument("--event", choices=EVENTS, help="held-out event for protocol B; protocol A if omitted")
    parser.add_argument("--ablation", action="store_true", help="also score text-only and image-only features")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args.task, args.event, args.ablation, args.batch_size)
