"""Analyse d'erreurs inter-modèles sur les prédictions de CrisisMMD (Milestone & Rapport Final).

Croise les fichiers de prédictions (`results/predictions/{task}_*_A_official.csv`)
et les métriques (`results/metrics/{task}_*_A_official.json`) avec les splits nettoyés
(`src.data.get_split`) pour extraire :
1. Les métriques globales avec intervalles de confiance bootstrap à 95 % (`src.evaluate`).
2. Les matrices de confusion et performances par classe et par événement (`event_name`).
3. La comparaison inter-modèles (accords/désaccords : ex. où BERTweet corrige TF-IDF,
   où TF-IDF a raison et BERTweet se trompe, et où tous les modèles échouent).
4. Les exemples d'erreurs avec le texte brut et le chemin de l'image (`splits/*.csv`).

Usage :
    python -m src.error_analysis [informative|humanitarian]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import LABELS, get_split
from src.evaluate import compute_metrics, evaluate_file

PRED_DIR = ROOT / "results" / "predictions"
METRIC_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"


def ensure_metrics(pred_path: Path) -> dict:
    """Charge le JSON de métriques officiel ou le génère via evaluate_file s'il manque."""
    metric_path = METRIC_DIR / f"{pred_path.stem}.json"
    if not metric_path.exists():
        metric_path = evaluate_file(pred_path)
    return json.loads(metric_path.read_text(encoding="utf-8"))


def analyze_single_model(pred_path: Path, test_df: pd.DataFrame, task: str, top_k: int = 8) -> tuple[pd.DataFrame, dict]:
    """Analyse détaillée d'un fichier de prédictions sur le split test nettoyé."""
    metrics = ensure_metrics(pred_path)
    pred_df = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
    labels = list(LABELS[task])
    proba_cols = [f"proba_{label}" for label in labels]

    merged = test_df[["tweet_id", "image_id", "event_name", "tweet_text", "image", "label"]].merge(
        pred_df[["tweet_id", "image_id", "y_true", "y_pred"] + proba_cols],
        on=["tweet_id", "image_id"],
        how="left",
        validate="one_to_one",
    )
    for col in proba_cols:
        merged[col] = merged[col].astype(float)

    probas = merged[proba_cols].to_numpy(dtype=float)
    merged["pred_confidence"] = probas[np.arange(len(merged)), [labels.index(y) for y in merged["y_pred"]]]
    merged["is_error"] = merged["y_true"] != merged["y_pred"]

    ci = metrics["bootstrap"]["intervals"]["f1_macro"]
    n_err = int(merged["is_error"].sum())

    print("=" * 84)
    print(f"MODÈLE : {pred_path.name}")
    print(
        f"F1 macro : {metrics['f1_macro']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}] | "
        f"F1 pondéré : {metrics['f1_weighted']:.4f} | Accuracy : {metrics['accuracy']:.4f} | "
        f"Erreurs : {n_err}/{len(merged)} ({100 * n_err / len(merged):.1f}%)"
    )
    print("=" * 84)

    per_cls = pd.DataFrame(metrics["per_class"]).T
    per_cls["support"] = per_cls["support"].astype(int)
    print("\n--- Métriques par classe (src/evaluate.py) ---")
    print(per_cls.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- Matrice de confusion (lignes = y_true, colonnes = y_pred) ---")
    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    print(cm_df.to_string())

    print("\n--- Performance par événement (F1 macro & taux d'erreur) ---")
    event_rows = []
    for ev, grp in merged.groupby("event_name"):
        ev_metrics = compute_metrics(grp["y_true"], grp["y_pred"], task, n_bootstrap=100)
        ev_err = grp["is_error"].mean() * 100
        event_rows.append(
            {
                "event_name": ev,
                "n_test": len(grp),
                "f1_macro": round(ev_metrics["f1_macro"], 4),
                "f1_weighted": round(ev_metrics["f1_weighted"], 4),
                "error_rate_%": round(ev_err, 2),
            }
        )
    ev_df = pd.DataFrame(event_rows).sort_values("f1_macro")
    print(ev_df.to_string(index=False))

    errors = merged[merged["is_error"]].sort_values("pred_confidence", ascending=False)
    print(f"\n--- Top {top_k} erreurs les plus confiantes ---")
    for _, row in errors.head(top_k).iterrows():
        print(
            f"[{row['event_name']}] tweet_id={row['tweet_id']} image_id={row['image_id']} | "
            f"vrai={row['y_true']} -> prédit={row['y_pred']} (conf={row['pred_confidence']:.4f})\n"
            f"  Texte : {row['tweet_text']}\n"
            f"  Image : {row['image']}"
        )

    return merged, metrics


def compare_models(loaded_models: dict[str, pd.DataFrame], task: str):
    """Compare les erreurs entre modèles (accords/désaccords) et sauvegarde les tableaux."""
    if len(loaded_models) < 2:
        return

    names = list(loaded_models.keys())
    base_name = names[0]
    comp = loaded_models[base_name][["tweet_id", "image_id", "event_name", "tweet_text", "image", "y_true"]].copy()

    for name, df in loaded_models.items():
        comp[f"pred_{name}"] = df["y_pred"].values
        comp[f"conf_{name}"] = df["pred_confidence"].values
        comp[f"err_{name}"] = df["is_error"].values

    err_cols = [f"err_{n}" for n in names]
    comp["n_models_wrong"] = comp[err_cols].sum(axis=1)

    print("\n" + "=" * 84)
    print(f"COMPARAISON INTER-MODÈLES — {task} ({', '.join(names)})")
    print("=" * 84)
    all_right = comp[comp["n_models_wrong"] == 0]
    all_wrong = comp[comp["n_models_wrong"] == len(names)]
    print(f"- Tweets correctement classés par TOUS les modèles : {len(all_right)}/{len(comp)} ({100 * len(all_right) / len(comp):.1f}%)")
    print(f"- Tweets mal classés par TOUS les modèles ({len(names)}/{len(names)}) : {len(all_wrong)}/{len(comp)} ({100 * len(all_wrong) / len(comp):.1f}%)")
    for name in names:
        only_right = comp[(~comp[f"err_{name}"]) & (comp["n_models_wrong"] == len(names) - 1)]
        print(f"- Tweets où SEUL {name} a raison : {len(only_right)} ({100 * len(only_right) / len(comp):.1f}%)")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES_DIR / f"error_analysis_{task}_comparison.csv"
    interesting = comp[comp["n_models_wrong"] > 0].sort_values(
        by=["n_models_wrong", f"conf_{base_name}"], ascending=[False, False]
    )
    interesting.to_csv(out_csv, index=False)
    print(f"\nTable comparative des erreurs ({len(interesting)} tweets) exportée dans : {out_csv.relative_to(ROOT)}")


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "informative"
    test_df = get_split(task, "A", "official", text_only=True)["test"]
    pred_files = sorted(PRED_DIR.glob(f"{task}_*_A_official.csv"))
    if not pred_files:
        print(f"Aucun fichier de prédiction trouvé pour {task} dans {PRED_DIR}")
        return

    loaded = {}
    for path in pred_files:
        model_tag = path.stem.replace(f"{task}_", "").replace("_A_official", "")
        loaded[model_tag], _ = analyze_single_model(path, test_df, task)

    compare_models(loaded, task)


if __name__ == "__main__":
    main()
