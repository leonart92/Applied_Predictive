"""Figures et tableaux de matrices depuis les métriques déjà écrites.

Aucun modèle n'est chargé ni relancé : le script relit les JSON de
results/metrics/ (qui contiennent déjà la matrice de confusion et les
intervalles bootstrap) et en tire les PNG et CSV correspondants. Le style suit
les figures du baseline TF-IDF. Un fichier de métriques absent est signalé et
ignoré, pour qu'un modèle dont seuls certains folds existent reste traitable.

Usage:
    python src/plot_metrics.py clip_fusion vit
"""
import argparse
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .data import LABELS, ROOT
    from .evaluate import MODELS
else:
    from data import LABELS, ROOT
    from evaluate import MODELS

METRIC_DIR = ROOT / "results" / "metrics"
FIGURE_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "results" / "tables"
TARGET_EVENTS = ("hurricane_maria", "california_wildfires")
# Où chaque régime écrit ses métriques : la racine porte le run officiel du
# protocole, in_domain/ et all_target/ ses deux populations supplémentaires.
REGIMES = {"in_domain": "in_domain", "out_of_domain": "", "all_target": "all_target"}
REGIME_TITLES = {"in_domain": "modèle in-domain", "out_of_domain": "événement inédit",
                 "all_target": "événement inédit · toutes les paires"}


def read_metrics(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def confusion_figure(metrics, title, destination):
    """Matrice de confusion au format des figures existantes du dépôt."""
    labels = metrics["labels"]
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6) if len(labels) == 2 else (11, 9))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, shrink=.8, label="Nombre de paires")
    ticks = [textwrap.fill(label.replace("_", " "), 20) for label in labels]
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=ticks, yticklabels=ticks,
           xlabel="Classe prédite", ylabel="Classe réelle", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def gap_figure(frame, task, model, destination):
    """F1 macro in-domain contre événement inédit, avec les intervalles bootstrap."""
    events = sorted(frame["event_name"].unique())
    fig, ax = plt.subplots(figsize=(1.9 + 2.1 * len(events), 5))
    width = .36
    positions = np.arange(len(events))
    colors = {"in_domain": "#4878a8", "out_of_domain": "#c86050"}
    for offset, regime in ((-width / 2, "in_domain"), (width / 2, "out_of_domain")):
        rows = [frame[(frame["event_name"] == event) & (frame["regime"] == regime)] for event in events]
        values = [float(row["f1_macro"].iloc[0]) if len(row) else np.nan for row in rows]
        lower = [value - float(row["f1_macro_lower"].iloc[0]) if len(row) else 0
                 for value, row in zip(values, rows)]
        upper = [float(row["f1_macro_upper"].iloc[0]) - value if len(row) else 0
                 for value, row in zip(values, rows)]
        bars = ax.bar(positions + offset, values, width, label=REGIME_TITLES[regime],
                      color=colors[regime], yerr=[lower, upper], capsize=4, ecolor="#404040")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    for position, event in zip(positions, events):
        group = frame[frame["event_name"] == event]
        inside = group[group["regime"] == "in_domain"]["f1_macro"]
        outside = group[group["regime"] == "out_of_domain"]["f1_macro"]
        if len(inside) and len(outside):
            gap = float(inside.iloc[0]) - float(outside.iloc[0])
            # Au-dessus du groupe, hors de portée des barres et de leurs intervalles.
            ax.annotate(f"écart {gap:+.3f}", (position, float(group["f1_macro_upper"].max()) + .09),
                        ha="center", fontsize=10, color="#303030")
    ax.set(xticks=positions, xticklabels=[event.replace("_", " ") for event in events],
           ylabel="F1 macro", ylim=(0, 1.25),
           title=f"{model} · {task} · protocole B\nmêmes paires test, modèle entraîné avec ou sans l'événement")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def export(model, events=TARGET_EVENTS):
    """Toutes les figures et tables d'un modèle ; rend la liste des fichiers écrits."""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model!r}; expected one of {MODELS}")
    written, rows = [], []
    for task in LABELS:
        official = read_metrics(METRIC_DIR / f"{task}_{model}_A_official.json")
        if official is None:
            print(f"  métriques A absentes pour {task}/{model}")
        else:
            written.append(confusion_figure(
                official, f"{model} · {task} · protocole A (test officiel)",
                FIGURE_DIR / f"{task}_{model}_A_official_confusion_matrix.png"))
        for event in events:
            for regime, folder in REGIMES.items():
                metrics = read_metrics(METRIC_DIR / folder / f"{task}_{model}_B_{event}.json")
                if metrics is None:
                    print(f"  métriques {regime} absentes pour {task}/{model}/{event}")
                    continue
                stem = f"{task}_{model}_B_{event}"
                written.append(confusion_figure(
                    metrics, f"{model} · {task} · {event}\n{REGIME_TITLES[regime]}",
                    FIGURE_DIR / f"{model}_B" / regime / f"{stem}_confusion_matrix.png"))
                table = TABLE_DIR / f"{model}_B_confusion_matrices" / regime / f"{stem}.csv"
                table.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(metrics["confusion_matrix"],
                             index=pd.Index(metrics["labels"], name="y_true"),
                             columns=pd.Index(metrics["labels"], name="y_pred")).to_csv(table)
                written.append(table)
                intervals = metrics["bootstrap"]["intervals"]
                rows.append(dict(task=task, model=model, event_name=event, regime=regime,
                                 n_samples=metrics["n_samples"],
                                 f1_macro=metrics["f1_macro"],
                                 f1_macro_lower=intervals["f1_macro"]["lower"],
                                 f1_macro_upper=intervals["f1_macro"]["upper"],
                                 f1_weighted=metrics["f1_weighted"], accuracy=metrics["accuracy"]))
    if rows:
        frame = pd.DataFrame(rows)
        summary = TABLE_DIR / f"{model}_B_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(summary, index=False)
        written.append(summary)
        for task in frame["task"].unique():
            written.append(gap_figure(frame[frame["task"] == task], task, model,
                                      FIGURE_DIR / f"{model}_B" / f"{task}_{model}_B_generalization_gap.png"))
    for path in written:
        print(f"écrit · {path.relative_to(ROOT)}")
    return written


def main(models):
    for model in models:
        print(f"=== {model} ===")
        export(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=MODELS)
    main(parser.parse_args().models)
