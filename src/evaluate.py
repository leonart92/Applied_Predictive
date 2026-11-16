"""Single evaluation implementation for all models (PROJET.md §6).

Evaluate saved predictions without running a model:
    python -m src.evaluate results/predictions/informative_tfidf_A_official.csv

Probabilities must follow data.LABELS[task] order when using write_predictions.
Scores use [0, 1], including the 95% bootstrap confidence intervals.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

if __package__:
    from .data import EVENTS, LABELS, ROOT, SEED, get_split
else:
    from data import EVENTS, LABELS, ROOT, SEED, get_split

MODELS = ('tfidf', 'bertweet', 'vit', 'clip_fusion', 'llm')
ID_COLUMNS = ('tweet_id', 'image_id', 'event_name')


def prediction_name(task, model, protocol, fold):
    """Validate metadata and return the filename required by PROJET.md."""
    if task not in LABELS or model not in MODELS:
        raise ValueError('Unknown task or model')
    if not ((protocol == 'A' and fold == 'official')
            or (protocol == 'B' and fold in EVENTS)):
        raise ValueError('Expected A/official or B/event_name')
    return f'{task}_{model}_{protocol}_{fold}.csv'


def _metadata(path):
    match = re.fullmatch(
        r'(informative|humanitarian)_(tfidf|bertweet|vit|clip_fusion|llm)_(A|B)_(.+)',
        Path(path).stem)
    if not match or Path(path).suffix != '.csv':
        raise ValueError('Prediction filename must be {task}_{model}_{protocol}_{fold}.csv')
    task, model, protocol, fold = match.groups()
    prediction_name(task, model, protocol, fold)
    return dict(task=task, model=model, protocol=protocol, fold=fold)


def _validate_frame(frame, task):
    labels = LABELS[task]
    probability_columns = [f'proba_{label}' for label in labels]
    required = list(ID_COLUMNS) + ['y_true', 'y_pred'] + probability_columns
    if len(frame.columns) != len(set(frame.columns)) or set(frame.columns) != set(required):
        raise ValueError(f'Prediction columns must be exactly {required}')
    if frame.empty:
        raise ValueError('Cannot evaluate an empty prediction file')
    for col in list(ID_COLUMNS) + ['y_true', 'y_pred']:
        if not frame[col].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'{col} must contain nonempty strings')
    if frame.duplicated(['tweet_id', 'image_id']).any():
        raise ValueError('Duplicate prediction pair')
    if not frame['event_name'].isin(EVENTS).all():
        raise ValueError('Unknown event in predictions')
    for col in ('y_true', 'y_pred'):
        if not frame[col].isin(labels).all():
            raise ValueError(f'Unknown label in {col}; invalid LLM answers must be resolved explicitly')
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('Probabilities must be finite values in [0, 1]')
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError('Class probabilities must sum to one per row')
    return frame[required].copy()


def _validate_population(frame, metadata, subset, split_dir):
    if subset not in ('test', 'all_target'):
        raise ValueError('subset must be test or all_target')
    if subset == 'all_target' and metadata['protocol'] != 'B':
        raise ValueError('all_target is supplementary and only available for B')
    splits = get_split(metadata['task'], metadata['protocol'], metadata['fold'], split_dir=split_dir)
    expected = splits['test' if subset == 'test' else 'test_all_target']
    cols = list(ID_COLUMNS) + ['y_true']
    expected = expected.rename(columns={'label': 'y_true'})[cols]
    actual = frame[cols]
    # Allow reordered prediction files, but require the identical evaluation set
    # and exact true labels. Never align or truncate mismatched arrays silently.
    key = ['tweet_id', 'image_id']
    if not (actual.sort_values(key).reset_index(drop=True)
            .equals(expected.sort_values(key).reset_index(drop=True))):
        raise ValueError('Prediction population/true labels differ from the requested cleaned split')


def write_predictions(frame, y_pred, probabilities, *, task, model, protocol, fold,
                      output_dir=None, subset='test', split_dir=None):
    """Save aligned predictions; no training or automatic probability invention.

    ``frame`` is the corresponding get_split population. ``probabilities`` is
    shape (n_rows, n_classes), in LABELS[task] order. Reorder predict_proba
    columns from a model's classes_ before calling. For a LLM, the caller owns
    the documented probability/output parsing policy; missing scores are errors.
    """
    name = prediction_name(task, model, protocol, fold)
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (len(frame), len(LABELS[task])):
        raise ValueError('Probability matrix has the wrong shape')
    predicted = np.asarray(y_pred)
    if predicted.shape != (len(frame),):
        raise ValueError('y_pred must have one label per row')
    result = frame[list(ID_COLUMNS)].reset_index(drop=True).copy()
    result['y_true'] = frame['label'].to_numpy()
    result['y_pred'] = predicted
    for i, label in enumerate(LABELS[task]):
        result[f'proba_{label}'] = probs[:, i]
    result = _validate_frame(result, task)
    metadata = dict(task=task, model=model, protocol=protocol, fold=fold)
    _validate_population(result, metadata, subset, split_dir)
    folder = Path(output_dir) if output_dir is not None else ROOT / 'results/predictions'
    if subset == 'all_target':
        folder /= 'all_target'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    result.to_csv(path, index=False)
    return path


def _scores_from_confusion(matrix):
    """Bootstrap scores with a fixed task class universe, even if a draw omits a class."""
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    denominators = support + predicted
    per_class_f1 = np.divide(2 * tp, denominators,
                             out=np.zeros(len(tp), dtype=float), where=denominators != 0)
    n = matrix.sum()
    return (float(per_class_f1.mean()), float(np.dot(per_class_f1, support) / n),
            float(tp.sum() / n))


def compute_metrics(y_true, y_pred, task, *, n_bootstrap=1000, seed=SEED):
    """Point metrics and paired, ordinary (not stratified) percentile bootstrap.

    Each draw resamples n aligned (truth, prediction) pairs with replacement.
    The fixed task labels and zero_division=0 are used in every draw. This keeps
    the macro denominator fixed when a rare class is missing from a subset.
    """
    if task not in LABELS:
        raise ValueError('Unknown task')
    labels = list(LABELS[task])
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or predicted.shape != truth.shape or len(truth) == 0:
        raise ValueError('Expected nonempty, aligned 1D truth and prediction arrays')
    if not np.isin(truth, labels).all() or not np.isin(predicted, labels).all():
        raise ValueError('Labels must belong to the requested task')
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError('n_bootstrap must be a positive integer')
    matrix = confusion_matrix(truth, predicted, labels=labels)
    report = classification_report(truth, predicted, labels=labels, target_names=labels,
                                   output_dict=True, zero_division=0)
    rng = np.random.default_rng(seed)
    indices = {label: i for i, label in enumerate(labels)}
    encoded = np.array([indices[a] * len(labels) + indices[b] for a, b in zip(truth, predicted)])
    bootstrap = np.empty((n_bootstrap, 3), dtype=float)
    for i in range(n_bootstrap):
        sample = encoded[rng.integers(0, len(encoded), size=len(encoded))]
        sample_matrix = np.bincount(sample, minlength=len(labels)**2).reshape(len(labels), len(labels))
        bootstrap[i] = _scores_from_confusion(sample_matrix)
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    names = ('f1_macro', 'f1_weighted', 'accuracy')
    return {
        'task': task, 'n_samples': len(truth), 'labels': labels,
        'f1_macro': float(f1_score(truth, predicted, labels=labels, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(truth, predicted, labels=labels, average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(truth, predicted)),
        'per_class': {label: dict(report[label], support=int(report[label]['support'])) for label in labels},
        'confusion_matrix': matrix.tolist(),
        'confusion_matrix_axes': {'rows': 'y_true', 'columns': 'y_pred'},
        'bootstrap': {'n_resamples': n_bootstrap, 'seed': seed, 'confidence_level': 0.95,
                      'method': 'paired ordinary percentile bootstrap over tweet-image rows',
                      'fixed_labels': True, 'zero_division': 0,
                      'intervals': {name: {'lower': float(intervals[0, i]), 'upper': float(intervals[1, i])}
                                    for i, name in enumerate(names)}},
    }


def evaluate_file(prediction_path, *, output_dir=None, subset='test', split_dir=None,
                  n_bootstrap=1000, seed=SEED):
    """Read a run CSV, validate against cleaned splits and save same-stem JSON.

    Supplementary B/all_target results are put in an all_target/ subdirectory
    to prevent overwriting the official-test comparison. They still use the
    prescribed run filename; subset is explicit in JSON and the CLI argument.
    """
    path = Path(prediction_path)
    metadata = _metadata(path)
    # Reading the header separately catches duplicate columns before pandas can
    # silently rename them. No model or inference code is called here.
    import csv
    with path.open(encoding='utf-8', newline='') as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV columns')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    frame = _validate_frame(frame, metadata['task'])
    _validate_population(frame, metadata, subset, split_dir)
    result = compute_metrics(frame['y_true'], frame['y_pred'], metadata['task'],
                             n_bootstrap=n_bootstrap, seed=seed)
    try:
        prediction_file = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        prediction_file = path.as_posix()
    result.update(metadata, subset=subset, prediction_file=prediction_file)
    folder = Path(output_dir) if output_dir is not None else ROOT / 'results/metrics'
    if subset == 'all_target':
        folder /= 'all_target'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f'{path.stem}.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                           encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--split-dir', type=Path)
    parser.add_argument('--subset', choices=('test', 'all_target'), default='test')
    args = parser.parse_args()
    path = evaluate_file(args.predictions, output_dir=args.output_dir,
                         split_dir=args.split_dir, subset=args.subset)
    print(f'Metrics written to {path}')


if __name__ == '__main__':
    main()
