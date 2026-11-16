"""Shared cleaned-split loader and protocols from PROJET.md §§3–4.

    from src.data import get_split
    splits = get_split('informative', 'A', 'official', text_only=True)
    train, dev, test = (splits[name] for name in ('train', 'dev', 'test'))

B also returns ``test_all_target``: all official rows of the unseen event,
for the supplementary evaluation only. Compare A and B on B's ``test``.
All DataFrames contain raw text and string identifiers. Images are paths
relative to the repository root. No model-specific normalization is applied.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LABELS = {
    'informative': ('informative', 'not_informative'),
    'humanitarian': ('affected_individuals', 'infrastructure_and_utility_damage',
                     'rescue_volunteering_or_donation_effort',
                     'other_relevant_information', 'not_humanitarian'),
}
EVENTS = ('hurricane_harvey', 'hurricane_irma', 'hurricane_maria',
          'mexico_earthquake', 'iraq_iran_earthquake',
          'california_wildfires', 'srilanka_floods')
SPLITS = ('train', 'dev', 'test')
REQUIRED_COLUMNS = ('event_name', 'tweet_id', 'image_id', 'tweet_text', 'image', 'label')


def load_split(task, split, *, split_dir=None):
    """Read one cleaned CSV without dropping rows or coercing text/IDs."""
    if task not in LABELS:
        raise ValueError(f'Unknown task: {task!r}; expected {tuple(LABELS)}')
    if split not in SPLITS:
        raise ValueError(f'Unknown split: {split!r}; expected {SPLITS}')
    folder = Path(split_dir) if split_dir is not None else ROOT / 'splits'
    frame = pd.read_csv(folder / f'{task}_{split}.csv', dtype=str,
                        keep_default_na=False, na_filter=False)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError(f'Empty split: {task}/{split}')
    for col in ('tweet_id', 'image_id', 'event_name', 'label', 'image'):
        if frame[col].str.strip().eq('').any():
            raise ValueError(f'Empty required value: {col}')
    if not frame['label'].isin(LABELS[task]).all():
        raise ValueError(f'Unexpected labels in {task}/{split}')
    if not frame['event_name'].isin(EVENTS).all():
        raise ValueError(f'Unexpected events in {task}/{split}')
    # Preparation owns image validation. Loading splits never silently removes
    # a missing image or changes a previously agreed evaluation population.
    return frame


def _text_train(frame):
    """Stable first-image selection, only after checking text/label consistency."""
    if (frame.groupby('tweet_id')[['tweet_text', 'label', 'event_name']]
            .nunique(dropna=False).gt(1).any().any()):
        raise ValueError('A tweet has conflicting text, label or event; cannot deduplicate')
    return frame.drop_duplicates('tweet_id', keep='first').reset_index(drop=True)


def get_split(task, protocol, fold, *, text_only=False, split_dir=None):
    """Return protocol DataFrames; never shuffle, normalize or write CSVs.

    A/official: the official train, dev and test sets.
    B/event_name: train and dev exclude the target; test is official test
    restricted to target. ``test_all_target`` concatenates target rows from
    train/dev/test, in that order; it is supplementary, not the A/B comparison.

    Set text_only=True for TF-IDF/BERTweet: deduplicate ONLY the returned train
    by tweet_id. Image/fusion models use the default False. Dev/test and the
    supplementary B population always keep their original pair multiplicity.
    """
    if protocol not in ('A', 'B'):
        raise ValueError('protocol must be A or B')
    if protocol == 'A' and fold != 'official':
        raise ValueError('Protocol A requires fold=official')
    if protocol == 'B' and fold not in EVENTS:
        raise ValueError(f'Protocol B requires a target event from {EVENTS}')
    official = {s: load_split(task, s, split_dir=split_dir) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if set(official[a]['tweet_id']) & set(official[b]['tweet_id']):
                raise ValueError(f'Tweet leakage between official {a} and {b}')
    if protocol == 'A':
        result = official
    else:
        result = {
            'train': official['train'].loc[official['train']['event_name'] != fold].copy(),
            'dev': official['dev'].loc[official['dev']['event_name'] != fold].copy(),
            'test': official['test'].loc[official['test']['event_name'] == fold].copy(),
            'test_all_target': pd.concat([
                official[s].loc[official[s]['event_name'] == fold] for s in SPLITS
            ], ignore_index=True),
        }
    if any(frame.empty for frame in result.values()):
        raise ValueError('The requested protocol contains an empty population')
    if text_only:
        result['train'] = _text_train(result['train'])
    return {key: frame.reset_index(drop=True) for key, frame in result.items()}
