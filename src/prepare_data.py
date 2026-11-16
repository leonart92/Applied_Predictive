"""Prepare the six official CrisisMMD agreed-label splits without text changes.

Run from the repository root: python src/prepare_data.py
Only missing files and images rejected by Pillow's verify/load are excluded.
All identifiers are read as strings; row order and multiplicity are preserved.
"""
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'informative': ('informative', 'not_informative'),
    'humanitarian': (
        'affected_individuals', 'infrastructure_and_utility_damage',
        'rescue_volunteering_or_donation_effort', 'other_relevant_information',
        'not_humanitarian',
    ),
}
SPLITS = ('train', 'dev', 'test')
COLUMNS = ['event_name', 'tweet_id', 'image_id', 'tweet_text', 'image',
           'label', 'label_text', 'label_image', 'label_text_image']


def require(condition, message):
    """Validation remains active even when Python is run with -O."""
    if not condition:
        raise ValueError(message)


def read_table(path, delimiter=','):
    quoting = csv.QUOTE_NONE if delimiter == '\t' else csv.QUOTE_MINIMAL
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter=delimiter, quoting=quoting)
        require(reader.fieldnames == COLUMNS, f'Unexpected columns: {path}')
        rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values())
                for row in rows), f'Malformed rows: {path}')
    return rows


def write_table(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def check_image(path):
    """Check structure, then reopen and fully decode pixels (no truncated mode)."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
    except FileNotFoundError as exc:
        return 'missing', str(exc)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        # A resource protection limit is not evidence of corruption.
        raise
    except (OSError, SyntaxError) as exc:
        if isinstance(exc, PermissionError):
            raise
        return 'corrupted', f'{type(exc).__name__}: {exc}'
    return 'valid', ''


def duplicates(rows):
    counts = Counter((r['tweet_id'], r['image_id']) for r in rows)
    return sum(n - 1 for n in counts.values()), counts


def multi_image_tweets(rows):
    images = {}
    for row in rows:
        images.setdefault(row['tweet_id'], set()).add(row['image_id'])
    return sum(len(ids) > 1 for ids in images.values())


def main():
    out = ROOT / 'splits'
    tables = ROOT / 'results/tables'
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    summary, exclusions, counts, duplicate_rows = [], [], [], []
    cache, prepared, provenance = {}, {}, {}
    for task, classes in TASKS.items():
        for split in SPLITS:
            source = ROOT / 'data/crisismmd_datasplit_agreed_label' / (
                f'task_{task}_text_img_agreed_lab_{split}.tsv')
            original = read_table(source, '\t')
            kept = []
            removed = Counter()
            for line, row in enumerate(original, start=2):
                for col in ('event_name', 'tweet_id', 'image_id', 'label'):
                    require(row[col].strip(), f'Empty {col}: {source}:{line}')
                require(row['label'] in classes, f'Unexpected label: {source}:{line}')
                # The local README defines label as a task label. Agreement is
                # checked row by row, rather than inferred from column names.
                require(row['label'] == row['label_text'] == row['label_image'],
                        f'Disagreement between task labels: {source}:{line}')
                require(row['label_text_image'] == 'Positive',
                        f'Unexpected agreement indicator: {source}:{line}')
                relative = Path(row['image'])
                require(not relative.is_absolute() and '..' not in relative.parts
                        and relative.parts[0] == 'data_image',
                        f'Unexpected image path: {source}:{line}')
                image = Path('data/CrisisMMD_v2.0') / relative
                if image.as_posix() not in cache:
                    cache[image.as_posix()] = check_image(ROOT / image)
                status, detail = cache[image.as_posix()]
                if status != 'valid':
                    removed[status] += 1
                    exclusions.append(dict(task=task, split=split, source_line=line,
                                           tweet_id=row['tweet_id'], image_id=row['image_id'],
                                           image=image.as_posix(), reason=status, detail=detail))
                else:
                    kept.append(dict(row, image=image.as_posix()))
            target = out / f'{task}_{split}.csv'
            write_table(target, kept, COLUMNS)
            reloaded = read_table(target)
            require(reloaded == kept, f'CSV round-trip failed: {target}')
            # Exact ordered comparison proves no deduplication, text changes,
            # label changes, or loss of multiple images, beyond logged exclusions.
            expected = [dict(row, image=(Path('data/CrisisMMD_v2.0') / row['image']).as_posix())
                        for row in original
                        if cache[(Path('data/CrisisMMD_v2.0') / row['image']).as_posix()][0] == 'valid']
            require(reloaded == expected, f'Rows changed or lost: {target}')
            require(set(r['label'] for r in reloaded) == set(classes),
                    f'Missing expected classes: {target}')
            # Reopen each retained unique image after CSV serialization.
            for image in sorted(set(r['image'] for r in reloaded)):
                status, detail = check_image(ROOT / image)
                require(status == 'valid', f'Output image invalid: {image}: {detail}')
            excess, pair_counts = duplicates(reloaded)
            for (tweet, image_id), n in sorted(pair_counts.items()):
                if n > 1:
                    duplicate_rows.append(dict(task=task, split=split, tweet_id=tweet,
                                               image_id=image_id, occurrences=n))
            summary.append(dict(task=task, split=split, initial_rows=len(original),
                                retained_rows=len(reloaded), missing_images=removed['missing'],
                                corrupted_images=removed['corrupted'],
                                removed_rows=sum(removed.values()),
                                duplicate_pair_excess=excess,
                                initial_multi_image_tweets=multi_image_tweets(original),
                                retained_multi_image_tweets=multi_image_tweets(reloaded)))
            require(len(original) == len(reloaded) + sum(removed.values()),
                    f'Accounting mismatch: {target}')
            event_counts = Counter((r['event_name'], r['label']) for r in reloaded)
            for event in sorted(set(r['event_name'] for r in original)):
                for label in classes:
                    counts.append(dict(task=task, split=split, event_name=event,
                                       label=label, count=event_counts[event, label]))
            require(sum(c['count'] for c in counts if c['task'] == task
                        and c['split'] == split) == len(reloaded), 'Count mismatch')
            prepared[task, split] = reloaded
            provenance[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
            provenance[target.relative_to(ROOT).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
            print(f'{task:13} {split:5}: {len(original):5} -> {len(reloaded):5} '
                  f'(missing={removed["missing"]}, corrupted={removed["corrupted"]}, '
                  f'duplicate excess={excess})', flush=True)
    require(all((out / f'{task}_{split}.csv').is_file()
                for task in TASKS for split in SPLITS), 'Missing output CSV')
    write_table(tables / 'preparation_summary.csv', summary, list(summary[0]))
    write_table(tables / 'excluded_images.csv', exclusions,
                ['task', 'split', 'source_line', 'tweet_id', 'image_id', 'image', 'reason', 'detail'])
    write_table(tables / 'duplicate_pairs.csv', duplicate_rows,
                ['task', 'split', 'tweet_id', 'image_id', 'occurrences'])
    write_table(tables / 'event_class_counts.csv', counts,
                ['task', 'split', 'event_name', 'label', 'count'])
    overlaps = []
    for task in TASKS:
        for a, b in itertools.combinations(SPLITS, 2):
            ra, rb = prepared[task, a], prepared[task, b]
            overlaps.append(dict(task=task, split_a=a, split_b=b,
                                 common_tweet_ids=len(set(r['tweet_id'] for r in ra) & set(r['tweet_id'] for r in rb)),
                                 common_pairs=len(set((r['tweet_id'], r['image_id']) for r in ra)
                                                  & set((r['tweet_id'], r['image_id']) for r in rb))))
    write_table(tables / 'split_overlaps.csv', overlaps, list(overlaps[0]))
    report = dict(status='passed', pillow_version=PIL.__version__,
                  checks=['six_outputs', 'schema', 'required_fields', 'expected_classes',
                          'task_label_agreement', 'image_verify_and_full_decode',
                          'csv_round_trip', 'ordered_source_preservation',
                          'pair_duplicates_reported', 'multiple_images_preserved',
                          'counts_and_exclusions_reconciled'],
                  reference_comparison='separate audit: results/tables/reference_verification.json (not recomputed here)',
                  unique_source_images=len(cache),
                  unique_missing_images=sum(s == 'missing' for s, _ in cache.values()),
                  unique_corrupted_images=sum(s == 'corrupted' for s, _ in cache.values()),
                  sha256=provenance)
    (tables / 'validation_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Validation passed. Reports: results/tables/. Project references are audited separately.')


if __name__ == '__main__':
    main()
