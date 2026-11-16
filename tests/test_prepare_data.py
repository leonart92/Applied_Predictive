"""Synthetic end-to-end checks; never touch the real dataset."""
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    'prepare_data', Path(__file__).resolve().parents[1] / 'src/prepare_data.py')
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    def test_image_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (16, 16)).save(root / 'valid.jpg')
            (root / 'corrupt.jpg').write_bytes(b'not an image')
            (root / 'truncated.jpg').write_bytes((root / 'valid.jpg').read_bytes()[:-20])
            for name, expected in [('valid.jpg', 'valid'), ('missing.jpg', 'missing'),
                                   ('corrupt.jpg', 'corrupted'), ('truncated.jpg', 'corrupted')]:
                with self.subTest(name=name):
                    self.assertEqual(prep.check_image(root / name)[0], expected)

    def test_pipeline_preserves_text_multiplicity_and_accounts_for_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / 'data/crisismmd_datasplit_agreed_label'
            sources.mkdir(parents=True)
            images = root / 'data/CrisisMMD_v2.0/data_image/event'
            images.mkdir(parents=True)
            for name in ('one.jpg', 'two.jpg'):
                Image.new('RGB', (8, 8)).save(images / name)
            (images / 'bad.jpg').write_bytes(b'broken')
            for task, labels in prep.TASKS.items():
                for split in prep.SPLITS:
                    rows = []
                    for i, label in enumerate(labels):
                        rows.append(dict(event_name='event', tweet_id='000123',
                                         image_id=f'000123_{i}',
                                         tweet_text='"Harvey death toll now stands at 70" via FOX NEWS — RT @x: é â€™, https://t.co/a  ',
                                         image='data_image/event/one.jpg' if i == 0 else 'data_image/event/two.jpg',
                                         label=label, label_text=label, label_image=label,
                                         label_text_image='Positive'))
                    # An exact duplicate must remain, as must distinct images of a tweet.
                    rows.append(dict(rows[0]))
                    rows.append(dict(rows[0], image='data_image/event/missing.jpg'))
                    rows.append(dict(rows[0], image='data_image/event/bad.jpg'))
                    path = sources / f'task_{task}_text_img_agreed_lab_{split}.tsv'
                    with path.open('w', encoding='utf-8', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=prep.COLUMNS, delimiter='\t',
                                                quotechar=None, quoting=csv.QUOTE_NONE)
                        writer.writeheader()
                        writer.writerows(rows)
            with patch.object(prep, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                prep.main()
            for task, labels in prep.TASKS.items():
                for split in prep.SPLITS:
                    output = prep.read_table(root / f'splits/{task}_{split}.csv')
                    self.assertEqual(len(output), len(labels) + 1)
                    self.assertEqual(output[0], output[-1])
                    self.assertEqual(output[0]['tweet_id'], '000123')
                    self.assertEqual(
                        output[0]['tweet_text'],
                        '"Harvey death toll now stands at 70" via FOX NEWS — RT @x: é â€™, https://t.co/a  ',
                    )
                    self.assertEqual(prep.multi_image_tweets(output), 1)
                    self.assertEqual(prep.duplicates(output)[0], 1)
            with (root / 'results/tables/preparation_summary.csv').open() as handle:
                for row in csv.DictReader(handle):
                    self.assertEqual(row['missing_images'], '1')
                    self.assertEqual(row['corrupted_images'], '1')
                    self.assertEqual(row['removed_rows'], '2')
            report = json.loads((root / 'results/tables/validation_report.json').read_text())
            self.assertEqual(report['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
