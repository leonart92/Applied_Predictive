"""Image access, fusion features and run alignment on synthetic data.

No checkpoint is downloaded and no model is trained: the CLIP encoders are
replaced by injected functions, so the test suite stays offline and fast.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from src.clip_fusion import (build_features, clip_tables, ensure_table, probabilities,
                             text_key, write_ablation)
from src.data import EVENTS, LABELS, get_split
from src.evaluate import evaluate_file, write_predictions
from src.images import ImageDataset, open_rgb, unique_images

SHADES = {}


def shade(index):
    """A solid colour per image, so a decoded image identifies its own file."""
    return (10 + 7 * index) % 250


def red_channel(image):
    """Transform standing in for a processor: the red level of the first pixel."""
    return torch.tensor([float(image.getpixel((0, 0))[0])])


class SyntheticSplits(unittest.TestCase):
    """Six cleaned splits whose images exist on disk, one colour per image."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.splits = self.root / 'splits'
        self.splits.mkdir()
        self.images = self.root / 'data/images'
        self.images.mkdir(parents=True)
        counter = 0
        for task, labels in LABELS.items():
            for split in ('train', 'dev', 'test'):
                rows = []
                for event in EVENTS:
                    for i, label in enumerate(labels):
                        tweet = f'000_{event}_{split}_{i}'
                        rows.append(dict(tweet_id=tweet, image_id=f'{tweet}_0', event_name=event,
                                         tweet_text=f'tweet {task} {event} {split} {i}',
                                         image=f'data/images/{task}_{tweet}_0.jpg', label=label))
                if split == 'train':  # un tweet à deux images, jamais dédupliqué ici
                    extra = dict(rows[0], image_id=rows[0]['tweet_id'] + '_1',
                                 image=rows[0]['image'].replace('_0.jpg', '_1.jpg'))
                    rows.append(extra)
                for row in rows:
                    path = self.root / row['image']
                    if not path.exists():
                        SHADES[row['image']] = shade(counter)
                        Image.new('RGB', (8, 8), (SHADES[row['image']], 0, 0)).save(path, quality=100)
                        counter += 1
                pd.DataFrame(rows).to_csv(self.splits / f'{task}_{split}.csv', index=False)

    def tearDown(self):
        self.temp.cleanup()

    def frame(self, task='informative', split='test'):
        return get_split(task, 'A', 'official', split_dir=self.splits)[split]


class ImageAccessTests(SyntheticSplits):
    def test_dataset_follows_the_dataframe_row_order(self):
        frame = self.frame()
        dataset = ImageDataset(frame['image'], red_channel,
                               [0] * len(frame), root=self.root)
        self.assertEqual(len(dataset), len(frame))
        for position, image in enumerate(frame['image']):
            value = float(dataset[position]['pixel_values'][0])
            # JPEG is lossy, so the decoded shade is compared with a tolerance.
            self.assertAlmostEqual(value, SHADES[image], delta=6)
            self.assertEqual(dataset[position]['labels'], 0)

    def test_dataset_without_labels_and_mismatched_labels(self):
        frame = self.frame()
        self.assertNotIn('labels', ImageDataset(frame['image'], red_channel, root=self.root)[0])
        with self.assertRaises(ValueError):
            ImageDataset(frame['image'], red_channel, [0], root=self.root)

    def test_missing_and_unreadable_images_raise(self):
        with self.assertRaises(FileNotFoundError):
            open_rgb('data/images/absent.jpg', root=self.root)
        broken = self.root / 'data/images/broken.jpg'
        broken.write_bytes(b'not an image')
        with self.assertRaises(OSError):
            open_rgb('data/images/broken.jpg', root=self.root)

    def test_unique_images_are_sorted_and_deduplicated(self):
        splits = get_split('informative', 'A', 'official', split_dir=self.splits)
        paths = unique_images(splits['train'], splits['test'])
        self.assertEqual(paths, sorted(set(paths)))
        self.assertEqual(set(paths),
                         set(splits['train']['image']) | set(splits['test']['image']))


class FusionFeatureTests(SyntheticSplits):
    def tables(self, frame, text_size=3, image_size=2):
        texts = {text_key(text): np.full(text_size, i + 1, dtype=np.float32)
                 for i, text in enumerate(dict.fromkeys(frame['tweet_text']))}
        images = {image: np.full(image_size, -(i + 1), dtype=np.float32)
                  for i, image in enumerate(dict.fromkeys(frame['image']))}
        return texts, images

    def test_fusion_concatenates_text_then_image_in_row_order(self):
        frame = self.frame()
        texts, images = self.tables(frame)
        fusion = build_features(frame, texts, images)
        self.assertEqual(fusion.shape, (len(frame), 5))
        np.testing.assert_array_equal(fusion[:, :3], build_features(frame, texts, images, 'text'))
        np.testing.assert_array_equal(fusion[:, 3:], build_features(frame, texts, images, 'image'))
        for position, row in frame.iterrows():
            np.testing.assert_array_equal(fusion[position, :3], texts[text_key(row['tweet_text'])])
            np.testing.assert_array_equal(fusion[position, 3:], images[row['image']])
        with self.assertRaises(ValueError):
            build_features(frame, texts, images, 'texte')

    def test_cache_only_computes_missing_keys_and_round_trips(self):
        path = self.root / 'cache/text.npz'
        items = {'a': 'payload a', 'b': 'payload b'}
        calls = []

        def embed(payloads):
            calls.append(list(payloads))
            return np.arange(2 * len(payloads), dtype=np.float32).reshape(len(payloads), 2)

        first = ensure_table(path, items, embed)
        self.assertEqual(calls, [['payload a', 'payload b']])
        again = ensure_table(path, items, embed)
        self.assertEqual(len(calls), 1, 'a complete cache must not call the encoder again')
        np.testing.assert_array_equal(first['a'], again['a'])

        grown = ensure_table(path, dict(items, c='payload c'), embed)
        self.assertEqual(calls[-1], ['payload c'])
        self.assertEqual(sorted(grown), ['a', 'b', 'c'])
        # Only the requested keys come back, even though the cache keeps the rest.
        self.assertEqual(sorted(ensure_table(path, {'b': 'payload b'}, embed)), ['b'])

    def test_encoder_returning_the_wrong_count_is_rejected(self):
        with self.assertRaises(ValueError):
            ensure_table(self.root / 'cache/bad.npz', {'a': 'x', 'b': 'y'},
                         lambda payloads: np.zeros((1, 2), dtype=np.float32))

    def test_tables_cover_every_split_and_are_computed_once(self):
        calls = []

        def embed_texts(payloads):
            calls.append(('text', len(payloads)))
            return np.ones((len(payloads), 4), dtype=np.float32)

        def embed_images(payloads):
            calls.append(('image', len(payloads)))
            return np.ones((len(payloads), 4), dtype=np.float32)

        cache = self.root / 'cache'
        texts, images = clip_tables(cache_dir=cache, root=self.root, split_dir=self.splits,
                                    embedders=(embed_texts, embed_images))
        self.assertEqual(len(calls), 2)
        every_image = unique_images(*[get_split(task, 'A', 'official', split_dir=self.splits)[split]
                                      for task in LABELS for split in ('train', 'dev', 'test')])
        self.assertEqual(sorted(images), every_image)
        self.assertEqual(calls[1], ('image', len(every_image)))
        clip_tables(cache_dir=cache, root=self.root, split_dir=self.splits,
                    embedders=(embed_texts, embed_images))
        self.assertEqual(len(calls), 2, 'the second run must read the cache only')
        self.assertTrue((cache / 'clip_vit_b32_text.npz').is_file())
        self.assertIn(text_key(self.frame()['tweet_text'].iloc[0]), texts)


class ProbabilityOrderTests(unittest.TestCase):
    def test_columns_follow_label_order_not_classes_order(self):
        labels = list(LABELS['humanitarian'])
        features = np.eye(len(labels) * 2, dtype=float)
        targets = labels + labels
        classifier = LogisticRegression(max_iter=1000).fit(features, targets)
        # scikit-learn sorts classes_ alphabetically; LABELS does not.
        self.assertNotEqual(list(classifier.classes_), labels)
        ordered = probabilities(classifier, features, labels)
        raw = classifier.predict_proba(features)
        for position, label in enumerate(labels):
            column = list(classifier.classes_).index(label)
            np.testing.assert_allclose(ordered[:, position], raw[:, column])
        np.testing.assert_allclose(ordered.sum(axis=1), 1, atol=1e-9)

    def test_class_missing_from_training_is_rejected(self):
        labels = list(LABELS['humanitarian'])
        features = np.eye(4, dtype=float)
        classifier = LogisticRegression(max_iter=1000).fit(features, labels[:4])
        with self.assertRaises(ValueError):
            probabilities(classifier, features, labels)


class RunAlignmentTests(SyntheticSplits):
    def probabilities_for(self, frame, task):
        probs = np.full((len(frame), len(LABELS[task])), 1 / len(LABELS[task]))
        return probs, [LABELS[task][0]] * len(frame)

    def test_image_model_outputs_validate_and_do_not_collide(self):
        task, event = 'informative', 'hurricane_maria'
        splits = get_split(task, 'B', event, split_dir=self.splits)
        written = []
        for model in ('vit', 'clip_fusion'):
            for subset, frame, folder in (('test', splits['test'], 'out_of_domain'),
                                          ('test', splits['test'], 'in_domain'),
                                          ('all_target', splits['test_all_target'], None)):
                probs, predicted = self.probabilities_for(frame, task)
                path = write_predictions(frame, predicted, probs, task=task, model=model,
                                         protocol='B', fold=event, subset=subset,
                                         split_dir=self.splits,
                                         output_dir=self.root / 'predictions' / (folder or 'root'))
                metrics = evaluate_file(path, subset=subset, split_dir=self.splits, n_bootstrap=20,
                                        output_dir=self.root / 'metrics' / (folder or 'root'),)
                written.append((path, metrics))
        self.assertEqual(len({path for path, _ in written}), len(written))
        self.assertEqual(len({metrics for _, metrics in written}), len(written))
        saved = pd.read_csv(written[0][0], dtype=str, keep_default_na=False)
        self.assertEqual(len(saved), len(splits['test']))
        self.assertFalse(saved.duplicated(['tweet_id', 'image_id']).any())

    def test_image_rows_are_not_deduplicated_by_tweet(self):
        # L'entrée étant l'image, un tweet à deux images reste deux exemples.
        pairs = get_split('informative', 'A', 'official', split_dir=self.splits)['train']
        text = get_split('informative', 'A', 'official', text_only=True, split_dir=self.splits)['train']
        self.assertEqual(len(pairs), len(text) + 1)
        self.assertTrue(pairs.duplicated('tweet_id').any())


class AblationTableTests(SyntheticSplits):
    def test_rows_are_replaced_per_key_and_other_runs_kept(self):
        path = self.root / 'ablation.csv'
        base = dict(task='informative', protocol='A', fold='official')
        write_ablation([dict(base, mode=mode, f1_macro_test=0.1) for mode in ('text', 'fusion')], path)
        write_ablation([dict(base, mode='fusion', f1_macro_test=0.9),
                        dict(base, task='humanitarian', mode='fusion', f1_macro_test=0.5)], path)
        frame = pd.read_csv(path)
        self.assertEqual(len(frame), 3)
        informative = frame[frame['task'] == 'informative'].set_index('mode')['f1_macro_test']
        self.assertEqual(informative['fusion'], 0.9)
        self.assertEqual(informative['text'], 0.1)


if __name__ == '__main__':
    unittest.main()
