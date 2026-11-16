"""Protocol and metric checks on synthetic data; no model is run."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import EVENTS, LABELS, get_split
from src.evaluate import compute_metrics, evaluate_file, write_predictions


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.splits = self.root / 'splits'
        self.splits.mkdir()
        for task, labels in LABELS.items():
            for split in ('train', 'dev', 'test'):
                rows = []
                for event in EVENTS:
                    for i, label in enumerate(labels):
                        tweet = f'000_{event}_{split}_{i}'
                        rows.append(dict(tweet_id=tweet, image_id=f'{tweet}_0', event_name=event,
                                         tweet_text='  RT é â€™\n@user HTTPURL  ', image='data/image.jpg', label=label))
                if split == 'train':
                    rows.append(dict(rows[0], image_id=rows[0]['tweet_id'] + '_1'))
                pd.DataFrame(rows).to_csv(self.splits / f'{task}_{split}.csv', index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_official_and_text_only_training(self):
        for task in LABELS:
            pairs = get_split(task, 'A', 'official', split_dir=self.splits)
            text = get_split(task, 'A', 'official', text_only=True, split_dir=self.splits)
            self.assertEqual(len(text['train']), len(pairs['train']) - 1)
            self.assertTrue(text['dev'].equals(pairs['dev']))
            self.assertTrue(text['test'].equals(pairs['test']))
            self.assertTrue(text['train'].iloc[0].tweet_id.startswith('000_'))
            self.assertEqual(text['train'].iloc[0].tweet_text, '  RT é â€™\n@user HTTPURL  ')

    def test_all_seven_event_folds(self):
        for task in LABELS:
            a = get_split(task, 'A', 'official', split_dir=self.splits)
            for event in EVENTS:
                b = get_split(task, 'B', event, split_dir=self.splits)
                self.assertNotIn(event, set(b['train'].event_name))
                self.assertNotIn(event, set(b['dev'].event_name))
                expected = a['test'].loc[a['test'].event_name == event].reset_index(drop=True)
                self.assertTrue(b['test'].equals(expected))
                self.assertEqual(set(b['test_all_target'].event_name), {event})
                self.assertEqual(len(b['test_all_target']), sum((a[s].event_name == event).sum()
                                                               for s in ('train', 'dev', 'test')))

    def test_wrong_protocol_and_conflicting_text_rejected(self):
        for protocol, fold in [('A', 'hurricane_maria'), ('B', 'official'), ('C', 'official')]:
            with self.assertRaises(ValueError):
                get_split('informative', protocol, fold, split_dir=self.splits)
        p = self.splits / 'informative_train.csv'
        frame = pd.read_csv(p, keep_default_na=False)
        frame.loc[len(frame) - 1, 'tweet_text'] = 'conflicting text'
        frame.to_csv(p, index=False)
        with self.assertRaises(ValueError):
            get_split('informative', 'A', 'official', text_only=True, split_dir=self.splits)

    def test_prediction_round_trip_and_invalid_populations(self):
        for task in LABELS:
            frame = get_split(task, 'A', 'official', split_dir=self.splits)['test']
            probabilities = np.array([[float(label == true) for label in LABELS[task]] for true in frame.label])
            path = write_predictions(frame, frame.label, probabilities, task=task, model='tfidf',
                                     protocol='A', fold='official', split_dir=self.splits,
                                     output_dir=self.root / 'predictions')
            output = evaluate_file(path, split_dir=self.splits, output_dir=self.root / 'metrics')
            metrics = json.loads(output.read_text())
            self.assertEqual(metrics['f1_macro'], 1.0)
            self.assertEqual(metrics['bootstrap']['n_resamples'], 1000)
            self.assertEqual(metrics['bootstrap']['seed'], 42)
            self.assertEqual(output.name, path.stem + '.json')
            broken = pd.read_csv(path, dtype=str, keep_default_na=False)
            broken.loc[0, 'y_true'] = LABELS[task][1]
            broken.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                evaluate_file(path, split_dir=self.splits, output_dir=self.root / 'metrics')
            with self.assertRaises(ValueError):
                write_predictions(frame, frame.label, np.zeros_like(probabilities), task=task, model='tfidf',
                                  protocol='A', fold='official', split_dir=self.splits,
                                  output_dir=self.root / 'predictions')

    def test_supplementary_target_results_are_separate(self):
        task, event = 'informative', 'hurricane_maria'
        b = get_split(task, 'B', event, split_dir=self.splits)
        outputs = []
        for subset, key in [('test', 'test'), ('all_target', 'test_all_target')]:
            frame = b[key]
            probs = np.array([[float(label == true) for label in LABELS[task]] for true in frame.label])
            path = write_predictions(frame, frame.label, probs, task=task, model='llm', protocol='B',
                                     fold=event, subset=subset, split_dir=self.splits,
                                     output_dir=self.root / 'predictions')
            outputs.append(evaluate_file(path, split_dir=self.splits, subset=subset,
                                         output_dir=self.root / 'metrics'))
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertTrue(all(path.is_file() for path in outputs))


class MetricTests(unittest.TestCase):
    def test_hand_computed_metrics_and_bootstrap(self):
        truth = np.array(['informative', 'informative', 'not_informative', 'not_informative'])
        pred = np.array(['informative', 'not_informative', 'not_informative', 'not_informative'])
        result = compute_metrics(truth, pred, 'informative', n_bootstrap=20)
        self.assertEqual(result['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result['f1_macro'], (2/3 + 4/5)/2)
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(result, compute_metrics(truth, pred, 'informative', n_bootstrap=20))
        rng = np.random.default_rng(42)
        scores = []
        for _ in range(20):
            indices = rng.integers(0, len(truth), size=len(truth))
            scores.append(f1_score(truth[indices], pred[indices], labels=LABELS['informative'],
                                   average='macro', zero_division=0))
        bounds = np.quantile(scores, [.025, .975])
        interval = result['bootstrap']['intervals']['f1_macro']
        np.testing.assert_allclose([interval['lower'], interval['upper']], bounds)

    def test_absent_classes_stay_in_macro_denominator(self):
        result = compute_metrics(['affected_individuals'], ['affected_individuals'],
                                 'humanitarian', n_bootstrap=5)
        self.assertAlmostEqual(result['f1_macro'], .2)
        self.assertEqual(len(result['confusion_matrix']), 5)
        self.assertEqual(result['per_class']['not_humanitarian']['support'], 0)

    def test_empty_misaligned_invalid_labels_rejected(self):
        for truth, pred in [([], []), (['informative'], []), (['Positive'], ['informative'])]:
            with self.assertRaises(ValueError):
                compute_metrics(truth, pred, 'informative')


if __name__ == '__main__':
    unittest.main()
