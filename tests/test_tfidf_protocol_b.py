"""Test the notebook's actual A/B alignment and evaluation path on synthetic data."""
import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import test_data_evaluate as fixtures
from src.data import LABELS, get_split
from src.evaluate import evaluate_file, write_predictions


def notebook_alignment():
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           'notebooks/02b_tfidf_protocol_b.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'align_a_predictions':
                namespace = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook alignment>', 'exec'), namespace)
                return namespace[node.name]
    raise AssertionError('Alignment function missing from notebook')


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        fixtures.ProtocolTests.setUp(self)
        self.align = notebook_alignment()
        self.event = 'hurricane_maria'
        self.test = get_split('informative', 'B', self.event, split_dir=self.splits)['test']
        official = get_split('informative', 'A', 'official', split_dir=self.splits)['test']
        self.a = official[['tweet_id', 'image_id', 'event_name']].copy()
        self.a['y_true'] = official.label
        self.a['y_pred'] = 'not_informative'
        self.a['proba_informative'] = '0.125'
        self.a['proba_not_informative'] = '0.875'

    def tearDown(self):
        self.temp.cleanup()

    def test_aligns_shuffled_outputs_preserving_scores(self):
        aligned = self.align(self.test, self.a.iloc[::-1], self.event)
        self.assertEqual(aligned.tweet_id.tolist(), self.test.tweet_id.tolist())
        self.assertEqual(aligned.y_pred.tolist(), ['not_informative'] * len(self.test))
        self.assertEqual(aligned.proba_informative.tolist(), ['0.125'] * len(self.test))

    def test_rejects_wrong_truth_or_pair(self):
        for column, value in [('y_true', 'wrong'), ('image_id', 'wrong')]:
            bad = self.a.copy()
            bad.loc[bad.event_name.eq(self.event), column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.align(self.test, bad, self.event)

    def test_rejects_duplicate_missing_and_additional_predictions(self):
        target = self.a.loc[self.a.event_name.eq(self.event)]
        extra = target.iloc[[0]].copy()
        extra['image_id'] = 'extra'
        for bad in [pd.concat([self.a, target.iloc[[0]]]),
                    self.a.drop(target.index[0]), pd.concat([self.a, extra])]:
            with self.assertRaises(ValueError):
                self.align(self.test, bad, self.event)

    def test_in_domain_outputs_do_not_collide_and_round_trip_unchanged(self):
        aligned = self.align(self.test, self.a, self.event)
        probs = aligned[['proba_' + label for label in LABELS['informative']]].to_numpy(float)
        paths, metric_paths = [], []
        for regime in ['in_domain', 'out_of_domain']:
            path = write_predictions(self.test, aligned.y_pred, probs, task='informative', model='tfidf',
                                     protocol='B', fold=self.event, output_dir=self.root / 'predictions' / regime,
                                     split_dir=self.splits)
            paths.append(path)
            metric_paths.append(evaluate_file(path, output_dir=self.root / 'metrics' / regime,
                                              split_dir=self.splits, n_bootstrap=20))
        self.assertNotEqual(paths[0], paths[1])
        saved = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
        np.testing.assert_array_equal(saved[['proba_' + label for label in LABELS['informative']]].to_numpy(float), probs)
        self.assertEqual(saved.y_pred.tolist(), aligned.y_pred.tolist())
        self.assertEqual(json.loads(metric_paths[0].read_text())['f1_macro'],
                         json.loads(metric_paths[1].read_text())['f1_macro'])

    def test_all_target_preserves_multiple_images_in_text_mode(self):
        event = 'hurricane_harvey'  # synthetic fixture has one extra training image here
        splits = get_split('informative', 'B', event, text_only=True, split_dir=self.splits)
        frame = splits['test_all_target']
        self.assertGreater(len(frame), frame.tweet_id.nunique())
        self.assertFalse(splits['train'].tweet_id.duplicated().any())
        probs = np.full((len(frame), 2), .5)
        path = write_predictions(frame, frame.label, probs, task='informative', model='tfidf',
                                 protocol='B', fold=event, subset='all_target', split_dir=self.splits,
                                 output_dir=self.root / 'predictions')
        result = evaluate_file(path, subset='all_target', split_dir=self.splits,
                               output_dir=self.root / 'metrics', n_bootstrap=20)
        self.assertEqual(json.loads(result.read_text())['n_samples'], len(frame))
        with self.assertRaises(ValueError):
            unique = frame.drop_duplicates('tweet_id')
            write_predictions(unique, unique.label, probs[:len(unique)], task='informative', model='tfidf',
                              protocol='B', fold=event, subset='all_target', split_dir=self.splits,
                              output_dir=self.root / 'predictions')


if __name__ == '__main__':
    unittest.main()
