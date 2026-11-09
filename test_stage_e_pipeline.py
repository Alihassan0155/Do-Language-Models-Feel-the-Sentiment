"""Meaningful Stage E split, leakage, control-geometry and test-phase checks."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import joblib
import stage_e_pipeline as pipeline
import run_stage_e as runner
from prepare_stage_e import grouped_split, normalized

class StageETests(unittest.TestCase):

    def test_grouped_split_prevents_duplicate_leakage(self):
        rows = []
        for label in (0, 1):
            for index in range(20):
                for text in (f'class {label} example {index}', f' CLASS {label}  EXAMPLE {index} '):
                    rows.append({'text': text, 'label': label})
        train, validation = grouped_split(rows, 0.2, 42)
        self.assertFalse({normalized(r['text']) for r in train} & {normalized(r['text']) for r in validation})
        self.assertEqual(len(train) + len(validation), len(rows))
        self.assertEqual(grouped_split(rows, 0.2, 42), (train, validation))

    def test_controls_and_pca_training_only(self):
        rng = np.random.default_rng(8)
        h = rng.normal(size=(80, 12))
        centered = rng.normal(size=(8, 12))
        centered -= centered.mean(axis=0)
        final = pipeline.units(centered)
        self.assertEqual(pipeline.rank(final), 7)
        info = {'width': 12, 'emotion_effective_rank': 7}
        with patch.object(pipeline, 'emotion_inputs', return_value=({'emotion8': final, 'emotion_raw8': final}, info)):
            recipes = pipeline.recipes({'seed': 42}, '0.5b', h)
        self.assertEqual(len(recipes), 25)
        self.assertTrue(np.allclose(recipes['pca8']['pca_train_mean'], h.mean(axis=0)))
        shifted_heldout = rng.normal(size=(20, 12)) + 20
        self.assertFalse(np.allclose(recipes['pca8']['pca_train_mean'], np.vstack([h, shifted_heldout]).mean(axis=0)))
        random = recipes['random8_seed_3']['directions']
        self.assertTrue(np.allclose(random @ random.T, np.eye(8)))
        self.assertTrue(np.array_equal(random, pipeline.random_directions(12, 3)))
        self.assertTrue(np.array_equal(recipes['random_rank7_seed_3']['directions'], random[:7]))
        self.assertEqual(pipeline.project(shifted_heldout, recipes['pca_rank7']).shape, (20, 7))

    def test_scaler_fit_only_train_and_export_is_frozen(self):
        rng = np.random.default_rng(7)
        x, xv = (rng.normal(size=(100, 8)), rng.normal(size=(80, 8)) + 0.4)
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
        yv = (xv[:, 0] + 0.5 * xv[:, 1] > 0.5).astype(int)
        probe, chosen, search = pipeline.fit_probe(x, y, xv, yv, [0.01, 0.1, 1.0])
        self.assertTrue(np.allclose(probe['scale'].mean_, x.mean(axis=0)))
        self.assertFalse(np.allclose(probe['scale'].mean_, np.vstack([x, xv]).mean(axis=0)))
        self.assertEqual(chosen, max(search, key=lambda r: (r['macro_f1'], -r['C'])))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'probe.joblib'
            joblib.dump(probe, path)
            self.assertTrue(np.array_equal(joblib.load(path).predict(xv), probe.predict(xv)))

    def test_evaluation_preflight_does_not_open_test_before_fit(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, 'E', Path(folder)), patch.object(pipeline, 'load_hidden', side_effect=AssertionError('Test accessed early')) as hidden:
            p = {'models': {'0.5b': {'methods': ['emotion8']}}}
            with self.assertRaises(FileNotFoundError):
                pipeline.evaluate_all(p)
            hidden.assert_not_called()

    def test_reset_is_scoped_to_stage_e(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            earlier = root / 'artifacts' / 'stage_d'
            earlier.mkdir(parents=True)
            sentinel = earlier / 'keep.txt'
            sentinel.write_text('earlier experiment', encoding='utf-8')
            stage_e = root / 'artifacts' / 'stage_e'
            stage_e.mkdir()
            (stage_e / 'old.txt').write_text('old results', encoding='utf-8')
            plots = root / 'visualizations' / 'stage_e'
            plots.mkdir(parents=True)
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'E', stage_e):
                runner.reset_outputs()
            self.assertTrue(sentinel.exists())
            self.assertFalse(stage_e.exists())
            self.assertFalse(plots.exists())
if __name__ == '__main__':
    unittest.main(verbosity=2)
