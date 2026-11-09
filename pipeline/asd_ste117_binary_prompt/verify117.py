"""Independent CPU verification of STE117 metrics/bootstrap and preservation."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import csv, json, sqlite3, hashlib, importlib.util
import numpy as np
from sklearn.metrics import roc_auc_score
OUT = WORKSPACE / 'artifacts/asd_ste117_binary_prompt'
ROOT = WORKSPACE

def read(name):
    return list(csv.DictReader((OUT / name).open(encoding='utf-8')))

def js(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))

def main():
    assert js('verification.json')['complete']
    y = np.array([int(r['label']) for r in js('samples.json')['tweet_eval_sentiment']])
    n = len(y)
    allci = read('full_confidence_intervals.csv') + read('matched_confidence_intervals.csv')
    checks = 0
    maxgap = 0.0
    aucchecks = 0
    seeds = list(range(2000, 2050))
    sw = np.random.default_rng(46).multinomial(50, np.full(50, 1 / 50), size=2000) / 50
    spec = importlib.util.spec_from_file_location('metric_helper', source_path(ROOT / 'artifacts/binary_reanalysis_20261019/reanalyze.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for stage, seed in [('full', 44), ('matched', 45)]:
        rng = np.random.default_rng(seed)
        w = np.zeros((2000, n))
        for lab in (0, 1):
            ix = np.flatnonzero(y == lab)
            w[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=2000)
        w /= n
        c = sqlite3.connect('file:' + (OUT / stage / 'scores.sqlite3').as_posix() + '?mode=ro&immutable=1', uri=True)
        for model in ('0.5b', '3b'):

            def compute(name):
                nonlocal maxgap, aucchecks
                a = np.array(c.execute('select logp_positive,logp_negative from scores where model=? and dataset=? and method=? order by example_index', (model, 'tweet_eval_sentiment', name)).fetchall(), dtype=np.float32)
                p = a[:, 0] > a[:, 1]
                m = (a[:, 0] - a[:, 1]).astype(float)
                tp = w @ ((y == 1) & p)
                tn = w @ ((y == 0) & ~p)
                fp = w @ ((y == 0) & p)
                fn = w @ ((y == 1) & ~p)
                values = np.stack([tp + tn, 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))], axis=1)
                if name in ('baseline', 'emotion_raw', 'emotion_denoised', 'random_rank7_seed_2000', 'random_rank7_seed_2049', 'matched_raw_seed_2000', 'matched_denoised_seed_2049'):
                    ix = np.arange(0, 2000, 100)
                    auc = helper.metrics(y, p, m, w[ix])[:, 2]
                    for i, v in zip(ix, auc):
                        gap = abs(float(v) - roc_auc_score(y, m, sample_weight=w[i]))
                        assert gap < 1e-12
                        maxgap = max(maxgap, gap)
                        aucchecks += 1
                return values
            base = compute('baseline')
            rand = np.stack([compute(f'random_rank7_seed_{s}') for s in seeds], axis=1)
            random = np.einsum('bs,bsm->bm', sw, rand) if stage == 'matched' else rand.mean(axis=1)
            vals = {'baseline': base, 'random_rank7_mean': random}
            for kind in ('raw', 'denoised'):
                vals['emotion_' + kind] = np.einsum('bs,bsm->bm', sw, np.stack([compute(f'matched_{kind}_seed_{s}') for s in seeds], axis=1)) if stage == 'matched' else compute('emotion_' + kind)
            for condition, draws in vals.items():
                for j, metric in enumerate(('accuracy', 'macro_f1')):
                    for stat, arr in [('performance', draws[:, j]), ('damage', base[:, j] - draws[:, j])] + ([('excess_vs_random', random[:, j] - draws[:, j])] if condition.startswith('emotion') else []):
                        ref = next((r for r in allci if (r['stage'], r['model'], r['condition'], r['metric'], r['statistic']) == (stage, model, condition, metric, stat)))
                        v = np.quantile(arr, [0.025, 0.975])
                        gap = float(np.max(np.abs(v - [float(ref['ci_low']), float(ref['ci_high'])])))
                        assert gap < 1e-12
                        maxgap = max(maxgap, gap)
                        checks += 1
        c.close()
    protocol = js('protocol.json')
    for rel, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
    result = dict(passed=True, independent_accuracy_f1_interval_checks=checks, independent_weighted_sklearn_auroc_checks=aucchecks, max_absolute_gap=maxgap, historical_inputs_unchanged=True)
    (OUT / 'independent_validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result)
if __name__ == '__main__':
    main()
