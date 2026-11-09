"""Stage I-A: Balanced-Training Rerun. CPU only; verified cached margins only."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, hashlib, json, sqlite3, sys, warnings, platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
TITLE = 'Stage I-A: Balanced-Training Rerun'
MODELS = ('0.5b', '3b')
SPLITS = ('train', 'validation', 'test')
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

def sha(p):
    p = source_path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, v):
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    t.replace(p)

def metrics(y, p, s):
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(accuracy=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average='macro', zero_division=0)), balanced_accuracy=float(balanced_accuracy_score(y, p)), auroc=float(roc_auc_score(y, s)), negative_recall=float(tn / (tn + fp)), positive_recall=float(tp / (tp + fn)), confusion_matrix=cm.tolist(), predicted_positive=int(p.sum()), n=len(y))

def counts_metrics(y, p):
    tn = np.count_nonzero((y == 0) & (p == 0))
    fp = np.count_nonzero((y == 0) & (p == 1))
    fn = np.count_nonzero((y == 1) & (p == 0))
    tp = np.count_nonzero((y == 1) & (p == 1))
    nr = tn / (tn + fp)
    pr = tp / (tp + fn)
    return np.array([(tn + tp) / len(y), 0.5 * (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn)), (nr + pr) / 2, nr, pr])

def paired_ci(y, p0, p1, s0, s1, draws, seed, positive_slope):
    names = ['accuracy', 'macro_f1', 'balanced_accuracy', 'negative_recall', 'positive_recall']
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == j) for j in (0, 1)]
    samples = np.empty((draws, 6))
    for i in range(draws):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        samples[i, :5] = counts_metrics(y[idx], p1[idx]) - counts_metrics(y[idx], p0[idx])
        samples[i, 5] = 0.0 if positive_slope else roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s0[idx])
    names += ['auroc']
    a = metrics(y, p0, s0)
    b = metrics(y, p1, s1)
    return [dict(metric=n, difference=b[n] - a[n], ci_low=float(np.quantile(samples[:, j], 0.025)), ci_high=float(np.quantile(samples[:, j], 0.975)), draws=draws, seed=seed) for j, n in enumerate(names)]

def main():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    assert args.bootstrap >= 100
    out = WORKSPACE / 'artifacts/stage_i_sentiment_correction'
    inp = out / 'reused_inputs'
    preservation = json.loads((inp / 'preservation_manifest.json').read_text(encoding='utf-8'))
    for rel, digest in preservation['preserved_sha256'].items():
        assert sha(inp / rel) == digest, 'Preserved input changed: ' + rel
    original = json.loads((inp / 'original_scoring_protocol.json').read_text(encoding='utf-8'))
    root = WORKSPACE
    for rel, digest in original['source_sha256'].items():
        assert sha(root / rel) == digest, 'External source changed: ' + rel
    data = {s: [json.loads(x) for x in (inp / (s + '.jsonl')).read_text(encoding='utf-8').splitlines()] for s in SPLITS}
    assert [len(data[s]) for s in SPLITS] == [24942, 1131, 6347]
    labels = {s: np.array([r['label'] for r in data[s]], dtype=np.int64) for s in SPLITS}
    for s in SPLITS:
        assert len({r['id'] for r in data[s]}) == len(data[s])
    assert not any((set((r['id'] for r in data[a])) & set((r['id'] for r in data[b])) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]))
    neg = np.flatnonzero(labels['train'] == 0)
    pos = np.flatnonzero(labels['train'] == 1)
    assert len(neg) == 7093 and len(pos) == 17849
    selected = np.sort(np.concatenate([neg, np.random.default_rng(42).choice(pos, 7093, replace=False)]))
    assert len(selected) == 14186
    idrows = [dict(example_index=int(i), id=data['train'][i]['id'], label=int(labels['train'][i])) for i in selected]
    pd.DataFrame(idrows).to_csv(out / 'balanced_training_ids.csv', index=False)
    save(out / 'balanced_training_manifest.json', dict(seed=42, sampler='numpy.random.default_rng(42).choice(original-order positive indices,7093,replace=False); retained indices sorted', negative=7093, positive=7093, total=14186, original_train_n=24942, training_ids_sha256=sha(out / 'balanced_training_ids.csv'), same_subset_for_both_models=True, validation_n=1131, test_n=6347))
    protocol = dict(title=TITLE, scoring_protocol=original, seed=42, C_grid=list(GRID), classifier='L2 logistic regression, lbfgs, intercept enabled, class_weight=None, tol=1e-5, max_iter=3000', features=['margin'], scaler='StandardScaler fit on balanced train only', selection='validation balanced accuracy primary, macro-F1 secondary, exact ties smallest C', final_fit='selected balanced-training model retained; no train-validation refit', bootstrap=dict(draws=args.bootstrap, seed=47, method='paired class-stratified percentile; fixed observed class counts', scope='correction minus original; conditional on fitted classifier'), test_status='Retrospective sensitivity experiment: all test examples previously examined; not independent confirmation', no_model_forward_passes=True, no_emotion_features=True, software=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__))
    save(out / 'protocol.json', protocol)
    save(out / 'state.json', dict(complete=False, phase='verified_inputs'))
    db = sqlite3.connect('file:' + str(inp / 'scores.sqlite3') + '?mode=ro', uri=True)
    assert db.execute('pragma quick_check').fetchone()[0] == 'ok'
    margins = {}
    for key in MODELS:
        margins[key] = {}
        for s in SPLITS:
            rows = db.execute('select example_index,id,label,logp_positive,logp_negative,margin from scores where model=? and split=? order by example_index', (key, s)).fetchall()
            assert len(rows) == len(data[s])
            for i, (r, v) in enumerate(zip(data[s], rows)):
                assert v[:3] == (i, r['id'], r['label'])
                assert np.isfinite(v[3:]).all()
                assert float(np.float32(v[3]) - np.float32(v[4])) == v[5]
            margins[key][s] = np.array([v[5] for v in rows])
    assert db.execute('select count(*) from scores').fetchone()[0] == 64840
    db.close()
    fitted = {}
    metadata = {}
    for key in MODELS:
        folder = out / key
        folder.mkdir(exist_ok=True)
        x = margins[key]['train'][selected, None]
        y = labels['train'][selected]
        val = margins[key]['validation'][:, None]
        search = []
        candidates = []
        print(f'{key}: fitting 7 C values on balanced train n={len(y)}; validation n=1131', flush=True)
        for c in GRID:
            model = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(C=c, solver='lbfgs', fit_intercept=True, class_weight=None, tol=1e-05, max_iter=3000, random_state=42, l1_ratio=0.0))])
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(x, y)
            score = model.decision_function(val)
            p = (score > 0).astype(int)
            m = metrics(labels['validation'], p, score)
            search.append(dict(C=c, validation_balanced_accuracy=m['balanced_accuracy'], validation_macro_f1=m['macro_f1'], validation_accuracy=m['accuracy']))
            candidates.append(model)
        winner = max(range(len(GRID)), key=lambda i: (search[i]['validation_balanced_accuracy'], search[i]['validation_macro_f1'], -GRID[i]))
        model = candidates[winner]
        fitted[key] = model
        scaler = model.named_steps['scaler']
        clf = model.named_steps['classifier']
        w = float(clf.coef_[0, 0])
        b = float(clf.intercept_[0])
        assert w != 0
        threshold = float(scaler.mean_[0] - b * scaler.scale_[0] / w)
        metadata[key] = dict(chosen_C=GRID[winner], coefficient_standardized=w, intercept=b, scaler_mean=float(scaler.mean_[0]), scaler_scale=float(scaler.scale_[0]), decision_boundary_in_margin_units=threshold, decision_sign='positive above threshold' if w > 0 else 'positive below threshold', validation_balanced_accuracy=search[winner]['validation_balanced_accuracy'], validation_macro_f1=search[winner]['validation_macro_f1'], n_iter=int(clf.n_iter_[0]), training_n=14186, features=1)
        joblib.dump(model, folder / 'correction_model.joblib')
        metadata[key]['model_sha256'] = sha(folder / 'correction_model.joblib')
        save(folder / 'model_metadata.json', metadata[key])
        pd.DataFrame(search).to_csv(folder / 'regularization_search.csv', index=False)
        print(f'{key}: selected C={GRID[winner]}, threshold={threshold:.6f}, validation balanced accuracy={search[winner]['validation_balanced_accuracy']:.4f}', flush=True)
    summary = []
    cis = []
    changes = []
    allmetrics = {}
    confusions = {}
    for key in MODELS:
        allmetrics[key] = {}
        for s in SPLITS:
            idx = selected if s == 'train' else np.arange(len(labels[s]))
            y = labels[s][idx]
            m = margins[key][s][idx]
            scores = fitted[key].decision_function(m[:, None])
            p0 = (m > 0).astype(int)
            p1 = (scores > 0).astype(int)
            if metadata[key]['coefficient_standardized'] > 0:
                assert np.array_equal(np.argsort(m, kind='stable'), np.argsort(scores, kind='stable'))
                assert roc_auc_score(y, m) == roc_auc_score(y, scores)
            methods = {'original_margin': metrics(y, p0, m), 'balanced_correction': metrics(y, p1, scores)}
            allmetrics[key][s] = methods
            for method, result in methods.items():
                summary.append(dict(model=key, split=s, method=method, chosen_C=metadata[key]['chosen_C'] if method == 'balanced_correction' else None, **{k: v for k, v in result.items() if k != 'confusion_matrix'}))
                confusions[f'{key}/{s}/{method}'] = result['confusion_matrix']
            pred = pd.DataFrame(dict(id=[data[s][i]['id'] for i in idx], example_index=idx, label=y, margin=m, original_prediction=p0, corrected_decision_score=scores, corrected_probability=fitted[key].predict_proba(m[:, None])[:, 1], corrected_prediction=p1))
            pred.to_csv(out / key / (s + '_predictions.csv'), index=False)
            if s == 'test':
                ci = paired_ci(y, p0, p1, m, scores, args.bootstrap, 47, metadata[key]['coefficient_standardized'] > 0)
                cis.extend([dict(model=key, **r) for r in ci])
                changes.append(dict(model=key, incorrect_corrected=int(np.count_nonzero((p0 != y) & (p1 == y))), correct_made_incorrect=int(np.count_nonzero((p0 == y) & (p1 != y))), changed=int(np.count_nonzero(p0 != p1))))
        save(out / key / 'split_metrics.json', allmetrics[key])
        print(f'{key}: train/validation/test metrics and {args.bootstrap} paired bootstrap draws saved', flush=True)
    pd.DataFrame(summary).to_csv(out / 'summary_table.csv', index=False)
    pd.DataFrame(cis).to_csv(out / 'paired_differences_95ci.csv', index=False)
    pd.DataFrame(changes).to_csv(out / 'prediction_changes.csv', index=False)
    save(out / 'confusion_matrices.json', confusions)
    save(out / 'release_run_summary.json', {'scope': 'new local inputs; historical comparisons deliberately omitted'})
    for rel, digest in original['source_sha256'].items():
        assert sha(root / rel) == digest
    for rel, digest in preservation['preserved_sha256'].items():
        assert sha(inp / rel) == digest
    for key in MODELS:
        loaded = joblib.load(out / key / 'correction_model.joblib')
        assert np.array_equal(loaded.predict(margins[key]['test'][:, None]), fitted[key].predict(margins[key]['test'][:, None]))
    save(out / 'verification.json', dict(cache_rows=64840, balanced_train_counts=[7093, 7093], validation_n=1131, test_n=6347, baseline_accuracy_unchanged=True, source_hashes_unchanged=True, preserved_inputs_unchanged=True, serialized_model_predictions_match=True, both_models_frozen_before_test=True))
    save(out / 'state.json', dict(complete=True, phase='complete', source_files_unchanged=True))
    print('\n' + text, flush=True)
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
