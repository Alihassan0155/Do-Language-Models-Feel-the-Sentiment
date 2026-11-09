"""Stage I-B: frozen margin + emotion projection correction. Read-only source caches."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, hashlib, json, sqlite3, sys, time, warnings, unicodedata, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
ROOT = WORKSPACE
OUT = WORKSPACE / 'artifacts/stage_i_emotion_guided_correction'
IA = ROOT / 'artifacts/stage_i_sentiment_correction'
E = ROOT / 'artifacts/stage_e'
MODELS = ('0.5b', '3b')
SPLITS = ('train', 'validation', 'test')
EMOTIONS = ('joy', 'love', 'excitement', 'gratitude', 'sadness', 'anger', 'fear', 'disgust')
SEEDS = tuple(range(10))
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

def sha(p):
    p = source_path(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def js(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    t.replace(p)

def arrsave(p, a):
    t = p.with_suffix(p.suffix + '.tmp')
    with t.open('wb') as f:
        np.save(f, a)
    t.replace(p)

def log(s):
    print(s, flush=True)
    with (OUT / 'execution.log').open('a', encoding='utf-8') as f:
        f.write(s + '\n')

def units(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    assert np.isfinite(v).all() and np.all(n > 0)
    return v / n

def audit():
    ia = js(IA / 'protocol.json')
    old = ia['scoring_protocol']
    ep = js(E / 'protocol.json')
    manifest = js(IA / 'balanced_training_manifest.json')
    ids = pd.read_csv(IA / 'balanced_training_ids.csv')
    data = {s: [json.loads(x) for x in (IA / 'reused_inputs' / (s + '.jsonl')).read_text(encoding='utf-8').splitlines()] for s in SPLITS}
    indices = {'train': ids.example_index.to_numpy(dtype=int), 'validation': np.arange(1131), 'test': np.arange(6347)}
    assert len(ids) == 14186 and ids.id.is_unique and (ids.label.value_counts().to_dict() == {0: 7093, 1: 7093})
    assert sha(IA / 'balanced_training_ids.csv') == manifest['training_ids_sha256']
    assert ids.id.tolist() == [data['train'][i]['id'] for i in indices['train']]
    assert ids.label.tolist() == [data['train'][i]['label'] for i in indices['train']]
    textsets = {s: {' '.join(unicodedata.normalize('NFKC', r['text']).casefold().split()) for r in data[s]} for s in SPLITS}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not textsets[a] & textsets[b], f'Text overlap: {a}/{b}'
    tracked = {str(p.relative_to(ROOT)): sha(p) for p in IA.rglob('*') if p.is_file()}
    tracked['artifacts/stage_e/protocol.json'] = sha(E / 'protocol.json')
    for filename in ['build_stage_b_vectors.py', 'build_stage_b_final_vectors.py', 'run_stage_c_goemotions.py', 'stage_e_common.py', 'stage_e_pipeline.py', 'stage_d.py', 'stage_f.py', 'stage_g.py', 'stage_h.py']:
        tracked[filename] = sha(ROOT / filename)
    cache = sqlite3.connect('file:' + str(IA / 'reused_inputs/scores.sqlite3') + '?mode=ro', uri=True)
    assert cache.execute('pragma quick_check').fetchone()[0] == 'ok'
    margins = {}
    vectors = {}
    info = {}
    baseline = {}
    hidden = {}
    reports = {}
    for key in MODELS:
        layer = ep['models'][key]['layer']
        total = ep['models'][key]['total_layers']
        width = ep['models'][key]['width']
        cfg = ep['models'][key]
        spec = old['models'][key]
        for field in ('model_id', 'checkpoint_commit', 'tokenizer_sha256', 'config_sha256'):
            assert cfg[field] == spec[field], f'{key} identity: {field}'
        assert cfg['prompts']['tweet_eval_sentiment'] == spec['prompt']
        assert (cfg['layer'], cfg['total_layers'], cfg['width']) == (layer, total, width)
        selected = ROOT / 'artifacts/stage_c_empathetic_dialogues/selected_intervention_layers.json'
        assert sha(selected) == cfg['selection_sha256']
        assert js(selected)['models'][key]['selected']['layer_1_based'] == layer
        tracked[str(selected.relative_to(ROOT))] = sha(selected)
        vectors[key] = {}
        info[key] = cfg
        rawpath = ROOT / 'artifacts/stage_b_raw_vectors' / key / 'emotion_directions_raw.float32.npy'
        rawmeta = js(rawpath.parent / 'vectors_metadata.json')
        finalpath = ROOT / 'artifacts/stage_b_final_vectors' / key / 'emotion_directions_final.float32.npy'
        finalmeta = js(finalpath.parent / 'final_vectors_metadata.json')
        assert finalmeta['raw_directions_sha256'] == sha(rawpath)
        for name, p, meta, expected in [('raw', rawpath, rawmeta, cfg['raw_sha256']), ('denoised', finalpath, finalmeta, cfg['final_sha256'])]:
            assert meta['complete'] and meta['model_id'] == spec['model_id'] and (tuple(meta['emotion_order']) == EMOTIONS) and (sha(p) == expected)
            v = np.load(p)
            assert v.shape == (8, total, width) and np.isfinite(v).all()
            vectors[key][name] = units(v[:, layer - 1])
            tracked[str(p.relative_to(ROOT))] = sha(p)
            mp = p.parent / ('vectors_metadata.json' if name == 'raw' else 'final_vectors_metadata.json')
            tracked[str(mp.relative_to(ROOT))] = sha(mp)
        assert 'after each decoder block' in rawmeta['pooling']
        assert 'projection' in finalmeta
        random = []
        for seed in SEEDS:
            q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(width, 8)), mode='reduced')
            q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
            v = q.T
            assert np.allclose(v @ v.T, np.eye(8), atol=1e-12)
            random.append(v)
        vectors[key]['random'] = np.stack(random)
        hidden[key] = {}
        margins[key] = {}
        for split in SPLITS:
            folder = E / 'hidden' / key / 'tweet_eval_sentiment' / split
            meta = js(folder / 'metadata.json')
            rows = data[split]
            assert len(rows) == {'train': 24942, 'validation': 1131, 'test': 6347}[split]
            assert meta['complete'] and meta['completed_rows'] == len(rows) and (meta['protocol_sha256'] == sha(E / 'protocol.json')) and (meta['source_sha256'] == sha(IA / 'reused_inputs' / (split + '.jsonl'))) and (meta['layer'] == layer) and (meta['width'] == width) and (meta['truncated_examples'] == 0)
            for fname, mfield in [('pooled.float32.npy', 'hidden_sha256'), ('labels.int64.npy', 'labels_sha256'), ('emotion8.float64.npy', 'emotion_features_sha256')]:
                assert sha(folder / fname) == meta[mfield]
                tracked[str((folder / fname).relative_to(ROOT))] = meta[mfield]
            h = np.load(folder / 'pooled.float32.npy', mmap_mode='r')
            assert h.shape == (len(rows), width) and h.dtype == np.float32 and np.isfinite(h).all()
            assert np.array_equal(np.load(folder / 'labels.int64.npy'), [r['label'] for r in rows])
            db = sqlite3.connect('file:' + str(folder / 'progress.sqlite3') + '?mode=ro', uri=True)
            records = db.execute('select row_index,id,text_sha256,text_tokens from completed order by row_index').fetchall()
            db.close()
            assert len(records) == len(rows)
            assert all((v[:3] == (i, r['id'], r['text_sha256']) and v[3] > 0 for i, (r, v) in enumerate(zip(rows, records))))
            for fname in ['metadata.json', 'signature.json', 'progress.sqlite3']:
                tracked[str((folder / fname).relative_to(ROOT))] = sha(folder / fname)
            scores = cache.execute('select example_index,id,label,logp_positive,logp_negative,margin from scores where model=? and split=? order by example_index', (key, split)).fetchall()
            assert len(scores) == len(rows)
            for i, (r, v) in enumerate(zip(rows, scores)):
                assert v[:3] == (i, r['id'], r['label'])
                assert np.isfinite(v[3:]).all()
                assert float(np.float32(v[3]) - np.float32(v[4])) == v[5]
            margins[key][split] = np.array([v[5] for v in scores])[indices[split]]
            hidden[key][split] = h
        clf = joblib.load(IA / key / 'correction_model.joblib')
        baseline[key] = clf
        f = pd.read_csv(IA / key / 'test_predictions.csv', float_precision='round_trip')
        m = margins[key]['test']
        y = np.array([r['label'] for r in data['test']])
        assert f.id.tolist() == [r['id'] for r in data['test']]
        assert np.array_equal(m, f.margin.to_numpy()) and np.array_equal(clf.predict(m[:, None]), f.corrected_prediction)
        current = js(IA / key / 'split_metrics.json')['test']
        expected = (current['original_margin']['accuracy'], current['balanced_correction']['accuracy'], current['original_margin']['auroc'])
        measured = (accuracy_score(y, m > 0), accuracy_score(y, clf.predict(m[:, None])), roc_auc_score(y, m))
        assert np.allclose(measured, expected, atol=1e-14, rtol=0)
        sample = indices['train'][:8]
        h = np.asarray(hidden[key]['train'][sample], dtype=np.float64)
        scores = h @ vectors[key]['denoised'].T
        saved = np.load(E / 'hidden' / key / 'tweet_eval_sentiment/train/emotion8.float64.npy', mmap_mode='r')[sample]
        assert np.allclose(scores, saved, rtol=1e-12, atol=1e-10)
        for name in ['raw', 'denoised']:
            a = h @ vectors[key][name].T
            b = np.array([[np.dot(row, v) for v in vectors[key][name]] for row in h])
            assert a.shape == (8, 8) and np.isfinite(a).all() and np.allclose(a, b, atol=1e-10)
        reports[key] = dict(layer_1_based=layer, block_index_0_based=layer - 1, width=width, total_layers=total, baseline_accuracy=float(measured[0]), sentiment_only_accuracy=float(measured[1]), baseline_auroc=float(measured[2]), raw_rank=int(np.linalg.matrix_rank(vectors[key]['raw'], tol=1e-05)), denoised_rank=int(np.linalg.matrix_rank(vectors[key]['denoised'], tol=1e-05)), random_rank=8, smoke_shape=[8, 8], cached_projection_matches_existing_stage_e=True)
        log(f'Audit/smoke {key}: block{layer}, width{width}; cached IDs/hashes/LP margins verified; baselines match Stage I-A')
    cache.close()
    protocol = dict(title='Stage I-B: Emotion-Guided Sentiment Correction', models={k: {field: info[k][field] for field in ('model_id', 'checkpoint_commit', 'tokenizer_sha256', 'config_sha256', 'layer', 'total_layers', 'width', 'raw_sha256', 'final_sha256')} for k in MODELS}, emotion_order=EMOTIONS, feature='cached Stage I-A margin plus pooled text-token post-block residual dot unit direction; no per-example normalization', token_position_exception='Stages B/C/E/F explicitly use mean text-token pooling; use that representation, not answer-prefix token', margin_scoring=old, training_counts=[7093, 7093], validation_counts=[312, 819], test_counts=[3972, 2375], C_grid=GRID, selection='validation balanced accuracy, then macro-F1, then smallest C', seed=42, random_seeds=SEEDS, random_construction='Gaussian QR orthonormal eight directions; unit norm; label-independent; same unprocessed h as raw/denoised', classifier='L2 lbfgs, intercept, class_weight=None, tol1e-5 max_iter3000; train-only scaler; no refit', bootstrap=dict(draws=2000, seed=47, class_stratified=True, paired=True, random_mean_ci='conditional on ten fixed directions; seed sample SD reported separately'), test_status='retrospective evaluation; test previously examined; not independent confirmation', new_qwen_forward_passes=0, cache_reuse=True, no_quantization=True, rank_limitation='emotion features span rank7; eight random orthonormal features span rank8', stage_b_revision_limitation='Original Stage B vectors record model ID/shape but not commit; downstream Stage E/I-A checkpoint identity verified', versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__))
    if (OUT / 'protocol.json').exists():
        assert js(OUT / 'protocol.json') == json.loads(json.dumps(protocol)), 'Protocol changed; refuse incompatible resume'
    if (OUT / 'repository_audit.json').exists():
        assert js(OUT / 'repository_audit.json')['source_sha256'] == tracked, 'Source identities changed; refuse stale feature-cache reuse'
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'repository_audit.json', dict(models=reports, source_sha256=tracked, interfaces='See IMPLEMENTATION_PLAN.md', all_caches_compatible=True))
    save(OUT / 'data_manifest.json', dict(balanced_manifest=manifest, ids_sha256=sha(IA / 'balanced_training_ids.csv'), counts={s: len(indices[s]) for s in SPLITS}, split_overlap=0, emotion_order=EMOTIONS))
    for s in SPLITS:
        pd.DataFrame([dict(id=data[s][i]['id'], source_index=int(i), label=data[s][i]['label']) for i in indices[s]]).to_csv(OUT / (s + '_ids.csv'), index=False)
    return (data, indices, margins, vectors, baseline, hidden, tracked)

def extract(data, indices, margins, vectors, hidden):
    db = sqlite3.connect(OUT / 'feature_cache.sqlite3')
    db.execute('pragma journal_mode=WAL')
    db.execute('pragma synchronous=FULL')
    db.execute('create table if not exists features(model text,split text,row_index integer,id text,raw blob,denoised blob,random blob,checksum text,primary key(model,split,row_index))')
    feats = {}
    try:
        for key in MODELS:
            feats[key] = {}
            folder = OUT / key
            folder.mkdir(exist_ok=True)
            arrsave(folder / 'random_directions.float64.npy', vectors[key]['random'])
            arrsave(folder / 'raw_unit_directions.float64.npy', vectors[key]['raw'])
            arrsave(folder / 'denoised_unit_directions.float64.npy', vectors[key]['denoised'])
            for split in SPLITS:
                source = indices[split]
                n = len(source)
                cached = {v[0]: v[1:] for v in db.execute('select row_index,id,raw,denoised,random,checksum from features where model=? and split=?', (key, split))}
                assert all((0 <= i < n for i in cached))
                a = np.empty((n, 8))
                b = np.empty((n, 8))
                c = np.empty((n, 10, 8))
                for i, (id_, rb, db_, xb, checksum) in cached.items():
                    assert id_ == data[split][source[i]]['id']
                    assert hashlib.sha256(id_.encode() + rb + db_ + xb).hexdigest() == checksum
                    raw = np.frombuffer(rb, dtype='<f8')
                    den = np.frombuffer(db_, dtype='<f8')
                    rand = np.frombuffer(xb, dtype='<f8')
                    assert raw.shape == (8,) and den.shape == (8,) and (rand.shape == (80,)) and np.isfinite(np.concatenate([raw, den, rand])).all()
                    a[i] = raw
                    b[i] = den
                    c[i] = rand.reshape(10, 8)
                pending = [i for i in range(n) if i not in cached]
                log(f'Features {key}/{split}: {len(cached)}/{n} saved; {len(pending)} pending (CPU cached-state projections)')
                for start in range(0, len(pending), 512):
                    idx = np.array(pending[start:start + 512])
                    h = np.asarray(hidden[key][split][source[idx]], dtype=np.float64)
                    raw = h @ vectors[key]['raw'].T
                    den = h @ vectors[key]['denoised'].T
                    rand = (h @ vectors[key]['random'].reshape(80, -1).T).reshape(len(idx), 10, 8)
                    assert np.isfinite(raw).all() and np.isfinite(den).all() and np.isfinite(rand).all()
                    records = []
                    for j, i in enumerate(idx):
                        id_ = data[split][source[i]]['id']
                        rb = raw[j].astype('<f8').tobytes()
                        db_ = den[j].astype('<f8').tobytes()
                        xb = rand[j].astype('<f8').tobytes()
                        records.append((key, split, int(i), id_, rb, db_, xb, hashlib.sha256(id_.encode() + rb + db_ + xb).hexdigest()))
                    with db:
                        db.executemany('insert into features values(?,?,?,?,?,?,?,?)', records)
                    a[idx] = raw
                    b[idx] = den
                    c[idx] = rand
                    if start % 4096 == 0 or start + len(idx) == len(pending):
                        log(f'  {key}/{split}: {len(cached) + start + len(idx)}/{n}')
                folder = OUT / key / split
                folder.mkdir(exist_ok=True)
                paths = {'raw': folder / 'raw_emotion_scores.float64.npy', 'denoised': folder / 'denoised_emotion_scores.float64.npy', 'random': folder / 'random_scores.float64.npy'}
                for name, v in [('raw', a), ('denoised', b), ('random', c)]:
                    arrsave(paths[name], v)
                arrsave(folder / 'margin.float64.npy', margins[key][split])
                save(folder / 'features_metadata.json', dict(complete=True, n=n, ids_sha256=sha(OUT / (split + '_ids.csv')), array_sha256={name: sha(p) for name, p in paths.items()}, margin_sha256=sha(folder / 'margin.float64.npy'), emotion_order=EMOTIONS, random_seed_axis=SEEDS))
                feats[key][split] = {'raw': a, 'denoised': b, **{f'random_seed_{seed}': c[:, seed] for seed in SEEDS}}
        db.execute('pragma wal_checkpoint(TRUNCATE)')
    finally:
        db.close()
    save(OUT / 'state.json', dict(complete=False, phase='features_complete', counts_per_model={'train': 14186, 'validation': 1131, 'test': 6347}))
    return feats

def getmetrics(y, p, s):
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(accuracy=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average='macro', zero_division=0)), balanced_accuracy=float(balanced_accuracy_score(y, p)), auroc=float(roc_auc_score(y, s)), negative_recall=float(tn / (tn + fp)), positive_recall=float(tp / (tp + fn)), confusion_matrix=cm.tolist(), predicted_positive=int(p.sum()), n=len(y))

def train(data, indices, margins, feats):
    fitted = {}
    selected = {}
    for key in MODELS:
        fitted[key] = {}
        selected[key] = {}
        y = np.array([data['train'][i]['label'] for i in indices['train']])
        yv = np.array([r['label'] for r in data['validation']])
        for name in feats[key]['train']:
            x = np.column_stack([margins[key]['train'], feats[key]['train'][name]])
            xv = np.column_stack([margins[key]['validation'], feats[key]['validation'][name]])
            assert x.shape == (14186, 9)
            folder = OUT / key / 'classifiers' / name
            folder.mkdir(parents=True, exist_ok=True)
            search = []
            candidates = []
            for c in GRID:
                p = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(C=c, solver='lbfgs', fit_intercept=True, class_weight=None, tol=1e-05, max_iter=3000, random_state=42, l1_ratio=0.0))])
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    p.fit(x, y)
                assert np.allclose(p.named_steps['scaler'].mean_, x.mean(axis=0), atol=1e-10)
                score = p.decision_function(xv)
                result = getmetrics(yv, (score > 0).astype(int), score)
                search.append(dict(C=c, validation_balanced_accuracy=result['balanced_accuracy'], validation_macro_f1=result['macro_f1'], validation_accuracy=result['accuracy'], n_iter=int(p.named_steps['classifier'].n_iter_[0])))
                candidates.append(p)
            best = max(range(len(GRID)), key=lambda i: (search[i]['validation_balanced_accuracy'], search[i]['validation_macro_f1'], -GRID[i]))
            p = candidates[best]
            fitted[key][name] = p
            meta = dict(**search[best], feature_order=['margin'] + (list(EMOTIONS) if name in ['raw', 'denoised'] else [f'random{i}' for i in range(8)]), coefficient=p.named_steps['classifier'].coef_.tolist(), intercept=p.named_steps['classifier'].intercept_.tolist(), scaler_mean=p.named_steps['scaler'].mean_.tolist(), scaler_scale=p.named_steps['scaler'].scale_.tolist(), scaler_fit_n=int(p.named_steps['scaler'].n_samples_seen_), train_only=True)
            temp = folder / 'model.joblib.tmp'
            joblib.dump(p, temp)
            temp.replace(folder / 'model.joblib')
            meta['model_sha256'] = sha(folder / 'model.joblib')
            save(folder / 'model_metadata.json', meta)
            pd.DataFrame(search).to_csv(folder / 'regularization_search.csv', index=False)
            selected[key][name] = meta
            log(f'Fit {key}/{name}: selected C={GRID[best]}, validation balanced accuracy={meta['validation_balanced_accuracy']:.5f}')
    save(OUT / 'frozen_selections.json', selected)
    save(OUT / 'state.json', dict(complete=False, phase='all_classifiers_frozen_before_test', new_classifiers=24))
    return (fitted, selected)

def weighted_metrics(y, p, s, weight, groups):
    tn = weight[(y == 0) & (p == 0)].sum()
    fp = weight[(y == 0) & (p == 1)].sum()
    fn = weight[(y == 1) & (p == 0)].sum()
    tp = weight[(y == 1) & (p == 1)].sum()
    n0 = tn + fp
    n1 = fn + tp
    nr = tn / n0
    pr = tp / n1
    inv, gcount = groups
    neg = np.bincount(inv[y == 0], weights=weight[y == 0], minlength=gcount)
    pos = np.bincount(inv[y == 1], weights=weight[y == 1], minlength=gcount)
    auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (n0 * n1)
    return np.array([(tn + tp) / (n0 + n1), 0.5 * (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn)), 0.5 * (nr + pr), auc])

def bootstrap(key, y, pred, scores, point):
    names = list(pred)
    n = len(y)
    byclass = [np.flatnonzero(y == j) for j in (0, 1)]
    rng = np.random.default_rng(47)
    draws = {name: np.empty((2000, 4)) for name in names}
    groups = {}
    for name in names:
        unique, inv = np.unique(scores[name], return_inverse=True)
        groups[name] = (inv, len(unique))
        check = weighted_metrics(y, pred[name], scores[name], np.ones(n), groups[name])
        assert np.allclose(check, [point[name][k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc')], atol=1e-12)
    for i in range(2000):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in byclass])
        weight = np.bincount(indices, minlength=n)
        for name in names:
            draws[name][i] = weighted_metrics(y, pred[name], scores[name], weight, groups[name])
        if (i + 1) % 500 == 0:
            log(f'Bootstrap {key}: {i + 1}/2000 paired draws, all methods and seeds')
    rows = []
    metrics = ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc')
    compare = ['raw', 'denoised'] + [f'random_seed_{s}' for s in SEEDS]
    for name in compare:
        for reference in ['sentiment_only', 'original']:
            delta = draws[name] - draws[reference]
            for j, m in enumerate(metrics):
                low, high = np.quantile(delta[:, j], [0.025, 0.975])
                rows.append(dict(model=key, method=name, reference=reference, metric=m, difference=point[name][m] - point[reference][m], ci_low=float(low), ci_high=float(high), draws=2000, seed=47, uncertainty='paired examples, conditional on fitted models and fixed direction(s)'))
    random_mean = np.mean([draws[f'random_seed_{s}'] for s in SEEDS], axis=0)
    for reference in ['sentiment_only', 'original']:
        delta = random_mean - draws[reference]
        for j, m in enumerate(metrics):
            low, high = np.quantile(delta[:, j], [0.025, 0.975])
            rows.append(dict(model=key, method='random_mean_10', reference=reference, metric=m, difference=float(np.mean([point[f'random_seed_{s}'][m] for s in SEEDS])) - point[reference][m], ci_low=float(low), ci_high=float(high), draws=2000, seed=47, uncertainty='paired examples conditional on ten fixed directions; seed SD separate'))
    for name in ['raw', 'denoised']:
        delta = draws[name] - random_mean
        for j, m in enumerate(metrics):
            low, high = np.quantile(delta[:, j], [0.025, 0.975])
            rows.append(dict(model=key, method=name, reference='random_mean_10', metric=m, difference=point[name][m] - float(np.mean([point[f'random_seed_{s}'][m] for s in SEEDS])), ci_low=float(low), ci_high=float(high), draws=2000, seed=47, uncertainty='paired examples conditional on ten fixed random directions; seed SD separate'))
    return rows

def evaluate(data, indices, margins, feats, baseline, fitted, selected):
    allmetrics = {}
    summary = []
    changes = []
    ci = []
    confusions = {}
    random_summary = []
    for key in MODELS:
        allmetrics[key] = {}
        testpred = None
        testscores = None
        for split in SPLITS:
            ids = [data[split][i]['id'] for i in indices[split]]
            y = np.array([data[split][i]['label'] for i in indices[split]])
            m = margins[key][split]
            scores = {'original': m, 'sentiment_only': baseline[key].decision_function(m[:, None])}
            pred = {'original': (m > 0).astype(int), 'sentiment_only': baseline[key].predict(m[:, None])}
            for name, p in fitted[key].items():
                x = np.column_stack([m, feats[key][split][name]])
                assert x.shape == (len(y), 9) and np.isfinite(x).all()
                scores[name] = p.decision_function(x)
                pred[name] = p.predict(x)
                assert np.array_equal(pred[name], scores[name] > 0)
            results = {name: getmetrics(y, pred[name], scores[name]) for name in pred}
            allmetrics[key][split] = results
            if split == 'test':
                saved = pd.read_csv(IA / key / 'test_predictions.csv', float_precision='round_trip')
                assert saved.id.tolist() == ids
                assert np.array_equal(saved.original_prediction, pred['original']) and np.array_equal(saved.corrected_prediction, pred['sentiment_only'])
                testpred = pred
                testscores = scores
                testy = y
            for name, result in results.items():
                row = dict(model=key, split=split, method=name, selected_C=selected[key][name]['C'] if name in selected[key] else js(IA / key / 'model_metadata.json')['chosen_C'] if name == 'sentiment_only' else None, **{k: v for k, v in result.items() if k != 'confusion_matrix'})
                summary.append(row)
                confusions[f'{key}/{split}/{name}'] = result['confusion_matrix']
                for reference in ['original', 'sentiment_only']:
                    fixed = (pred[reference] != y) & (pred[name] == y)
                    broken = (pred[reference] == y) & (pred[name] != y)
                    changes.append(dict(model=key, split=split, method=name, reference=reference, incorrect_corrected=int(fixed.sum()), correct_made_incorrect=int(broken.sum()), negative_errors_corrected=int((fixed & (y == 0)).sum()), positive_errors_corrected=int((fixed & (y == 1)).sum()), negative_correct_broken=int((broken & (y == 0)).sum()), positive_correct_broken=int((broken & (y == 1)).sum())))
            frame = pd.DataFrame(dict(id=ids, label=y, source_index=indices[split], margin=m))
            for name in pred:
                frame[name + '_prediction'] = pred[name]
                frame[name + '_decision_score'] = scores[name]
            frame.to_csv(OUT / key / (split + '_predictions.csv'), index=False)
            for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'auroc', 'negative_recall', 'positive_recall', 'predicted_positive']:
                values = np.array([results[f'random_seed_{s}'][metric] for s in SEEDS])
                random_summary.append(dict(model=key, split=split, metric=metric, mean=float(values.mean()), sd=float(values.std(ddof=1)), seeds=10, minimum=float(values.min()), maximum=float(values.max()), uncertainty='sample SD across fixed directions; not an example CI'))
        save(OUT / key / 'all_split_metrics.json', allmetrics[key])
        ci.extend(bootstrap(key, testy, testpred, testscores, allmetrics[key]['test']))
    pd.DataFrame(summary).to_csv(OUT / 'metrics_all_methods.csv', index=False)
    pd.DataFrame(changes).to_csv(OUT / 'prediction_changes.csv', index=False)
    pd.DataFrame(ci).to_csv(OUT / 'paired_differences_95ci.csv', index=False)
    pd.DataFrame(random_summary).to_csv(OUT / 'random_control_mean_sd.csv', index=False)
    pd.DataFrame([row for row in summary if row['method'].startswith('random_seed_')]).to_csv(OUT / 'random_control_per_seed_metrics.csv', index=False)
    save(OUT / 'confusion_matrices.json', confusions)
    return (allmetrics, summary, changes, ci, random_summary)

def report(allmetrics, summary, changes, ci, random_summary, selected):
    lines = ['Stage I-B: Emotion-Guided Sentiment Correction', '', 'Retrospective binary TweetEval evaluation; test previously examined, not independent confirmation. Predictive utility of features, not a causal emotion-specific effect.', 'Frozen train IDs 14,186; validation1,131; test6,347. Exact saved Stage I-A margin and classifiers reused.', 'Cached Stage E post-block text-token pooled representations reused; no new model forwards, fine-tuning, generation, intervention or quantization.', 'Layers: 0.5B=14/24 (index13), 3B=18/36 (index17). Unit-vector projections; no example normalization.', '', '| Method | Qwen 0.5B Accuracy | Qwen 0.5B AUROC | Qwen 3B Accuracy | Qwen 3B AUROC |', '|---|---:|---:|---:|---:|']
    for name, label in [('original', 'Original Qwen'), ('sentiment_only', 'Balanced sentiment-only'), ('raw', 'Raw emotion-guided'), ('denoised', 'Denoised emotion-guided')]:
        a = allmetrics['0.5b']['test'][name]
        b = allmetrics['3b']['test'][name]
        lines.append(f'| {label} | {a['accuracy']:.2%} | {a['auroc']:.4f} | {b['accuracy']:.2%} | {b['auroc']:.4f} |')

    def randomstat(key, metric):
        return next((r for r in random_summary if r['model'] == key and r['split'] == 'test' and (r['metric'] == metric)))
    a = randomstat('0.5b', 'accuracy')
    aa = randomstat('0.5b', 'auroc')
    b = randomstat('3b', 'accuracy')
    bb = randomstat('3b', 'auroc')
    lines.append(f'| Random-feature control (10 seeds) | {a['mean']:.2%} +/- {a['sd']:.2%} | {aa['mean']:.4f} +/- {aa['sd']:.4f} | {b['mean']:.2%} +/- {b['sd']:.2%} | {bb['mean']:.4f} +/- {bb['sd']:.4f} |')
    comparison = []
    for key in MODELS:
        base = allmetrics[key]['test']['sentiment_only']
        lines += ['', f'Model {key}']
        for name in ['raw', 'denoised']:
            m = allmetrics[key]['test'][name]
            ra = randomstat(key, 'accuracy')
            ru = randomstat(key, 'auroc')
            lines += [f'{name}: selected C={selected[key][name]['C']}; accuracy={m['accuracy']:.4%}, macro-F1={m['macro_f1']:.6f}, balanced accuracy={m['balanced_accuracy']:.6f}, AUROC={m['auroc']:.6f}.', f'Against sentiment-only: accuracy {(m['accuracy'] - base['accuracy']) * 100:+.3f} percentage points; AUROC {m['auroc'] - base['auroc']:+.6f}.', f'Negative recall={m['negative_recall']:.6f}; positive recall={m['positive_recall']:.6f}; predicted positives={m['predicted_positive']} (gold2375); confusion [negative,positive]={m['confusion_matrix']}.', f'Against random mean: accuracy {(m['accuracy'] - ra['mean']) * 100:+.3f} points; AUROC {m['auroc'] - ru['mean']:+.6f}; higher accuracy than {sum((m['accuracy'] > allmetrics[key]['test'][f'random_seed_{s}']['accuracy'] for s in SEEDS))}/10 random controls.']
            for ref in ['original', 'sentiment_only', 'random_mean_10']:
                if ref != 'random_mean_10':
                    ch = next((r for r in changes if r['model'] == key and r['split'] == 'test' and (r['method'] == name) and (r['reference'] == ref)))
                    lines.append(f'Compared with {ref}: fixed {ch['incorrect_corrected']}, broke {ch['correct_made_incorrect']}; fixed negative/positive {ch['negative_errors_corrected']}/{ch['positive_errors_corrected']}, broke negative/positive {ch['negative_correct_broken']}/{ch['positive_correct_broken']}.')
                else:
                    lines.append('Compared with random mean (conditional on ten fixed directions):')
                for r in ci:
                    if r['model'] == key and r['method'] == name and (r['reference'] == ref):
                        lines.append(f'  {r['metric']} difference {r['difference']:+.6f}; paired95% CI [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}].')
            comparison.append(dict(model=key, method=name, accuracy=m['accuracy'], auroc=m['auroc'], delta_accuracy_vs_sentiment_only=m['accuracy'] - base['accuracy'], delta_auroc_vs_sentiment_only=m['auroc'] - base['auroc'], delta_accuracy_vs_random_mean=m['accuracy'] - ra['mean'], delta_auroc_vs_random_mean=m['auroc'] - ru['mean']))
        raw = allmetrics[key]['test']['raw']
        den = allmetrics[key]['test']['denoised']
        lines.append(f'Denoised minus raw: accuracy {(den['accuracy'] - raw['accuracy']) * 100:+.3f} points; AUROC {den['auroc'] - raw['auroc']:+.6f}.')
    lines += ['', 'Individual random-control test results:', pd.DataFrame([r for r in summary if r['split'] == 'test' and r['method'].startswith('random_seed_')]).to_string(index=False), '', 'Full train/validation/test metrics:', pd.DataFrame(summary).to_string(index=False), '', 'Interpretation and limitations:', 'Gains indicate information available to a supervised classifier beyond the cached margin; they do not show the frozen language model causally uses these features. Raw and denoised are separate feature definitions, evaluated without selecting a winner for deployment. No random seed was selected using test performance.', 'Random controls use eight orthonormal dimensions versus rank-seven emotion directions; dimensional feature counts match but subspace rank and geometry do not. Both emotion sets and random controls use identical unprocessed pooled activations. Scaler removes feature scale differences; correlation/regularization geometry still differs.', 'Paired class-stratified percentile CIs use2,000 draws seed47 and fixed fitted classifiers. Per-seed intervals condition on each direction set; random mean intervals condition on all ten directions. Random sample SD represents direction variability separately, not a population CI or hierarchical uncertainty. No multiplicity adjustment: nominal intervals are exploratory given many comparisons and test reuse.', 'The source Stage B vector metadata does not contain checkpoint commits. Stage E and I-A commit/tokenizer/config identities and vector hashes match; the missing historical Stage B commit is a provenance limitation, not independently verified historical checkpoint equality.', 'The already frozen prompt retains three-label wording, but this evaluation uses a binary readout and excludes neutral gold examples. Do not compare with Stage A generated-label/three-class accuracy or Stage E margin-free probing results.']
    lines += ['', 'Concise measured interpretation:', 'Both raw and denoised emotion features improve test accuracy and AUROC beyond the fixed sentiment-only classifier. Gains are larger for0.5B (+6.255/+6.208 accuracy points) than3B (+1.544/+1.528 points). Both emotion methods exceed all ten random controls in accuracy and AUROC on each model.', 'Denoising gives slightly lower accuracy/AUROC than raw (0.047/0.016 accuracy points lower for0.5B/3B), while balanced accuracy and positive recall are higher. These small differences do not establish equivalence or a significant raw-versus-denoised difference. Both classes recall improves over sentiment-only for both emotion methods, so gains are not solely a one-class threshold tradeoff.', 'All primary emotion-minus-sentiment-only accuracy and AUROC nominal95% intervals lie above zero, conditional on frozen models. Multiple comparisons and reused test data limit confirmatory interpretation. Random gains are modest, and some seeds reduce performance relative to sentiment-only; all seeds are reported without selection.']
    text = '\n'.join(lines) + '\n'
    (OUT / 'results_summary.txt').write_text(text, encoding='utf-8')
    (OUT / 'RESULTS_REPORT.md').write_text(text, encoding='utf-8')
    pd.DataFrame(comparison).to_csv(OUT / 'summary_comparisons.csv', index=False)
    (OUT / 'README.md').write_text('Stage I-B: Emotion-Guided Sentiment Correction\n\nReproduce/resume (CPU cached-feature reuse, no GPU passes):\n\n```powershell\npython WORK_DIR\\artifacts\\stage_i_emotion_guided_correction\\stage_i_b.py\n```\n\nPhases: `--phase audit`, `extract`, `fit`, `evaluate`, or `all` (default). Audit always checks immutable source hashes and exact baselines. Feature SQLite rows are transactional/checksummed; completed valid rows are reused. Extracted arrays include IDs and feature order. Existing source experiment folders must remain available. All new outputs are isolated here. Model selections are frozen before test evaluation. See IMPLEMENTATION_PLAN.md, repository_audit.json, protocol.json, verification.json, metrics_all_methods.csv, paired_differences_95ci.csv and RESULTS_REPORT.md. Test is reused: retrospective predictive utility, not independent or causal confirmation.\n', encoding='utf-8')
    log('\n' + text)

def finish(tracked, allmetrics):
    for rel, digest in tracked.items():
        assert sha(ROOT / rel) == digest, 'Source changed: ' + rel
    verified = {}
    for key in MODELS:
        for split in SPLITS:
            f = pd.read_csv(OUT / key / (split + '_predictions.csv'), float_precision='round_trip')
            ids = pd.read_csv(OUT / (split + '_ids.csv'))
            assert f.id.tolist() == ids.id.tolist() and f.label.tolist() == ids.label.tolist()
            m = np.load(OUT / key / split / 'margin.float64.npy')
            assert np.array_equal(m, f.margin.to_numpy())
            for name, result in allmetrics[key][split].items():
                actual = getmetrics(f.label.to_numpy(), f[name + '_prediction'].to_numpy(), f[name + '_decision_score'].to_numpy())
                for metric in ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc', 'negative_recall', 'positive_recall'):
                    assert abs(result[metric] - actual[metric]) < 1e-12
            verified[f'{key}/{split}'] = dict(n=len(f), id_label_order=True, margin_exact=True, all_saved_metrics_recomputed=True)
    save(OUT / 'verification.json', dict(complete=True, source_files_unchanged=True, stage_i_a_all_file_hashes_unchanged=True, baselines_reproduced=True, model_identity_checked=True, layer_and_activation_convention_checked=True, original_stage_b_commit_metadata_missing=True, cached_projection_smoke_passed=True, all_features_finite=True, feature_cache_resume_checksum_verification=True, same_frozen_balanced_ids=True, no_split_overlap=True, no_new_forward_passes=True, train_only_scalers=True, all_classifiers_converged=True, all_selections_frozen_before_test=True, splits=verified))
    if (OUT / 'failure.json').exists():
        failure = js(OUT / 'failure.json')
        failure.update(resolved=True, resolution='CSV float_precision=round_trip restores exact cached float margins; audit, full experiment and independent saved-output verification subsequently passed')
        save(OUT / 'failure.json', failure)
    save(OUT / 'state.json', dict(complete=True, phase='complete', models=list(MODELS), retrospective=True, new_forward_passes=0))
    log('Stage I-B complete: saved-output verification passed; Stage I-A and all reused source files unchanged.')

def load_feats():
    return {key: {s: {'raw': np.load(OUT / key / s / 'raw_emotion_scores.float64.npy'), 'denoised': np.load(OUT / key / s / 'denoised_emotion_scores.float64.npy'), **{f'random_seed_{seed}': np.load(OUT / key / s / 'random_scores.float64.npy', mmap_mode='r')[:, seed] for seed in SEEDS}} for s in SPLITS} for key in MODELS}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['audit', 'extract', 'fit', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    log('Starting Stage I-B phase=' + args.phase)
    data, indices, margins, vectors, baseline, hidden, tracked = audit()
    if args.phase == 'audit':
        return
    if args.phase in ['all', 'extract']:
        feats = extract(data, indices, margins, vectors, hidden)
    else:
        assert js(OUT / 'repository_audit.json')['source_sha256'] == tracked
        for key in MODELS:
            for s in SPLITS:
                folder = OUT / key / s
                meta = js(folder / 'features_metadata.json')
                assert meta['complete'] and meta['ids_sha256'] == sha(OUT / (s + '_ids.csv'))
                for name, fname in [('raw', 'raw_emotion_scores.float64.npy'), ('denoised', 'denoised_emotion_scores.float64.npy'), ('random', 'random_scores.float64.npy')]:
                    assert sha(folder / fname) == meta['array_sha256'][name]
                assert sha(folder / 'margin.float64.npy') == meta['margin_sha256'] and np.array_equal(np.load(folder / 'margin.float64.npy'), margins[key][s])
        feats = load_feats()
    if args.phase == 'extract':
        return
    if args.phase in ['all', 'fit']:
        fitted, selected = train(data, indices, margins, feats)
    else:
        selected = js(OUT / 'frozen_selections.json')
        fitted = {key: {name: joblib.load(OUT / key / 'classifiers' / name / 'model.joblib') for name in selected[key]} for key in MODELS}
        for key in MODELS:
            for name in selected[key]:
                assert sha(OUT / key / 'classifiers' / name / 'model.joblib') == selected[key][name]['model_sha256']
    if args.phase == 'fit':
        return
    allmetrics, summary, changes, ci, random_summary = evaluate(data, indices, margins, feats, baseline, fitted, selected)
    report(allmetrics, summary, changes, ci, random_summary, selected)
    finish(tracked, allmetrics)
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        main()
    except Exception as error:
        if OUT.exists():
            save(OUT / 'failure.json', dict(error=repr(error), recommendation='Inspect affected assertion and recorded source/feature metadata; do not substitute model/layer/IDs. Fix only Stage I-B then resume.'))
        raise
