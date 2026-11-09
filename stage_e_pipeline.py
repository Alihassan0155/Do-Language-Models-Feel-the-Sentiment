from __future__ import annotations
'Sequential E1-E5: fresh selected-layer states, matched probes and locked evaluation.'
from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import csv
import hashlib
import json
import sqlite3
import warnings
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from stage_e_common import ROOT, ART, E, MODELS, DATASETS, SPLITS, EMOTIONS, atomic_json, read_rows, selection, sha
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
SEEDS = list(range(10))

def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def units(values):
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm <= 0):
        raise ValueError('Emotion directions must be finite and nonzero.')
    return values / norm

def rank(values):
    singular = np.linalg.svd(values, compute_uv=False)
    return int(np.sum(singular > singular[0] * 1e-05))

def emotion_inputs(key):
    layer, selected_hash = selection(key)
    final_folder = ART / 'stage_b_final_vectors' / key
    raw_folder = ART / 'stage_b_raw_vectors' / key
    final_meta = json.loads((final_folder / 'final_vectors_metadata.json').read_text(encoding='utf-8'))
    raw_meta = json.loads((raw_folder / 'vectors_metadata.json').read_text(encoding='utf-8'))
    raw_path = raw_folder / 'emotion_directions_raw.float32.npy'
    final_path = final_folder / 'emotion_directions_final.float32.npy'
    for meta in (final_meta, raw_meta):
        if not meta.get('complete') or meta['model_id'] != MODELS[key] or tuple(meta['emotion_order']) != EMOTIONS:
            raise ValueError(f'Stage B model/order/completion mismatch: {key}')
    if final_meta['raw_directions_sha256'] != sha(raw_path):
        raise ValueError('Raw/final Stage B provenance mismatch.')
    raw, final = (np.load(raw_path), np.load(final_path))
    expected = (8, final_meta['layers'], final_meta['hidden_size'])
    if raw.shape != expected or final.shape != expected or (not 1 <= layer <= final.shape[1]):
        raise ValueError('Stage B direction shapes differ.')
    directions = {'emotion8': units(final[:, layer - 1]), 'emotion_raw8': units(raw[:, layer - 1])}
    info = dict(model_id=MODELS[key], layer=layer, total_layers=final.shape[1], width=final.shape[2], selection_sha256=selected_hash, raw_sha256=sha(raw_path), final_sha256=sha(final_path), emotion_effective_rank=rank(directions['emotion8']), raw_emotion_effective_rank=rank(directions['emotion_raw8']))
    return (directions, info)

def methods_for(info):
    methods = ['emotion8', 'emotion_raw8', 'pca8', 'full_hidden']
    methods += [f'random8_seed_{seed}' for seed in SEEDS]
    r = info['emotion_effective_rank']
    if r < 8:
        methods += [f'pca_rank{r}']
        methods += [f'random_rank{r}_seed_{seed}' for seed in SEEDS]
    return methods

def protocol(batch_size=8):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import transformers
    import nnsight
    from transformers import AutoConfig, AutoTokenizer
    from stage_d import prompt_for
    config = dict(schema=2, seed=42, prompt='template_1', emotion_order=list(EMOTIONS), extraction_batch_size=batch_size, cpu_threads=1, cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), feature_definition='pooled float32 text-token residual dot unit direction; no example L2 normalization', storage='fresh single selected-layer float32 states shared by every probe; no Stage D feature reuse', precision='float16 model forward; float32 token pooling and hidden storage; float64 projections/probe fitting', random_seeds=SEEDS, random_rule='training-independent isotropic Gaussian orthonormal directions', pca_rule='top 8 PCs of training pooled states only; training centering; no whitening', rank_controls='also compare random/PCA dimensions matched to effective emotion rank when below 8', scaling='fit StandardScaler on training only', regularization_grid=C_GRID, regularization_selection='maximize validation macro-F1; exact ties choose smallest C', optimizer='lbfgs, warm-start ascending C, tol=1e-5, max_iter=3000; convergence required', final_fit='train only; no refit on validation', test_rule='freeze all model/dataset/method probes before evaluating any tests; decision threshold 0.5', labels={'0': 'negative', '1': 'positive'}, neutral_policy='excluded', sst2_test='872 official validation examples as previously observed labeled proxy, not new unbiased test', reset_note='This is a restart after earlier E tuning outputs; deleting files does not erase prior observations.', dataset_manifest_sha256=sha(E / 'data' / 'manifest.json'), versions=dict(torch=torch.__version__, transformers=transformers.__version__, nnsight=nnsight.__version__, sklearn=sklearn.__version__, numpy=np.__version__), models={})
    for key in MODELS:
        _, info = emotion_inputs(key)
        cfg = AutoConfig.from_pretrained(MODELS[key], local_files_only=True, revision=model_revision(MODELS[key]))
        tok = AutoTokenizer.from_pretrained(MODELS[key], use_fast=True, local_files_only=True, revision=model_revision(MODELS[key]))
        info.update(config_sha256=digest_json(cfg.to_dict()), checkpoint_commit=getattr(cfg, '_commit_hash', None), tokenizer_sha256=hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(), methods=methods_for(info), prompts={d: prompt_for(key, d, 'template_1') for d in DATASETS}, max_tokens=256 if key == 'bert_large_uncased' else 1024)
        config['models'][key] = info
    path = E / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config:
        raise ValueError('Locked Stage E protocol or inputs changed; do not mix runs. Review before --reset.')
    if not path.exists():
        atomic_json(path, config)
    return config

def assert_protocol(p):
    if sha(E / 'data' / 'manifest.json') != p['dataset_manifest_sha256']:
        raise ValueError('Split manifest changed.')
    for key, saved in p['models'].items():
        _, current = emotion_inputs(key)
        for field in ('selection_sha256', 'raw_sha256', 'final_sha256', 'layer', 'width'):
            if current[field] != saved[field]:
                raise ValueError('Stage B/C inputs changed during Stage E.')

def hidden_folder(key, dataset, split):
    return E / 'hidden' / key / dataset / split

def extract_models(p, batch_size=8, limit=None):
    import gc
    import torch
    from nnsight import NNsight
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from stage_d import token_batch
    if batch_size != p['extraction_batch_size']:
        raise ValueError('Batch size differs from the frozen extraction protocol.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for fresh Stage E extraction.')
    torch.manual_seed(p['seed'])
    np.random.seed(p['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('CUDA:', torch.cuda.get_device_name(0), flush=True)
    protocol_hash = sha(E / 'protocol.json')
    for key, info in p['models'].items():
        final_directions = emotion_inputs(key)[0]['emotion8']
        base = wrapped = tokenizer = None
        try:
            for dataset in DATASETS:
                for split in SPLITS:
                    rows, entry = read_rows(dataset, split)
                    folder = hidden_folder(key, dataset, split)
                    folder.mkdir(parents=True, exist_ok=True)
                    signature = dict(protocol_sha256=protocol_hash, model=key, dataset=dataset, split=split, source_sha256=entry['sha256'], n=len(rows), width=info['width'], layer=info['layer'])
                    sig_path = folder / 'signature.json'
                    if sig_path.exists() and json.loads(sig_path.read_text(encoding='utf-8')) != signature:
                        raise ValueError('Hidden checkpoint recipe changed.')
                    if not sig_path.exists():
                        atomic_json(sig_path, signature)
                    labels = np.array([r['label'] for r in rows], dtype=np.int64)
                    labels_path = folder / 'labels.int64.npy'
                    if labels_path.exists() and (not np.array_equal(np.load(labels_path), labels)):
                        raise ValueError('Hidden checkpoint labels changed.')
                    if not labels_path.exists():
                        np.save(labels_path, labels)
                    db = sqlite3.connect(folder / 'progress.sqlite3')
                    try:
                        db.execute('CREATE TABLE IF NOT EXISTS completed(row_index INTEGER PRIMARY KEY, id TEXT, text_sha256 TEXT, text_tokens INTEGER)')
                        done = {r[0] for r in db.execute('SELECT row_index FROM completed')}
                        path = folder / 'pooled.float32.npy'
                        if path.exists():
                            h = np.load(path, mmap_mode='r+')
                            if h.shape != (len(rows), info['width']) or h.dtype != np.float32:
                                raise ValueError('Hidden cache dimensions/dtype differ.')
                        else:
                            if done:
                                raise ValueError('Hidden state array missing despite completed checkpoint rows.')
                            h = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), info['width']))
                        feature_path = folder / 'emotion8.float64.npy'
                        if feature_path.exists():
                            features = np.load(feature_path, mmap_mode='r+')
                            if features.shape != (len(rows), 8) or features.dtype != np.float64:
                                raise ValueError('Emotion feature array dimensions/dtype differ.')
                        else:
                            if done:
                                raise ValueError('Emotion features missing despite completed hidden rows.')
                            features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float64, shape=(len(rows), 8))
                        selected = min(limit, len(rows)) if limit else len(rows)
                        pending = [i for i in range(selected) if i not in done]
                        print(f'E1 {key}/{dataset}/{split}: {len(done)}/{len(rows)} saved; {len(pending)} pending', flush=True)
                        if pending and base is None:
                            tokenizer = AutoTokenizer.from_pretrained(MODELS[key], use_fast=True, local_files_only=True, revision=model_revision(MODELS[key]))
                            if hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() != info['tokenizer_sha256']:
                                raise ValueError('Tokenizer differs from locked protocol.')
                            tokenizer.padding_side = 'right' if key == 'bert_large_uncased' else 'left'
                            if tokenizer.pad_token is None:
                                tokenizer.pad_token = tokenizer.eos_token
                            loader = AutoModel if key == 'bert_large_uncased' else AutoModelForCausalLM
                            base = loader.from_pretrained(MODELS[key], dtype=torch.float16, local_files_only=True, revision=model_revision(MODELS[key])).eval().to('cuda')
                            actual_layers = len(base.encoder.layer) if key == 'bert_large_uncased' else len(base.model.layers)
                            if actual_layers != info['total_layers'] or base.config.hidden_size != info['width']:
                                raise ValueError('Loaded model differs from Stage B.')
                            if getattr(base.config, '_commit_hash', None) != info['checkpoint_commit']:
                                raise ValueError('Loaded model snapshot differs from the locked protocol.')
                            wrapped = NNsight(base)
                        for start in range(0, len(pending), batch_size):
                            indices = pending[start:start + batch_size]
                            inputs, mask, counts, clipped = token_batch(tokenizer, key, info['prompts'][dataset], [rows[i] for i in indices])
                            if any(clipped):
                                raise ValueError('A Stage E example exceeds the frozen token limit; no silent truncation is permitted.')
                            weights = mask.unsqueeze(-1)
                            denominator = weights.sum(dim=1)
                            kwargs = dict(attention_mask=inputs['attention_mask'])
                            if key == 'bert_large_uncased':
                                kwargs['token_type_ids'] = inputs.get('token_type_ids')
                            else:
                                kwargs['use_cache'] = False
                            with torch.inference_mode():
                                with wrapped.trace(inputs['input_ids'], **kwargs):
                                    if key == 'bert_large_uncased':
                                        residual = wrapped.encoder.layer[info['layer'] - 1].nns_output[0]
                                    else:
                                        residual = wrapped.model.layers[info['layer'] - 1].output
                                    pooled = ((residual.float() * weights).sum(dim=1) / denominator).cpu().save()
                            values = pooled.numpy().astype(np.float32, copy=False)
                            if values.shape != (len(indices), info['width']) or not np.isfinite(values).all():
                                raise ValueError('Invalid extracted residual shape/values.')
                            h[indices] = values
                            h.flush()
                            features[indices] = values.astype(np.float64) @ final_directions.T
                            features.flush()
                            with db:
                                db.executemany('INSERT INTO completed VALUES (?,?,?,?)', [(i, rows[i]['id'], rows[i]['text_sha256'], c) for i, c in zip(indices, counts)])
                            if start % (batch_size * 50) == 0 or start + len(indices) == len(pending):
                                print(f'  {len(done) + start + len(indices)}/{len(rows)}', flush=True)
                        total = db.execute('SELECT COUNT(*) FROM completed').fetchone()[0]
                        complete = total == len(rows)
                        atomic_json(folder / 'metadata.json', dict(signature, complete=complete, completed_rows=total, labels_sha256=sha(labels_path), hidden_sha256=sha(path) if complete else None, emotion_features_sha256=sha(feature_path) if complete else None, storage_dtype='float32', fresh_forward=True, truncated_examples=0))
                        del h, features
                    finally:
                        db.close()
        finally:
            del wrapped, base, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

def load_hidden(p, key, dataset, split):
    rows, entry = read_rows(dataset, split)
    folder = hidden_folder(key, dataset, split)
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if not meta.get('complete') or meta['protocol_sha256'] != sha(E / 'protocol.json') or meta['source_sha256'] != entry['sha256'] or (meta['completed_rows'] != len(rows)):
        raise ValueError(f'Incomplete/incompatible fresh hidden cache: {folder}')
    path = folder / 'pooled.float32.npy'
    if sha(path) != meta['hidden_sha256'] or sha(folder / 'labels.int64.npy') != meta['labels_sha256']:
        raise ValueError('Hidden cache checksum changed.')
    h = np.load(path, mmap_mode='r')
    y = np.load(folder / 'labels.int64.npy')
    if h.shape != (len(rows), p['models'][key]['width']) or h.dtype != np.float32 or (not np.isfinite(h).all()) or (not np.array_equal(y, [r['label'] for r in rows])) or (set(y.tolist()) != {0, 1}):
        raise ValueError('Hidden features/labels/order changed.')
    features_path = folder / 'emotion8.float64.npy'
    if sha(features_path) != meta['emotion_features_sha256']:
        raise ValueError('Emotion features changed.')
    emotion_features = np.load(features_path, mmap_mode='r')
    if emotion_features.shape != (len(rows), 8) or not np.isfinite(emotion_features).all():
        raise ValueError('Invalid saved Emotion8 feature shape/values.')
    db = sqlite3.connect(f'file:{(folder / 'progress.sqlite3').as_posix()}?mode=ro', uri=True)
    try:
        recorded = {r[0]: r[1:] for r in db.execute('SELECT row_index,id,text_sha256 FROM completed')}
    finally:
        db.close()
    if len(recorded) != len(rows) or any((recorded.get(i) != (r['id'], r['text_sha256']) for i, r in enumerate(rows))):
        raise ValueError('Source identities differ from the hidden checkpoint.')
    return (h, y, rows, meta)

def fit_pca(h, seed):
    estimator = PCA(n_components=8, svd_solver='randomized', random_state=seed, n_oversamples=16, iterated_power=4, whiten=False)
    estimator.fit(np.asarray(h, dtype=np.float64))
    return estimator

def random_directions(width, seed, count=8):
    if width < count:
        raise ValueError('Hidden width is smaller than the requested random subspace.')
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(width, 8)), mode='reduced')
    q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
    return q[:, :count].T

def recipes(p, key, htrain):
    directions, info = emotion_inputs(key)
    estimator = fit_pca(htrain, p['seed'])
    recipes = {'emotion8': dict(kind='projection', directions=directions['emotion8'], offset=np.zeros(8), feature_order=EMOTIONS), 'emotion_raw8': dict(kind='projection', directions=directions['emotion_raw8'], offset=np.zeros(8), feature_order=EMOTIONS), 'pca8': dict(kind='projection', directions=estimator.components_, offset=estimator.mean_ @ estimator.components_.T, pca_train_mean=estimator.mean_, explained_variance_ratio=estimator.explained_variance_ratio_), 'full_hidden': dict(kind='full', dimensions=info['width'])}
    effective = info['emotion_effective_rank']
    if effective < 8:
        directions_pca = estimator.components_[:effective]
        recipes[f'pca_rank{effective}'] = dict(kind='projection', directions=directions_pca, offset=estimator.mean_ @ directions_pca.T, pca_train_mean=estimator.mean_, explained_variance_ratio=estimator.explained_variance_ratio_[:effective])
    for seed in SEEDS:
        directions_random = random_directions(info['width'], seed)
        recipes[f'random8_seed_{seed}'] = dict(kind='projection', directions=directions_random, offset=np.zeros(8), seed=seed)
        if effective < 8:
            recipes[f'random_rank{effective}_seed_{seed}'] = dict(kind='projection', directions=directions_random[:effective], offset=np.zeros(effective), seed=seed)
    if set(recipes) != set(info['methods'] if 'methods' in info else methods_for(info)):
        raise ValueError('Control recipes differ from the frozen method list.')
    return recipes

def project(h, recipe):
    if recipe['kind'] == 'full':
        return np.asarray(h, dtype=np.float64)
    return np.asarray(h, dtype=np.float64) @ recipe['directions'].T - recipe['offset']

def metric_values(y, prediction, probability):
    return dict(accuracy=float(accuracy_score(y, prediction)), macro_f1=float(f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)), auroc=float(roc_auc_score(y, probability)))

def fit_probe(x, y, xv, yv, grid):
    """Fit one scaler, tune C only on validation, and retain a train-only classifier."""
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    zv = scaler.transform(xv)
    lr = LogisticRegression(C=grid[0], solver='lbfgs', max_iter=3000, tol=1e-05, warm_start=True, class_weight=None, random_state=42)
    best = fitted = None
    search = []
    for c in sorted(grid):
        lr.set_params(C=c)
        for iteration_limit in (3000, 10000, 30000):
            lr.set_params(max_iter=iteration_limit)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    lr.fit(z, y)
                break
            except ConvergenceWarning:
                if iteration_limit == 30000:
                    raise
                print(f'  C={c:g}: not converged at {iteration_limit} iterations; retrying', flush=True)
        if not np.array_equal(lr.classes_, [0, 1]):
            raise ValueError('Unexpected binary classifier classes.')
        probability = lr.predict_proba(zv)[:, 1]
        prediction = lr.predict(zv)
        values = metric_values(yv, prediction, probability)
        item = dict(C=c, **values, iterations=int(lr.n_iter_[0]), iteration_limit=int(lr.max_iter))
        search.append(item)
        if best is None or (item['macro_f1'], -c) > (best['macro_f1'], -best['C']):
            best = item
            fitted = copy.deepcopy(lr)
    pipeline = Pipeline([('scale', scaler), ('logistic', fitted)])
    return (pipeline, best, search)

def fit_all(p):
    for key, info in p['models'].items():
        for dataset in DATASETS:
            h, y, _, train_meta = load_hidden(p, key, dataset, 'train')
            hv, yv, _, val_meta = load_hidden(p, key, dataset, 'validation')
            folder = E / 'probes' / key / dataset
            folder.mkdir(parents=True, exist_ok=True)
            recipe_path = folder / 'transforms.joblib'
            recipe_signature = dict(protocol_sha256=sha(E / 'protocol.json'), train_hidden_sha256=train_meta['hidden_sha256'], pca_fit_split='train')
            if recipe_path.exists():
                saved = json.loads((folder / 'transforms_metadata.json').read_text(encoding='utf-8'))
                if saved['signature'] != recipe_signature or sha(recipe_path) != saved['sha256']:
                    raise ValueError('Saved train-only control transforms changed.')
                transform = joblib.load(recipe_path)
            else:
                transform = recipes(p, key, h)
                tmp = folder / 'transforms.tmp.joblib'
                joblib.dump(transform, tmp, compress=3)
                tmp.replace(recipe_path)
                atomic_json(folder / 'transforms_metadata.json', dict(signature=recipe_signature, sha256=sha(recipe_path)))
            if set(transform) != set(info['methods']):
                raise ValueError('Missing or extra control methods.')
            for method in info['methods']:
                target = folder / method
                target.mkdir(parents=True, exist_ok=True)
                signature = dict(protocol_sha256=sha(E / 'protocol.json'), model=key, dataset=dataset, method=method, layer=info['layer'], train_source_sha256=train_meta['source_sha256'], validation_source_sha256=val_meta['source_sha256'], train_hidden_sha256=train_meta['hidden_sha256'], validation_hidden_sha256=val_meta['hidden_sha256'], transforms_sha256=sha(recipe_path))
                result_path = target / 'training_results.json'
                if result_path.exists():
                    old = json.loads(result_path.read_text(encoding='utf-8'))
                    if not old.get('complete') or old['signature'] != signature or sha(target / 'probe.joblib') != old['probe_sha256'] or (sha(target / 'regularization_search.csv') != old['search_sha256']):
                        raise ValueError('Saved fitted probe changed.')
                    print(f'E2/E4 {key}/{dataset}/{method}: complete; skip', flush=True)
                    continue
                x, xv = (project(h, transform[method]), project(hv, transform[method]))
                if method == 'emotion8':
                    for split_name, projected in (('train', x), ('validation', xv)):
                        saved = np.load(hidden_folder(key, dataset, split_name) / 'emotion8.float64.npy', mmap_mode='r')
                        if not np.allclose(projected, saved, atol=1e-10, rtol=1e-10):
                            raise ValueError('E1 and probe Emotion8 coordinates disagree.')
                print(f'E2/E4 fitting {key}/{dataset}/{method}, dimension={x.shape[1]}', flush=True)
                pipeline, best, search = fit_probe(x, y, xv, yv, p['regularization_grid'])
                tmp = target / 'probe.tmp.joblib'
                joblib.dump(pipeline, tmp, compress=3)
                tmp.replace(target / 'probe.joblib')
                with (target / 'regularization_search.csv').open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(search[0]))
                    writer.writeheader()
                    writer.writerows(search)
                atomic_json(result_path, dict(signature=signature, complete=True, dimensions=x.shape[1], n_train=len(y), n_validation=len(yv), chosen_C=best['C'], validation_metrics=best, probe_sha256=sha(target / 'probe.joblib'), search_sha256=sha(target / 'regularization_search.csv'), train_only_scaler=True, test_used_for_fitting=False))
                print(f'  selected C={best['C']:g}; validation macro-F1={best['macro_f1']:.4f}', flush=True)
                del x, xv
            del h, hv

def evaluate_all(p):
    for key, info in p['models'].items():
        for dataset in DATASETS:
            folder = E / 'probes' / key / dataset
            transform_meta = json.loads((folder / 'transforms_metadata.json').read_text(encoding='utf-8'))
            if transform_meta['signature']['protocol_sha256'] != sha(E / 'protocol.json') or sha(folder / 'transforms.joblib') != transform_meta['sha256']:
                raise ValueError('Frozen feature transform changed.')
            for method in info['methods']:
                path = folder / method
                r = json.loads((path / 'training_results.json').read_text(encoding='utf-8'))
                if not r['complete'] or r['signature']['protocol_sha256'] != sha(E / 'protocol.json') or sha(path / 'probe.joblib') != r['probe_sha256'] or (sha(path / 'regularization_search.csv') != r['search_sha256']):
                    raise ValueError('Finish fitting all probes before test evaluation.')
    for key, info in p['models'].items():
        for dataset in DATASETS:
            h, y, rows, test_meta = load_hidden(p, key, dataset, 'test')
            source = E / 'probes' / key / dataset
            transforms = joblib.load(source / 'transforms.joblib')
            for method in info['methods']:
                training_path = source / method / 'training_results.json'
                trained = json.loads(training_path.read_text(encoding='utf-8'))
                target = E / 'evaluation' / key / dataset / method
                target.mkdir(parents=True, exist_ok=True)
                signature = dict(protocol_sha256=sha(E / 'protocol.json'), training_results_sha256=sha(training_path), test_source_sha256=test_meta['source_sha256'], test_hidden_sha256=test_meta['hidden_sha256'], threshold=0.5)
                result_path = target / 'test_results.json'
                if result_path.exists():
                    old = json.loads(result_path.read_text(encoding='utf-8'))
                    if old['signature'] != signature or sha(target / 'predictions.csv') != old['predictions_sha256']:
                        raise ValueError('Locked test results changed; do not mix test runs.')
                    print(f'E3 {key}/{dataset}/{method}: locked results reused', flush=True)
                    continue
                pipeline = joblib.load(source / method / 'probe.joblib')
                x = project(h, transforms[method])
                probability = pipeline.predict_proba(x)[:, 1]
                prediction = pipeline.predict(x)
                if not np.isfinite(probability).all() or set(prediction.tolist()) - {0, 1}:
                    raise ValueError('Invalid classifier predictions.')
                with (target / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id', 'gold_label', 'predicted_label', 'probability_positive'])
                    writer.writerows(((r['id'], int(a), int(b), float(c)) for r, a, b, c in zip(rows, y, prediction, probability)))
                values = metric_values(y, prediction, probability)
                atomic_json(result_path, dict(signature=signature, complete=True, model=key, dataset=dataset, method=method, layer=info['layer'], dimensions=trained['dimensions'], chosen_C=trained['chosen_C'], n=len(y), gold_counts={'negative': int((y == 0).sum()), 'positive': int((y == 1).sum())}, metrics=values, predictions_sha256=sha(target / 'predictions.csv'), test_tuning=False, test_kind='previously observed official-validation proxy' if dataset == 'sst2' else 'official binary test'))
                print(f'E3 {key}/{dataset}/{method}: accuracy={values['accuracy']:.4f}, F1={values['macro_f1']:.4f}, AUROC={values['auroc']:.4f}', flush=True)
                del x
            del h

def report(p):
    names = {'0.5b': 'Qwen 0.5B', '1.5b': 'Qwen 1.5B', '3b': 'Qwen 3B', 'bert_large_uncased': 'BERT-large'}
    long_rows = []
    for key, info in p['models'].items():
        for dataset in DATASETS:
            for method in info['methods']:
                folder = E / 'evaluation' / key / dataset / method
                path = folder / 'test_results.json'
                if not path.exists():
                    raise ValueError('Complete all fixed test evaluations before making the final comparison.')
                r = json.loads(path.read_text(encoding='utf-8'))
                if not r['complete'] or r['signature']['protocol_sha256'] != sha(E / 'protocol.json') or sha(folder / 'predictions.csv') != r['predictions_sha256']:
                    raise ValueError('Comparison contains an incompatible/corrupt test result.')
                long_rows.append(dict(model=key, dataset=dataset, method=method, layer=info['layer'], dimensions=r['dimensions'], n=r['n'], chosen_C=r['chosen_C'], **r['metrics']))
    with (E / 'test_metrics_all_methods.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    lines = ['STAGE E1-E5 - SENTIMENT DECODING RESULTS', '', 'Fresh float32 selected-layer pooled states; fixed T1; same states/splits for every method.', 'Coordinates: h dot unit direction, without normalizing each example h.', 'L2 logistic regression; scaler fit on train only; C selected on tuning macro-F1.', 'All probes frozen before any final test evaluation. F1 means macro-F1.', 'TweetEval is binary (neutral excluded). SST-2 uses 872 official-validation examples as a', 'previously observed labeled test proxy; it is not a new unbiased test benchmark.', 'Random controls: ten fixed seeds; report mean +/- sample SD, never best seed.', 'Random-seed SD is variation across directions, not a test-set confidence interval.', 'Emotion coordinates may have effective rank below 8; rank-matched controls are included.', 'Full hidden is a regularized reference, not a guaranteed upper bound.', '']
    for metric, title in (('accuracy', 'ACCURACY'), ('macro_f1', 'MACRO-F1'), ('auroc', 'AUROC')):
        wide = []
        lines += [title + ' - FINAL TEST / SST-2 PROXY', 'Model        Dataset       Emotion8 Random8 mean+/-SD    PCA8     Full h    Raw8', '-' * 88]
        for key, info in p['models'].items():
            for dataset in DATASETS:
                group = [r for r in long_rows if r['model'] == key and r['dataset'] == dataset]
                by_method = {r['method']: r for r in group}
                rand = np.array([by_method[f'random8_seed_{seed}'][metric] for seed in SEEDS])
                row = dict(model=names[key], dataset=dataset, layer=info['layer'], emotion8=by_method['emotion8'][metric], random8_mean=float(rand.mean()), random8_sd=float(rand.std(ddof=1)), pca8=by_method['pca8'][metric], full_hidden=by_method['full_hidden'][metric], emotion_raw8=by_method['emotion_raw8'][metric], emotion_effective_rank=info['emotion_effective_rank'], random_seeds=len(SEEDS))
                effective = info['emotion_effective_rank']
                if effective < 8:
                    rand_rank = np.array([by_method[f'random_rank{effective}_seed_{seed}'][metric] for seed in SEEDS])
                    row.update(random_rankmatched_mean=float(rand_rank.mean()), random_rankmatched_sd=float(rand_rank.std(ddof=1)), pca_rankmatched=by_method[f'pca_rank{effective}'][metric])
                wide.append(row)
                ds = 'SST-2 proxy' if dataset == 'sst2' else 'TweetEval'
                lines.append(f'{names[key]:<12} {ds:<13} {row['emotion8']:.4f}   {row['random8_mean']:.4f}+/-{row['random8_sd']:.4f}   {row['pca8']:.4f}   {row['full_hidden']:.4f}   {row['emotion_raw8']:.4f}')
        fields = sorted(set().union(*(r.keys() for r in wide)))
        with (E / f'comparison_{metric}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(wide)
        lines.append('')
    lines += ['RANKS AND COMPLETENESS']
    for key, info in p['models'].items():
        lines.append(f'{names[key]}: layer {info['layer']}/{info['total_layers']}, hidden width {info['width']}, denoised rank {info['emotion_effective_rank']}/8, raw rank {info['raw_emotion_effective_rank']}/8.')
    lines += [f'All {len(long_rows)} method x model x dataset test evaluations complete.', 'E1-E5 complete. Detailed class predictions are saved under evaluation/.', 'Rank-matched results are in the comparison CSVs and test_metrics_all_methods.csv.', 'Decoding establishes available sentiment information, not causal use of emotion directions.', 'Compare Emotion8 to both generic 8-D and rank-matched controls before making specificity claims.', 'No bootstrap test CIs or paired significance tests are included in this stage.', '']
    (E / 'stage_e_results_summary.txt').write_bytes('\n'.join(lines).encode('utf-8'))
    atomic_json(E / 'completion.json', dict(complete=True, protocol_sha256=sha(E / 'protocol.json'), test_evaluations=len(long_rows), models=list(p['models']), datasets=list(DATASETS), output_sha256={name: sha(E / name) for name in ['stage_e_results_summary.txt', 'test_metrics_all_methods.csv', 'comparison_accuracy.csv', 'comparison_macro_f1.csv', 'comparison_auroc.csv']}))
    print(f'E5 comparison and summary saved: {E}', flush=True)
