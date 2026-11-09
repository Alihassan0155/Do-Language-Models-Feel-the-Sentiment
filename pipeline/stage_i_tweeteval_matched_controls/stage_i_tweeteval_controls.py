"""TweetEval matched controls: reuse frozen caches and tested SST-2 Stage I helpers."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os, sys
sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import argparse, hashlib, json, sqlite3, importlib.util, time, unicodedata, platform
import numpy as np, pandas as pd, joblib, sklearn
ROOT = WORKSPACE
OUT = ROOT / 'artifacts/stage_i_tweeteval_matched_controls'
IA = ROOT / 'artifacts/stage_i_sentiment_correction'
IB = ROOT / 'artifacts/stage_i_emotion_guided_correction'
SST = ROOT / 'artifacts/stage_i_sst2'
E = ROOT / 'artifacts/stage_e'
spec = importlib.util.spec_from_file_location('sst2_stage_i_reference', source_path(SST / 'stage_i_sst2.py'))
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
u.OUT = OUT
KEYS = u.KEYS
SPLITS = u.SPLITS
sha = u.sha
js = u.js
save = u.save
arrsave = u.arrsave
log = u.log
met = u.met
EMOTIONS = u.EMOTIONS

def rodb(p):
    wal = Path(str(p) + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0, 'Uncheckpointed source database WAL: ' + str(p)
    return sqlite3.connect('file:' + str(p) + '?mode=ro&immutable=1', uri=True)

def normalized(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def audit():
    old = js(IB / 'protocol.json')
    sp = js(SST / 'protocol.json')
    ep = js(E / 'protocol.json')
    iap = js(IA / 'protocol.json')
    assert js(IB / 'verification.json')['complete'] and js(SST / 'verification.json')['complete']
    data = {s: [json.loads(t) for t in (IA / 'reused_inputs' / (s + '.jsonl')).read_text(encoding='utf-8').splitlines()] for s in SPLITS}
    bid = pd.read_csv(IA / 'balanced_training_ids.csv')
    bm = js(IA / 'balanced_training_manifest.json')
    assert sha(IA / 'balanced_training_ids.csv') == bm['training_ids_sha256']
    indices = {'train': bid.example_index.to_numpy(dtype=int), 'validation': np.arange(1131), 'test': np.arange(6347)}
    selected = {s: [data[s][i] for i in indices[s]] for s in SPLITS}
    counts = {'train': [7093, 7093], 'validation': [312, 819], 'test': [3972, 2375]}
    for s in SPLITS:
        r = selected[s]
        ids = pd.read_csv(IB / (s + '_ids.csv'))
        assert ids.id.tolist() == [v['id'] for v in r] and ids.label.tolist() == [v['label'] for v in r] and (ids.source_index.tolist() == indices[s].tolist())
        assert [sum((v['label'] == c for v in r)) for c in (0, 1)] == counts[s]
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not {r['id'] for r in selected[a]} & {r['id'] for r in selected[b]} and (not {normalized(r['text']) for r in selected[a]} & {normalized(r['text']) for r in selected[b]})
    tracked = {}
    for folder in (IA, IB, SST):
        for p in folder.rglob('*'):
            if p.is_file():
                tracked[str(p.relative_to(ROOT))] = sha(p)

    def track(p):
        tracked[str(p.relative_to(ROOT))] = sha(p)
    track(E / 'protocol.json')
    track(E / 'data/manifest.json')
    selection = ROOT / 'artifacts/stage_c_empathetic_dialogues/selected_intervention_layers.json'
    track(selection)
    for s in SPLITS:
        import pyarrow as pa, pyarrow.ipc as ipc
        source = ROOT / 'artifacts/datasets/tweet_eval_sentiment' / s / 'data-00000-of-00001.arrow'
        official = ipc.open_stream(pa.memory_map(str(source), 'r')).read_all().to_pylist()
        track(source)
        for r in data[s]:
            gold = official[r['source_row_index']]
            assert gold['text'] == r['text'] and gold['label'] in (0, 2) and (int(gold['label'] == 2) == r['label'])
            assert hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256']
    hidden = {}
    margins = {}
    baseline = {}
    vectors = {}
    ranks = {}
    reconstruction = {}
    db = rodb(IA / 'reused_inputs/scores.sqlite3')
    for key in KEYS:
        cfg = old['models'][key]
        ec = ep['models'][key]
        ic = iap['scoring_protocol']['models'][key]
        for field in ('model_id', 'checkpoint_commit', 'tokenizer_sha256', 'config_sha256', 'layer', 'total_layers', 'width', 'raw_sha256', 'final_sha256'):
            assert cfg[field] == sp['models'][key][field] and cfg[field] == ec[field], key + ' identity ' + field
        for field in ('model_id', 'checkpoint_commit', 'tokenizer_sha256', 'config_sha256'):
            assert cfg[field] == ic[field]
        assert ec['prompts']['tweet_eval_sentiment'] == ic['prompt']
        assert js(selection)['models'][key]['selected']['layer_1_based'] == cfg['layer']
        assert cfg['layer'] == ec['layer'] and cfg['width'] == ec['width']
        hidden[key] = {}
        margins[key] = {}
        baseline[key] = {}
        vectors[key] = {}
        ranks[key] = {}
        bmets = js(IB / key / 'all_split_metrics.json')
        for name, stage, fn, mn, expected in [('raw', 'stage_b_raw_vectors', 'emotion_directions_raw.float32.npy', 'vectors_metadata.json', cfg['raw_sha256']), ('denoised', 'stage_b_final_vectors', 'emotion_directions_final.float32.npy', 'final_vectors_metadata.json', cfg['final_sha256'])]:
            p = ROOT / 'artifacts' / stage / key / fn
            meta = js(p.parent / mn)
            assert sha(p) == expected and meta['complete'] and (meta['model_id'] == cfg['model_id']) and (tuple(meta['emotion_order']) == EMOTIONS)
            track(p)
            track(p.parent / mn)
            a = np.asarray(np.load(p)[:, cfg['layer'] - 1], dtype=np.float64)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            vectors[key][name] = a
            basis, sing = u.basis(a, sp['rank_relative_tolerance'])
            ranks[key][name] = dict(rank=len(basis), singular_values=sing.tolist(), relative_tolerance=sp['rank_relative_tolerance'])
            assert np.array_equal(basis, np.load(SST / key / 'bases' / (name + '_orthonormal_basis.npy')))
            assert len(basis) == 7, 'Unexpected rank: investigate before controls'
        for s in SPLITS:
            folder = E / 'hidden' / key / 'tweet_eval_sentiment' / s
            meta = js(folder / 'metadata.json')
            assert meta['complete'] and meta['completed_rows'] == len(data[s]) and (meta['source_sha256'] == sha(IA / 'reused_inputs' / (s + '.jsonl'))) and (meta['protocol_sha256'] == sha(E / 'protocol.json')) and (meta['layer'] == cfg['layer']) and (meta['width'] == cfg['width']) and (meta['truncated_examples'] == 0)
            for filename, field in [('pooled.float32.npy', 'hidden_sha256'), ('labels.int64.npy', 'labels_sha256'), ('emotion8.float64.npy', 'emotion_features_sha256')]:
                p = folder / filename
                assert sha(p) == meta[field]
                track(p)
            for filename in ('metadata.json', 'signature.json', 'progress.sqlite3'):
                track(folder / filename)
            h = np.load(folder / 'pooled.float32.npy', mmap_mode='r')
            assert h.shape == (len(data[s]), cfg['width']) and h.dtype == np.float32 and np.isfinite(h).all()
            assert np.array_equal(np.load(folder / 'labels.int64.npy'), [r['label'] for r in data[s]])
            progress = rodb(folder / 'progress.sqlite3')
            rr = progress.execute('select row_index,id,text_sha256,text_tokens from completed order by row_index').fetchall()
            progress.close()
            assert len(rr) == len(data[s])
            assert all((r[:3] == (i, row['id'], row['text_sha256']) and r[3] > 0 for i, (row, r) in enumerate(zip(data[s], rr))))
            hh = np.asarray(h[indices[s]], dtype=np.float64)
            hidden[key][s] = hh
            rr = db.execute('select example_index,id,label,logp_positive,logp_negative,margin from scores where model=? and split=? order by example_index', (key, s)).fetchall()
            assert len(rr) == len(data[s])
            for i, (r, row) in enumerate(zip(rr, data[s])):
                assert r[:3] == (i, row['id'], row['label']) and np.isfinite(r[3:]).all() and (float(np.float32(r[3]) - np.float32(r[4])) == r[5])
            m = np.array([r[5] for r in rr])[indices[s]]
            assert np.array_equal(m, np.load(IB / key / s / 'margin.float64.npy'))
            margins[key][s] = m
            fm = js(IB / key / s / 'features_metadata.json')
            assert fm['complete'] and fm['n'] == len(selected[s]) and (fm['ids_sha256'] == sha(IB / (s + '_ids.csv')))
            for name in ('raw', 'denoised'):
                fp = IB / key / s / (name + '_emotion_scores.float64.npy')
                assert sha(fp) == fm['array_sha256'][name]
                assert np.allclose(hh @ vectors[key][name].T, np.load(fp), atol=1e-10, rtol=1e-12)
            assert sha(IB / key / s / 'random_scores.float64.npy') == fm['array_sha256']['random']
            f = pd.read_csv(IB / key / (s + '_predictions.csv'), float_precision='round_trip')
            assert f.id.tolist() == [r['id'] for r in selected[s]] and f.label.tolist() == [r['label'] for r in selected[s]] and np.array_equal(f.margin, m)
            baseline[key][s] = f
            y = f.label.to_numpy()
            for name in ('original', 'sentiment_only', 'raw', 'denoised', *[f'random_seed_{seed}' for seed in range(10)]):
                res = met(y, f[name + '_prediction'].to_numpy(), f[name + '_decision_score'].to_numpy())
                oldres = bmets[s][name]
                assert all((res[mm] == oldres[mm] for mm in res))
                if name == 'original':
                    assert np.array_equal(f[name + '_prediction'], m > 0)
                elif name == 'sentiment_only':
                    clf = joblib.load(IA / key / 'correction_model.joblib')
                    md = js(IA / key / 'model_metadata.json')
                    assert sha(IA / key / 'correction_model.joblib') == md['model_sha256']
                    assert np.array_equal(clf.predict(m[:, None]), f[name + '_prediction'])
                    assert np.allclose(clf.decision_function(m[:, None]), f[name + '_decision_score'], atol=1e-12, rtol=1e-12)
                else:
                    source = IB / key / 'classifiers' / name
                    clf = joblib.load(source / 'model.joblib')
                    md = js(source / 'model_metadata.json')
                    assert sha(source / 'model.joblib') == md['model_sha256']
                    feat = np.load(IB / key / s / (name + '_emotion_scores.float64.npy')) if name in ('raw', 'denoised') else np.load(IB / key / s / 'random_scores.float64.npy')[:, int(name.split('_')[-1])]
                    x = np.column_stack([m, feat])
                    assert np.array_equal(clf.predict(x), f[name + '_prediction'])
                    assert np.allclose(clf.decision_function(x), f[name + '_decision_score'], atol=1e-12, rtol=1e-12)
                if name != 'original':
                    c = clf.named_steps['classifier']
                    assert c.solver == 'lbfgs' and c.class_weight is None and (c.tol == 1e-05) and (c.max_iter == 3000) and c.fit_intercept and (clf.named_steps['scaler'].n_samples_seen_ == 14186)
        expected = {'0.5b': ['85.88%', '85.88%', '92.14%', '92.09%', '87.10%'], '3b': ['89.81%', '92.48%', '94.03%', '94.01%', '92.58%']}[key]
        point = bmets['test']
        actual = [f'{point[n]['accuracy']:.2%}' for n in ('original', 'sentiment_only', 'raw', 'denoised')] + [f'{np.mean([point[f'random_seed_{i}']['accuracy'] for i in range(10)]):.2%}']
        assert actual == expected, 'Baseline verification failed; stop'
        reconstruction[key] = dict(baselines=actual, ranks=ranks[key], layer=cfg['layer'], width=cfg['width'], all_saved_predictions_reconstructed=True)
        log(f'Audit {key}: cached hidden states/LP margins/model identities/IDs verified; rank7; existing baseline accuracies {actual}')
    db.close()
    protocol = dict(title='TweetEval Stage I rank-matched follow-up', models=old['models'], emotion_order=EMOTIONS, margin_scoring=old['margin_scoring'], feature_convention=old['feature'], C_grid=sp['C_grid'], classifier=sp['classifier'], selection=sp['selection'], random_rank_seeds=sp['random_rank_seeds'], rank_relative_tolerance=sp['rank_relative_tolerance'], pca=sp['pca'], bootstrap=sp['bootstrap'], split_counts=counts, test_status='retrospective: official binary TweetEval test already examined; no independent confirmation', new_qwen_forward_passes=0, new_baseline_fits=0, no_quantization=True, CPU_threads=1, control_methodology_source='artifacts/stage_i_sst2/stage_i_sst2.py; import basis/fit/bootstrap/weighted/met, exact QR convention', historical_stage_b_commit_limitation=sp['checkpoint_history_limitation'], versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__))
    if (OUT / 'protocol.json').exists():
        assert js(OUT / 'protocol.json') == json.loads(json.dumps(protocol))
        assert js(OUT / 'repository_audit.json')['source_sha256'] == tracked, 'Old sources changed; incompatible resume'
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'repository_audit.json', dict(source_sha256=tracked, models=reconstruction, zip_archives_excluded=True, prior_artifacts_read_only=True))
    save(OUT / 'numerical_rank_checks.json', ranks)
    for s in SPLITS:
        p = OUT / 'splits' / (s + '.jsonl')
        p.parent.mkdir(exist_ok=True)
        p.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in selected[s])), encoding='utf-8')
        pd.DataFrame([dict(id=r['id'], label=r['label'], source_index=int(i)) for r, i in zip(selected[s], indices[s])]).to_csv(OUT / (s + '_ids.csv'), index=False)
    save(OUT / 'data_manifest.json', dict(source_balanced_manifest=bm, splits={s: dict(n=len(selected[s]), counts=counts[s], sha256=sha(OUT / 'splits' / (s + '.jsonl')), ids_sha256=sha(OUT / (s + '_ids.csv'))) for s in SPLITS}, same_frozen_examples=True, labels_verified_against_official_arrow=True))
    u.rows = lambda s: selected[s]
    return (protocol, selected, hidden, margins, baseline, vectors, tracked)

def features(protocol, data, hidden, margins, vectors):
    sp = js(SST / 'protocol.json')
    info = {}
    offset = {}
    n = 0
    for s in SPLITS:
        offset[s] = [n, n + len(data[s])]
        n += len(data[s])
    for key in KEYS:
        folder = OUT / key
        folder.mkdir(exist_ok=True)
        if (folder / 'features_metadata.json').exists():
            md = js(folder / 'features_metadata.json')
            assert md['protocol_sha256'] == sha(OUT / 'protocol.json')
            for rel, h in md['sha256'].items():
                assert sha(folder / rel) == h
            info[key] = md
            continue
        h = np.concatenate([hidden[key][s] for s in SPLITS])
        m = np.concatenate([margins[key][s] for s in SPLITS])
        bdir = folder / 'bases'
        bdir.mkdir(exist_ok=True)
        arrays = {}
        ranks = {}
        for name in ('raw', 'denoised'):
            b, singular = u.basis(vectors[key][name], protocol['rank_relative_tolerance'])
            rank = len(b)
            ranks[name] = rank
            arrsave(bdir / (name + '_unit_vectors.npy'), vectors[key][name])
            arrsave(bdir / (name + '_orthonormal_basis.npy'), b)
            assert np.array_equal(b, np.load(SST / key / 'bases' / (name + '_orthonormal_basis.npy')))
            arrays['orthonormal_' + name] = np.column_stack([m, h @ b.T])
        for rank in sorted(set(ranks.values())):
            for seed in protocol['random_rank_seeds']:
                q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(h.shape[1], rank)), mode='reduced')
                q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
                v = q.T
                assert np.allclose(v @ v.T, np.eye(rank), atol=1e-12)
                name = f'random_rank{rank}_seed_{seed}'
                assert np.array_equal(v, np.load(SST / key / 'bases' / (name + '.npy')))
                arrsave(bdir / (name + '.npy'), v)
                arrays[name] = np.column_stack([m, h @ v.T])
            training = h[slice(*offset['train'])]
            pca = u.PCA(n_components=rank, svd_solver='randomized', random_state=42, n_oversamples=16, iterated_power=4, whiten=False).fit(training)
            assert np.allclose(pca.mean_, training.mean(0), atol=1e-12)
            assert pca.n_samples_ == 14186
            joblib.dump(pca, bdir / f'pca_rank{rank}.joblib')
            arrays[f'pca_rank{rank}'] = np.column_stack([m, pca.transform(h)])
            save(bdir / f'pca_rank{rank}.json', dict(fit_split='balanced_train', fit_n=14186, explained_variance_ratio=pca.explained_variance_ratio_.tolist(), same_PCA_hyperparameters_as_SST2=True))
        fdir = folder / 'features'
        fdir.mkdir(exist_ok=True)
        for name, a in arrays.items():
            assert a.shape == (n, 8) and np.isfinite(a).all()
            arrsave(fdir / (name + '.npy'), a)
        md = dict(protocol_sha256=sha(OUT / 'protocol.json'), methods=list(arrays), ranks=ranks, offset=offset, feature_definition='margin, then seven basis coordinates; no example normalization', pca_fit_training_only=True, bases_identical_to_SST2_for_random_and_emotion=True, sha256={str(p.relative_to(folder)): sha(p) for p in [*bdir.rglob('*'), *fdir.rglob('*')] if p.is_file()})
        save(folder / 'features_metadata.json', md)
        info[key] = md
        log(f'Features {key}: {len(arrays)} new controls; exact emotion/random bases match SST-2; balanced-training-only PCA; no Qwen passes')
    return info

def choices():
    result = {}
    for key in KEYS:
        result[key] = {'original': None, 'sentiment_only': js(IA / key / 'model_metadata.json')['chosen_C']}
        for name in ('raw', 'denoised', *[f'random_seed_{i}' for i in range(10)]):
            result[key][name.replace('random_seed_', 'random8_seed_')] = js(IB / key / 'classifiers' / name / 'model_metadata.json')['C']
    return result

def evaluate(protocol, data, info, selected, baseline):
    frozen = js(OUT / 'frozen_before_test.json')
    assert frozen['protocol_sha256'] == sha(OUT / 'protocol.json') and frozen['wrapper_sha256'] == sha(Path(__file__))
    for key in KEYS:
        for name, h in frozen['classifiers'][key].items():
            assert sha(OUT / key / 'classifiers' / name / 'model.joblib') == h
    allmetrics = {}
    metricrows = []
    changes = []
    confusions = {}
    stats = []
    cis = []
    oldC = choices()
    for key in KEYS:
        allmetrics[key] = {}
        models = {name: joblib.load(OUT / key / 'classifiers' / name / 'model.joblib') for name in info[key]['methods']}
        for split in SPLITS:
            f = baseline[key][split]
            y = f.label.to_numpy()
            m = f.margin.to_numpy()
            pred = {}
            scores = {}
            columns = dict(id=f.id.to_numpy(), label=y, source_index=f.source_index.to_numpy(), margin=m)
            for name in ('original', 'sentiment_only', 'raw', 'denoised', *[f'random_seed_{i}' for i in range(10)]):
                actual = name.replace('random_seed_', 'random8_seed_')
                pred[actual] = f[name + '_prediction'].to_numpy()
                scores[actual] = f[name + '_decision_score'].to_numpy()
            for name, p in models.items():
                x = np.load(OUT / key / 'features' / (name + '.npy'), mmap_mode='r')[slice(*info[key]['offset'][split])]
                assert np.array_equal(x[:, 0], m)
                pred[name] = p.predict(x)
                scores[name] = p.decision_function(x)
                assert np.array_equal(pred[name], scores[name] > 0)
            allmetrics[key][split] = {name: met(y, pred[name], s) for name, s in scores.items()}
            for name, res in allmetrics[key][split].items():
                c = selected[key][name]['C'] if name in selected[key] else oldC[key][name]
                metricrows.append(dict(model=key, split=split, method=name, selected_C=c, result_source='new matched control' if name in models else 'unchanged saved Stage I-A/I-B', **{k: v for k, v in res.items() if k != 'confusion_matrix'}))
                confusions[f'{key}/{split}/{name}'] = res['confusion_matrix']
                columns[name + '_prediction'] = pred[name]
                columns[name + '_decision_score'] = scores[name]
                for ref in ('sentiment_only', 'original'):
                    fixed = (pred[ref] != y) & (pred[name] == y)
                    broken = (pred[ref] == y) & (pred[name] != y)
                    changes.append(dict(model=key, split=split, method=name, reference=ref, incorrect_corrected=int(fixed.sum()), correct_made_incorrect=int(broken.sum())))
            pd.DataFrame(columns).to_csv(OUT / key / (split + '_predictions.csv'), index=False)
            for group, prefix in [('random8', 'random8_seed_'), ('random_rank7', 'random_rank7_seed_')]:
                nn = [name for name in scores if name.startswith(prefix)]
                for mm in ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc', 'negative_recall', 'positive_recall'):
                    a = np.array([allmetrics[key][split][name][mm] for name in nn])
                    stats.append(dict(model=key, split=split, method=group, metric=mm, mean=float(a.mean()), sd=float(a.std(ddof=1)), seeds=len(nn), minimum=float(a.min()), maximum=float(a.max())))
            if split == 'test':
                cis.extend(u.bootstrap(key, y, pred, scores, allmetrics[key][split], info[key]['ranks']))
        save(OUT / key / 'all_split_metrics.json', allmetrics[key])
    pd.DataFrame(metricrows).to_csv(OUT / 'metrics_all_methods.csv', index=False)
    pd.DataFrame(changes).to_csv(OUT / 'prediction_changes.csv', index=False)
    pd.DataFrame(stats).to_csv(OUT / 'random_control_mean_sd.csv', index=False)
    pd.DataFrame([r for r in metricrows if r['method'].startswith('random')]).to_csv(OUT / 'random_control_per_seed_metrics.csv', index=False)
    pd.DataFrame(cis).to_csv(OUT / 'paired_differences_95ci.csv', index=False)
    save(OUT / 'confusion_matrices.json', confusions)
    return (allmetrics, metricrows, stats, cis, changes)

def report(allmetrics, metricrows, stats, cis, changes):
    lines = ['TweetEval Stage I: Rank-Matched Controls', '', 'Retrospective binary TweetEval; previously examined official test, not independent confirmation.', 'Same frozen balanced train14,186 (7,093/class), validation1,131 (312 negative/819 positive), test6,347 (3,972 negative/2,375 positive).', 'Cached margins and post-block text-token pooled states reused; zero new Qwen passes, zero old-baseline refits.', 'Layers14/24 and18/36. Both raw/denoised spans verified rank7. Emotion/random bases are bit-identical to SST-2 control bases. PCA independently fitted on balanced TweetEval training states only.', 'Imported tested SST-2 fit/bootstrap/rank/metrics helpers. All53 new classifiers per model frozen before evaluation.', '', '| Method | TweetEval 0.5B Accuracy | AUROC | TweetEval 3B Accuracy | AUROC |', '|---|---:|---:|---:|---:|']
    table = []

    def stat(key, name, mm):
        return next((r for r in stats if r['model'] == key and r['split'] == 'test' and (r['method'] == name) and (r['metric'] == mm)))
    methods = [('original', 'Original Qwen'), ('sentiment_only', 'Balanced sentiment-only'), ('raw', 'Raw emotion-guided'), ('denoised', 'Denoised emotion-guided'), ('random8', 'Random8 (10 seeds)'), ('random_rank7', 'Random rank7 (50 seeds)'), ('pca_rank7', 'PCA rank7'), ('orthonormal_raw', 'Orthonormal raw emotion'), ('orthonormal_denoised', 'Orthonormal denoised emotion')]
    for name, label in methods:
        vals = []
        row = {'method': name}
        for key in KEYS:
            for mm in ('accuracy', 'auroc'):
                if name in ('random8', 'random_rank7'):
                    a = stat(key, name, mm)
                    vals.append(f'{a['mean']:.2%} +/- {a['sd']:.2%}' if mm == 'accuracy' else f'{a['mean']:.4f} +/- {a['sd']:.4f}')
                    row[key + '_' + mm] = a['mean']
                    row[key + '_' + mm + '_seed_sd'] = a['sd']
                else:
                    v = allmetrics[key]['test'][name][mm]
                    vals.append(f'{v:.2%}' if mm == 'accuracy' else f'{v:.4f}')
                    row[key + '_' + mm] = v
        lines.append('| ' + label + ' | ' + ' | '.join(vals) + ' |')
        table.append(row)
    pd.DataFrame(table).to_csv(OUT / 'summary_table.csv', index=False)

    def interval(key, name, ref, metric):
        return next((r for r in cis if r['model'] == key and r['method'] == name and (r['reference'] == ref) and (r['metric'] == metric)))
    for key in KEYS:
        lines += ['', f'Model {key}: measured primary and geometrically matched comparisons']
        for name in ('raw', 'denoised', 'orthonormal_raw', 'orthonormal_denoised'):
            r = allmetrics[key]['test'][name]
            ch = next((v for v in changes if v['model'] == key and v['split'] == 'test' and (v['method'] == name) and (v['reference'] == 'sentiment_only')))
            mc = next((v['selected_C'] for v in metricrows if v['model'] == key and v['split'] == 'test' and (v['method'] == name)))
            lines.append(f'{name}: C={mc}, accuracy={r['accuracy']:.6f}, macro-F1={r['macro_f1']:.6f}, balanced_accuracy={r['balanced_accuracy']:.6f}, AUROC={r['auroc']:.6f}, negative/positive recall={r['negative_recall']:.6f}/{r['positive_recall']:.6f}, confusion {r['confusion_matrix']}; fixed {ch['incorrect_corrected']}, broke {ch['correct_made_incorrect']}.')
            for ref in ('sentiment_only', 'random_rank7_mean', 'pca_rank7'):
                for metric in u.BOOTMET:
                    c = interval(key, name, ref, metric)
                    lines.append(f'  vs {ref} {metric}: difference {c['difference']:+.6f}, paired95% CI [{c['ci_low']:+.6f},{c['ci_high']:+.6f}].')
            for ref in ('random_rank7_mean', 'pca_rank7'):
                a = interval(key, name, ref, 'accuracy')
                b = interval(key, name, ref, 'auroc')
                claims = []
                for m, c in [('accuracy', a), ('AUROC', b)]:
                    state = 'positive nominal interval' if c['ci_low'] > 0 else 'negative nominal interval' if c['ci_high'] < 0 else 'interval includes zero'
                    claims.append(m + ': ' + state)
                lines.append(f'  Evidence for {name} versus {ref}: ' + ', '.join(claims) + '.')
    lines += ['', 'Side-by-side SST-2 / TweetEval comparison (absolute scores are not a controlled dataset comparison):', '| Model | Method | SST-2 Accuracy | SST-2 AUROC | TweetEval Accuracy | TweetEval AUROC |', '|---|---|---:|---:|---:|---:|']
    compare = []
    ss = pd.read_csv(SST / 'metrics_all_methods.csv')
    ssstats = pd.read_csv(SST / 'random_control_mean_sd.csv')
    ssci = pd.read_csv(SST / 'paired_differences_95ci.csv')
    for key in KEYS:
        for name, label in methods:
            if name in ('random8', 'random_rank7'):
                tv = stat(key, name, 'accuracy')['mean']
                ta = stat(key, name, 'auroc')['mean']
                sv = float(ssstats[(ssstats.model == key) & (ssstats.split == 'test') & (ssstats.method == name) & (ssstats.metric == 'accuracy')].iloc[0]['mean'])
                sa = float(ssstats[(ssstats.model == key) & (ssstats.split == 'test') & (ssstats.method == name) & (ssstats.metric == 'auroc')].iloc[0]['mean'])
            else:
                t = allmetrics[key]['test'][name]
                s = ss[(ss.model == key) & (ss.split == 'test') & (ss.method == name)].iloc[0]
                tv = t['accuracy']
                ta = t['auroc']
                sv = float(s.accuracy)
                sa = float(s.auroc)
            lines.append(f'| {key} | {label} | {sv:.2%} | {sa:.4f} | {tv:.2%} | {ta:.4f} |')
            compare.append(dict(model=key, method=name, sst2_accuracy=sv, sst2_auroc=sa, tweeteval_accuracy=tv, tweeteval_auroc=ta))
        for name in ('raw', 'denoised'):
            for ref in ('sentiment_only', 'random_rank7_mean', 'pca_rank7'):
                for metric in ('accuracy', 'auroc'):
                    t = interval(key, name, ref, metric)
                    s = ssci[(ssci.model == key) & (ssci.method == name) & (ssci.reference == ref) & (ssci.metric == metric)].iloc[0]
                    lines.append(f'Cross-dataset {key}/{name} versus {ref} {metric}: SST-2 {s.difference:+.6f} [{s.ci_low:+.6f},{s.ci_high:+.6f}]; TweetEval {t['difference']:+.6f} [{t['ci_low']:+.6f},{t['ci_high']:+.6f}].')
    pd.DataFrame(compare).to_csv(OUT / 'sst2_vs_tweeteval.csv', index=False)
    lines += ['', 'Measured interpretation:']
    for key in KEYS:
        for name in ('raw', 'denoised'):
            base = interval(key, name, 'sentiment_only', 'auroc')
            pc = interval(key, name, 'pca_rank7', 'auroc')
            rc = interval(key, name, 'random_rank7_mean', 'auroc')
            lines.append(f'{key}/{name}: AUROC beyond sentiment-only {base['difference']:+.6f} [{base['ci_low']:+.6f},{base['ci_high']:+.6f}]; beyond rank-random mean {rc['difference']:+.6f} [{rc['ci_low']:+.6f},{rc['ci_high']:+.6f}]; beyond PCA {pc['difference']:+.6f} [{pc['ci_low']:+.6f},{pc['ci_high']:+.6f}].')
        matched = [interval(key, n, 'pca_rank7', m) for n in ('orthonormal_raw', 'orthonormal_denoised') for m in ('accuracy', 'auroc')]
        positive = sum((r['ci_low'] > 0 for r in matched))
        lines.append(f'{key}: {positive}/4 orthonormal-emotion-minus-PCA accuracy/AUROC nominal intervals are wholly positive. Assess these matched comparisons before claiming an emotion-specific advantage; eight correlated coordinates alone can change scaler/L2 geometry.')
    lines += ['', 'Measured protocol differences and hypotheses:', 'Measured: both datasets use14,186 balanced training examples, fixed layers and identical random/emotion bases. SST-2 new internal training-derived test is balanced5,000; TweetEval official binary test is unbalanced6,347 and previously examined. Validation sizes/balance differ. SST-2 frozen T1 is binary; TweetEval frozen T1 retains three labels with binary scoring. SST-2 includes movie-review sentence fragments; TweetEval is social-media text.', 'Hypotheses, not causal conclusions: differences may reflect domain transfer of Reddit-derived GoEmotions vectors, how much information the original margin already captures, different prompt conditioning, and differing class priors. This experiment does not isolate those mechanisms; do not attribute a dataset contrast to any one explanation.', 'A predictive advantage over these random/PCA controls supports the utility of this learned subspace under the specified protocol. It does not establish that each coordinate uniquely encodes the named emotion, exclude every generic subspace, or show causal use of emotions. Null and negative control differences limit emotion-specific claims.', 'CIs:2,000 class-stratified paired draws seed47, frozen models, percentile95%; random mean intervals condition on50 fixed directions. Direction-seed sample SD is reported separately, not combined into an example CI. Nominal intervals have no multiplicity adjustment. TweetEval analyses are retrospective.', 'Historical Stage B checkpoint commits are absent; vector hashes/model IDs/dimensions/downstream pinned checkpoints verified, without claiming historical checkpoint equivalence.', '', 'All test metrics and selected C values:', pd.DataFrame([r for r in metricrows if r['split'] == 'test']).to_string(index=False)]
    text = '\n'.join(lines) + '\n'
    (OUT / 'results_summary.txt').write_text(text, encoding='utf-8')
    (OUT / 'RESULTS_REPORT.md').write_text(text, encoding='utf-8')
    log('\n' + text)
    (OUT / 'README.md').write_text('TweetEval Stage I rank-matched controls\n\n```powershell\npython -B WORK_DIR\\artifacts\\stage_i_tweeteval_matched_controls\\stage_i_tweeteval_controls.py\n```\n\nCPU only, one thread; no Qwen forwards or quantization. Phases `audit`, `features`, `fit`, `evaluate`, `all` (default). Existing Stage I-A/B predictions and features remain unchanged. Reuses tested SST-2 rank, fitting and bootstrap functions by importing its script without calling its main. All source hashes verified before reuse and after completion. Features and fits resume with protocol/hash checks; bootstrap resumes at completed model boundaries. PCA uses balanced training only. Every new classifier is frozen before test evaluation; no test-based seed selection.\n\nSee protocol.json, repository_audit.json, data_manifest.json, numerical_rank_checks.json, frozen_before_test.json, metrics_all_methods.csv, paired_differences_95ci.csv, random_control_mean_sd.csv, sst2_vs_tweeteval.csv, results_summary.txt and verification.json. Source SST-2/Stage E/I-A/I-B directories must remain available. ZIP archives excluded. TweetEval test is previously examined, so the analysis is retrospective.\n', encoding='utf-8')

def verify(protocol, data, info, allmetrics, tracked):
    for rel, h in tracked.items():
        assert sha(ROOT / rel) == h, 'Prior artifact modified: ' + rel
    verified = {}
    cis = pd.read_csv(OUT / 'paired_differences_95ci.csv', float_precision='round_trip')
    for key in KEYS:
        frozen = js(OUT / 'frozen_before_test.json')
        assert frozen['wrapper_sha256'] == sha(Path(__file__))
        for name in info[key]['methods']:
            dest = OUT / key / 'classifiers' / name
            md = js(dest / 'metadata.json')
            search = pd.read_csv(dest / 'C_search.csv', float_precision='round_trip')
            best = max(range(len(search)), key=lambda i: (search.iloc[i].validation_balanced_accuracy, search.iloc[i].validation_macro_f1, -search.iloc[i].C))
            assert float(search.iloc[best].C) == md['C']
            assert sha(dest / 'model.joblib') == md['model_sha256']
            p = joblib.load(dest / 'model.joblib')
            x = np.load(OUT / key / 'features' / (name + '.npy'), mmap_mode='r')
            train = x[slice(*info[key]['offset']['train'])]
            assert p.named_steps['scaler'].n_samples_seen_ == 14186 and np.allclose(p.named_steps['scaler'].mean_, train.mean(0), atol=1e-12)
        for split in SPLITS:
            f = pd.read_csv(OUT / key / (split + '_predictions.csv'), float_precision='round_trip')
            old = pd.read_csv(IB / key / (split + '_predictions.csv'), float_precision='round_trip')
            rr = data[split]
            assert f.id.tolist() == [r['id'] for r in rr] and f.label.tolist() == [r['label'] for r in rr] and np.array_equal(f.margin, old.margin)
            y = f.label.to_numpy()
            for name, res in allmetrics[key][split].items():
                score = f[name + '_decision_score'].to_numpy()
                pred = f[name + '_prediction'].to_numpy()
                assert met(y, pred, score) == res
                if name in info[key]['methods']:
                    p = joblib.load(OUT / key / 'classifiers' / name / 'model.joblib')
                    x = np.load(OUT / key / 'features' / (name + '.npy'), mmap_mode='r')[slice(*info[key]['offset'][split])]
                    assert np.array_equal(p.predict(x), pred) and np.allclose(p.decision_function(x), score, atol=1e-12, rtol=1e-12)
                else:
                    orig = name.replace('random8_seed_', 'random_seed_')
                    assert np.array_equal(pred, old[orig + '_prediction']) and np.array_equal(score, old[orig + '_decision_score'])
            verified[key + '/' + split] = True
        draws = np.load(OUT / key / 'bootstrap/draws.npz')
        for row in cis[cis.model == key].itertuples():
            j = u.BOOTMET.index(row.metric)
            lo, hi = np.quantile(draws[row.method][:, j] - draws[row.reference][:, j], [0.025, 0.975])
            assert abs(lo - row.ci_low) < 1e-12 and abs(hi - row.ci_high) < 1e-12
    save(OUT / 'verification.json', dict(complete=True, old_artifacts_unchanged=True, saved_predictions_models_metrics_reconstructed=verified, all_validation_C_choices_verified=True, train_only_scalers_and_pca=True, no_new_Qwen_passes=True, rank_and_bases_match_SST2=True, baseline_reconstruction_passed=True, all_CIs_match_saved_draws=True, new_classifiers=106, bootstrap_draws_per_model=2000, zip_archives_excluded=True))
    save(OUT / 'state.json', dict(complete=True, phase='complete'))
    log('COMPLETE: saved classifiers/predictions/metrics/C choices/bootstrap intervals and previous-artifact hashes independently verified.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', 'audit', 'features', 'fit', 'evaluate'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol, data, hidden, margins, baseline, vectors, tracked = audit()
    if args.phase == 'audit':
        return
    info = features(protocol, data, hidden, margins, vectors)
    if args.phase == 'features':
        return
    if args.phase in ('all', 'fit'):
        selected = u.fit(protocol, info)
        frozen = js(OUT / 'frozen_before_test.json')
        frozen['wrapper_sha256'] = sha(Path(__file__))
        frozen['immutable_baseline_predictions_sha256'] = {f'{k}/{s}': sha(IB / k / (s + '_predictions.csv')) for k in KEYS for s in SPLITS}
        save(OUT / 'frozen_before_test.json', frozen)
    else:
        selected = js(OUT / 'frozen_before_test.json')['selected']
    if args.phase == 'fit':
        return
    allmetrics, metricrows, stats, cis, changes = evaluate(protocol, data, info, selected, baseline)
    report(allmetrics, metricrows, stats, cis, changes)
    verify(protocol, data, info, allmetrics, tracked)
if __name__ == '__main__':
    main()
