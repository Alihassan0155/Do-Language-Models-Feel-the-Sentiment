"""Isolated SST-2 Stage I; new disjoint data, frozen Qwen, train-only correction."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os, sys
sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
ROOT = WORKSPACE
OUT = ROOT / 'artifacts/stage_i_sst2'
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse, hashlib, json, sqlite3, unicodedata, time, gc, warnings, platform, importlib.util
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
KEYS = ('0.5b', '3b')
SPLITS = ('train', 'validation', 'test')
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
EMOTIONS = ('joy', 'love', 'excitement', 'gratitude', 'sadness', 'anger', 'fear', 'disgust')

def sha(p):
    p = source_path(p)
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def js(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    t.replace(p)

def arrsave(p, a):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    with t.open('wb') as f:
        np.save(f, a)
    t.replace(p)

def log(s):
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'execution.log').open('a', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + s + '\n')

def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def rows(split):
    return [json.loads(s) for s in (OUT / 'splits' / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]

def met(y, p, s):
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(accuracy=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average='macro')), balanced_accuracy=float(balanced_accuracy_score(y, p)), auroc=float(roc_auc_score(y, s)), negative_recall=float(tn / (tn + fp)), positive_recall=float(tp / (tp + fn)), confusion_matrix=cm.tolist(), n=len(y))

def prepare():
    import pyarrow as pa, pyarrow.ipc as ipc
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        verify_sources()
        return js(OUT / 'protocol.json')
    trainfile = ROOT / 'artifacts/datasets/sst2/train/data-00000-of-00001.arrow'
    valfile = ROOT / 'artifacts/datasets/sst2/validation/data-00000-of-00001.arrow'
    train = ipc.open_stream(pa.memory_map(str(trainfile), 'r')).read_all().to_pylist()
    val = ipc.open_stream(pa.memory_map(str(valfile), 'r')).read_all().to_pylist()
    assert len(train) == 67349 and len(val) == 872
    used = set()
    sources = []
    tracked = {}

    def record(p, table, ids):
        ids = {int(i) for i in ids}
        assert all((0 <= i < len(train) for i in ids))
        if ids:
            used.update(ids)
            sources.append(dict(path=str(p.relative_to(ROOT)), table=table, count=len(ids), indices=sorted(ids)))
            tracked[str(p.relative_to(ROOT))] = sha(p)

    def parsed(r):
        if not isinstance(r, dict):
            return None
        ident = str(r.get('id', ''))
        if ident.startswith('sst2:train:'):
            return int(ident.split(':')[2])
        if r.get('dataset') == 'sst2' and r.get('source_split') == 'train':
            return r.get('source_row_index', r.get('row_index'))
        return None
    paths = [p for p in ROOT.rglob('*') if p.is_file() and (not any((x in p.parts for x in ('.git', '.cache', '.venv', 'stage_i_sst2'))))]
    for p in paths:
        if p.suffix == '.sqlite3':
            assert not Path(str(p) + '-wal').exists() or Path(str(p) + '-wal').stat().st_size == 0, 'Uncheckpointed source WAL: ' + str(p)
            db = sqlite3.connect('file:' + str(p) + '?mode=ro&immutable=1', uri=True)
            try:
                for table, in db.execute("select name from sqlite_master where type='table'"):
                    qt = '"' + table.replace('"', '""') + '"'
                    cols = {r[1] for r in db.execute('pragma table_info(' + qt + ')')}
                    ids = set()
                    if {'dataset', 'source_split', 'source_row_index'} <= cols:
                        ids.update((r[0] for r in db.execute('select distinct source_row_index from ' + qt + " where dataset='sst2' and source_split='train'")))
                    if {'dataset', 'split', 'row_index'} <= cols:
                        ids.update((r[0] for r in db.execute('select distinct row_index from ' + qt + " where dataset='sst2' and split='train'")))
                    if 'id' in cols:
                        ids.update((int(r[0].split(':')[2]) for r in db.execute('select distinct id from ' + qt + " where id like 'sst2:train:%'")))
                    if p.name == 'sst2_train_12000_cuda.sqlite3' and 'row_index' in cols:
                        ids.update((r[0] for r in db.execute('select distinct row_index from ' + qt)))
                    record(p, table, ids)
            finally:
                db.close()
        elif p.suffix == '.jsonl':
            ids = {i for r in (json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()) if (i := parsed(r)) is not None}
            record(p, 'jsonl', ids)
        elif p.name.endswith('samples.json') or p.name == 'sst2_train_12000_sample.json':
            obj = js(p)
            seq = obj if isinstance(obj, list) else obj.get('samples', [])
            if isinstance(seq, dict):
                seq = [r for group in seq.values() if isinstance(group, list) for r in group]
            ids = {i for r in seq if (i := parsed(r)) is not None}
            if p.name == 'sst2_train_12000_sample.json':
                ids.update(obj['row_indices'])
            record(p, 'json', ids)
    oldnorm = {norm(train[i]['sentence']) for i in used} | {norm(r['sentence']) for r in val}
    oldexact = {train[i]['sentence'] for i in used} | {r['sentence'] for r in val}
    labelsets = {}
    for r in train:
        labelsets.setdefault(norm(r['sentence']), set()).add(r['label'])
    reasons = dict(previous_id=0, previous_normalized_text=0, conflicting_normalized_labels=0, remaining_duplicate=0)
    pool = {0: [], 1: []}
    seen = set()
    exact_excluded = 0
    for i, r in enumerate(train):
        t = norm(r['sentence'])
        if i in used:
            reasons['previous_id'] += 1
            continue
        if t in oldnorm:
            reasons['previous_normalized_text'] += 1
            exact_excluded += int(r['sentence'] in oldexact)
            continue
        if len(labelsets[t]) != 1:
            reasons['conflicting_normalized_labels'] += 1
            continue
        if t in seen:
            reasons['remaining_duplicate'] += 1
            continue
        seen.add(t)
        pool[r['label']].append(i)
    assert all((len(pool[c]) >= 10593 for c in (0, 1))), str({c: len(pool[c]) for c in pool})
    rng = np.random.default_rng(42)
    selected = {s: [] for s in SPLITS}
    for c in (0, 1):
        order = rng.permutation(pool[c])
        offset = 0
        for s, n in zip(SPLITS, (7093, 1000, 2500)):
            selected[s].extend((int(i) for i in order[offset:offset + n]))
            offset += n
    allids = set()
    alltexts = set()
    splitinfo = {}
    for s in SPLITS:
        order = np.array(selected[s])[rng.permutation(len(selected[s]))]
        data = []
        for i in order:
            i = int(i)
            r = train[i]
            t = norm(r['sentence'])
            assert i not in used and i not in allids and (t not in oldnorm) and (t not in alltexts)
            allids.add(i)
            alltexts.add(t)
            data.append(dict(id=f'sst2:train:{i}', source_split='train', source_row_index=i, label=r['label'], text=r['sentence'], text_sha256=digest(r['sentence']), normalized_text_sha256=digest(t)))
        p = OUT / 'splits' / (s + '.jsonl')
        p.parent.mkdir(exist_ok=True)
        p.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in data)), encoding='utf-8')
        pd.DataFrame([{k: v for k, v in r.items() if k != 'text'} for r in data]).to_csv(OUT / 'splits' / (s + '_ids.csv'), index=False)
        splitinfo[s] = dict(n=len(data), label_counts={str(c): sum((r['label'] == c for r in data)) for c in (0, 1)}, sha256=sha(p))
    ep = js(ROOT / 'artifacts/stage_e/protocol.json')
    ib = js(ROOT / 'artifacts/stage_i_emotion_guided_correction/protocol.json')
    models = ib['models']
    for key, cfg in models.items():
        assert cfg['layer'] == {'0.5b': 14, '3b': 18}[key]
        for name, fn, expected in [('raw', 'emotion_directions_raw.float32.npy', cfg['raw_sha256']), ('final', 'emotion_directions_final.float32.npy', cfg['final_sha256'])]:
            p = ROOT / 'artifacts' / ('stage_b_raw_vectors' if name == 'raw' else 'stage_b_final_vectors') / key / fn
            assert sha(p) == expected and np.load(p, mmap_mode='r').shape == (8, cfg['total_layers'], cfg['width'])
            tracked[str(p.relative_to(ROOT))] = sha(p)
        assert ep['models'][key]['prompts']['sst2'] == ep['models']['0.5b']['prompts']['sst2']
    for p in paths:
        if p.is_relative_to(ROOT / 'artifacts/stage_i_sentiment_correction') or p.is_relative_to(ROOT / 'artifacts/stage_i_emotion_guided_correction'):
            tracked[str(p.relative_to(ROOT))] = sha(p)
    for p in [trainfile, valfile, ROOT / 'artifacts/stage_e/protocol.json', ROOT / 'stage_h.py', ROOT / 'stage_d.py', ROOT / 'stage_e_pipeline.py', ROOT / 'artifacts/stage_c_empathetic_dialogues/selected_intervention_layers.json']:
        tracked[str(p.relative_to(ROOT))] = sha(p)
    protocol = dict(models=models, prompt=ep['models']['0.5b']['prompts']['sst2'], candidate_ids={'positive': 30487, 'negative': 42224}, scoring='float16 CUDA frozen forward; last attended prompt token full-vocabulary float32 log_softmax; float32 positive-minus-negative subtraction; margin>0 positive', pooling='post-block layer-1 hook; float32 mean over raw-text overlapping tokens; exclude prompt/special/pad; float64 projections onto unit vectors; no example normalization', C_grid=GRID, selection='validation balanced accuracy, macro-F1, then smallest C; train-only fit; no refit', classifier='StandardScaler train-only; L2 lbfgs intercept class_weight=None tol1e-5 max_iter3000 random_state42', split_info=splitinfo, seed=42, random8_seeds=list(range(10)), random_rank_seeds=list(range(2000, 2050)), rank_relative_tolerance=1e-05, pca='training-only randomized PCA, seed42, oversamples16 iterated_power4 no whitening; training centering', bootstrap=dict(draws=2000, seed=47, paired=True, class_stratified=True, primary=['auroc', 'balanced_accuracy'], secondary=['accuracy', 'macro_f1'], intervals='percentile conditional on frozen classifiers/bases; nominal no multiplicity correction; seed SD separate'), test_status='new internal held-out official TRAIN subset, not official test or independent external dataset', checkpoint_history_limitation='Stage B saved model ID/shape but no original commit; cannot establish historical checkpoint identity', no_quantization=True, no_finetuning=True)
    save(OUT / 'data_usage_audit.json', dict(official_train=67349, official_validation_excluded=872, previous_train_union_count=len(used), previous_train_indices=sorted(used), sources=sources, exclusions=reasons, additional_exact_text_excluded=exact_excluded, clean_available={str(c): len(pool[c]) for c in pool}, new_split_counts=splitinfo, reused_samples=0, text_normalization='NFKC casefold whitespace collapse; reject conflicting labels; earliest unique remaining index', split_procedure='numpy default_rng42: permute each class ascending official IDs, allocate train/val/test, permute each merged split', limitations='SST-2 contains phrase fragments; no review-level original IDs available, so normalized text exclusion cannot guarantee review-level independence'))
    save(OUT / 'repository_preflight.json', dict(source_sha256=tracked, completed_before_extraction=True, old_sst2_caches_reusable=False, reason='No new example IDs are in prior examined caches; all prior IDs excluded', stage_i_c='No existing implementation located; implement rank controls here', provenance_limitation=protocol['checkpoint_history_limitation']))
    save(OUT / 'protocol.json', protocol)
    log('Preflight complete: ' + json.dumps(dict(previous_union=len(used), exclusions=reasons, available={str(c): len(pool[c]) for c in pool}, splits={s: v['n'] for s, v in splitinfo.items()}, reused=0)))
    return protocol

def verify_sources():
    for rel, h in js(OUT / 'repository_preflight.json')['source_sha256'].items():
        assert sha(ROOT / rel) == h, 'Previous source changed: ' + rel
    p = js(OUT / 'protocol.json')
    a = js(OUT / 'data_usage_audit.json')
    used = set(a['previous_train_indices'])
    seen = set()
    texts = set()
    for s in SPLITS:
        assert sha(OUT / 'splits' / (s + '.jsonl')) == p['split_info'][s]['sha256']
        for r in rows(s):
            assert r['source_row_index'] not in used and r['id'] not in seen and (r['normalized_text_sha256'] not in texts)
            seen.add(r['id'])
            texts.add(r['normalized_text_sha256'])
            assert r['text_sha256'] == digest(r['text'])
    return p

def extract(protocol, smoke=False, batch_size=8):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    assert torch.cuda.is_available(), 'CUDA required'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('stage_d_reference', source_path(ROOT / 'stage_d.py'))
    ref = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CODE_ROOT))
    spec.loader.exec_module(ref)
    for key in KEYS:
        cfg = protocol['models'][key]
        folder = OUT / key
        folder.mkdir(exist_ok=True)
        allrows = [r for s in SPLITS for r in rows(s)]
        n = len(allrows)
        width = cfg['width']
        hp = folder / 'pooled.float32.npy'
        new = not hp.exists()
        h = np.lib.format.open_memmap(hp, mode='w+' if new else 'r+', dtype=np.float32, shape=(n, width))
        db = sqlite3.connect(folder / 'scores.sqlite3')
        db.execute('pragma synchronous=FULL')
        db.execute('create table if not exists scores(row_index integer primary key,id text unique,logp_positive real,logp_negative real,margin real,text_tokens integer,hidden_sha256 text)')
        completed = {r[0]: r[1:] for r in db.execute('select * from scores')}
        for i, v in completed.items():
            assert v[0] == allrows[i]['id'] and np.isfinite(h[i]).all() and (hashlib.sha256(h[i].tobytes()).hexdigest() == v[-1]) and (float(np.float32(v[1]) - np.float32(v[2])) == v[3])
        pending = [i for i in range(n) if i not in completed]
        pending = pending[:8] if smoke else pending
        log(f'Extraction {key}: {len(completed)}/{n} saved; {len(pending)} pending; CUDA {torch.cuda.get_device_name(0)}; FP16, no quantization')
        if pending:
            tok = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['checkpoint_commit'], use_fast=True, local_files_only=True)
            assert digest(tok.backend_tokenizer.to_str()) == cfg['tokenizer_sha256']
            tok.padding_side = 'left'
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            ids = {name: tok.encode(name, add_special_tokens=False) for name in ('positive', 'negative')}
            assert ids == {name: [i] for name, i in protocol['candidate_ids'].items()}
            snapshot = ROOT / '.cache/huggingface/hub' / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots' / cfg['checkpoint_commit']
            parsed_config = AutoConfig.from_pretrained(cfg['model_id'], revision=cfg['checkpoint_commit'], local_files_only=True)
            assert digest(json.dumps(parsed_config.to_dict(), sort_keys=True, ensure_ascii=False)) == cfg['config_sha256']
            assert parsed_config._commit_hash == cfg['checkpoint_commit']
            model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], revision=cfg['checkpoint_commit'], dtype=torch.float16, local_files_only=True, attn_implementation='sdpa').eval().to('cuda')
            model.requires_grad_(False)
            assert len(model.model.layers) == cfg['total_layers'] and model.config.hidden_size == width and (next(model.parameters()).device.type == 'cuda') and (next(model.parameters()).dtype == torch.float16)
            save(folder / 'model_identity.json', dict(**cfg, candidate_ids=protocol['candidate_ids'], precision='float16', device=torch.cuda.get_device_name(0), transformers=__import__('transformers').__version__, torch=torch.__version__, weights={p.name: sha(p) for p in snapshot.glob('*.safetensors')}, historical_stage_b_commit_verified=False))
            cache = {}
            mask = None

            def capture(module, args, output):
                residual = output[0] if isinstance(output, tuple) else output
                assert residual.shape[-1] == width
                cache['pooled'] = ((residual.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)).cpu().numpy()
                if cache.get('verify', False):
                    cache['residual'] = residual.detach().clone()
            hook = model.model.layers[cfg['layer'] - 1].register_forward_hook(capture)
            start = time.monotonic()
            j = 0
            actual = batch_size
            lastlog = start
            try:
                while j < len(pending):
                    batchindices = pending[j:j + actual]
                    batchrows = [allrows[i] for i in batchindices]
                    inputs, mask, counts, truncated = ref.token_batch(tok, key, protocol['prompt'], batchrows)
                    assert not any(truncated), 'Truncation refused'
                    try:
                        cache['verify'] = not (folder / 'smoke_verification.json').exists()
                        with torch.inference_mode():
                            output = model(**inputs, use_cache=False, output_hidden_states=cache['verify'])
                            positions = torch.arange(inputs['input_ids'].shape[1], device='cuda').expand_as(inputs['input_ids'])
                            last = positions.masked_fill(~inputs['attention_mask'].bool(), -1).max(1).values
                            logits = output.logits[torch.arange(len(batchrows), device='cuda'), last].float()
                            lp = torch.log_softmax(logits, -1)[:, [30487, 42224]].cpu().numpy()
                            m = lp[:, 0] - lp[:, 1]
                            pooled = cache['pooled']
                            if cache['verify']:
                                assert torch.equal(cache['residual'], output.hidden_states[cfg['layer']])
                                manual = ((output.hidden_states[cfg['layer']].float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)).cpu().numpy()
                                assert np.array_equal(manual, pooled)
                                hook.remove()
                                plain = model(**inputs, use_cache=False)
                                check = torch.log_softmax(plain.logits[torch.arange(len(batchrows), device='cuda'), last].float(), -1)[:, [30487, 42224]].cpu().numpy()
                                assert np.array_equal(lp, check), 'Hook changed model outputs'
                                hook = model.model.layers[cfg['layer'] - 1].register_forward_hook(capture)
                                save(folder / 'smoke_verification.json', dict(n=len(batchrows), training_only=True, hook_equals_hidden_states=True, hook_does_not_change_logits=True, layer_1_based=cfg['layer'], block_index_0_based=cfg['layer'] - 1, candidate_ids=protocol['candidate_ids'], mask_counts=counts, text_pooling_source='stage_d.token_batch reused exactly', float32_margin_subtraction=True, no_truncation=True))
                                del plain, manual
                                cache.pop('residual', None)
                        assert pooled.shape == (len(batchrows), width) and np.isfinite(pooled).all() and np.isfinite(lp).all()
                        h[batchindices] = pooled
                        h.flush()
                        with db:
                            db.executemany('insert into scores values(?,?,?,?,?,?,?)', [(int(i), allrows[i]['id'], float(lp[k, 0]), float(lp[k, 1]), float(m[k]), int(counts[k]), hashlib.sha256(pooled[k].tobytes()).hexdigest()) for k, i in enumerate(batchindices)])
                        j += len(batchindices)
                        now = time.monotonic()
                        del output, logits, inputs
                        if now - lastlog >= 15 or j == len(pending):
                            rate = j / (now - start)
                            free, total = torch.cuda.mem_get_info()
                            log(f'  {key}: {len(completed) + j}/{n}; {rate:.1f} examples/s; ETA {(len(pending) - j) / rate / 60:.1f} min; batch {actual}; GPU allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB, free {free / 2 ** 30:.2f} GiB')
                            lastlog = now
                            with (OUT / 'gpu_monitor.jsonl').open('a', encoding='utf-8') as f:
                                f.write(json.dumps(dict(model=key, completed=len(completed) + j, n=n, batch=actual, allocated=torch.cuda.memory_allocated(), peak=torch.cuda.max_memory_allocated(), free=free, total=total)) + '\n')
                    except torch.cuda.OutOfMemoryError:
                        if actual == 1:
                            raise
                        actual = max(1, actual // 2)
                        cache.pop('residual', None)
                        cache.pop('pooled', None)
                        gc.collect()
                        torch.cuda.empty_cache()
                        log(f'CUDA OOM: retry unsaved batch with batch_size={actual}')
            finally:
                hook.remove()
            del model, tok, cache
            gc.collect()
            torch.cuda.empty_cache()
        count = db.execute('select count(*) from scores').fetchone()[0]
        db.close()
        h.flush()
        del h
        save(folder / 'extraction_state.json', dict(complete=count == n, completed=count, total=n, protocol_sha256=sha(OUT / 'protocol.json'), hidden_shape=[n, width]))
        if not smoke:
            assert count == n
    save(OUT / 'state.json', dict(complete=False, phase='smoke_complete' if smoke else 'extraction_complete'))

def basis(v, tol):
    _, s, vh = np.linalg.svd(v, full_matrices=False)
    rank = int(np.sum(s > s[0] * tol))
    b = vh[:rank]
    assert np.allclose(b @ b.T, np.eye(rank), atol=1e-12)
    return (b, s)

def features(protocol):
    offset = {}
    j = 0
    for s in SPLITS:
        offset[s] = slice(j, j + protocol['split_info'][s]['n'])
        j += protocol['split_info'][s]['n']
    info = {}
    for key in KEYS:
        folder = OUT / key
        if (folder / 'features_metadata.json').exists():
            meta = js(folder / 'features_metadata.json')
            assert meta['protocol_sha256'] == sha(OUT / 'protocol.json')
            for rel, d in meta['sha256'].items():
                assert sha(folder / rel) == d
            info[key] = meta
            continue
        cfg = protocol['models'][key]
        state = js(folder / 'extraction_state.json')
        assert state['complete']
        h = np.load(folder / 'pooled.float32.npy', mmap_mode='r')
        db = sqlite3.connect('file:' + str(folder / 'scores.sqlite3') + '?mode=ro', uri=True)
        rr = db.execute('select row_index,id,logp_positive,logp_negative,margin from scores order by row_index').fetchall()
        db.close()
        allrows = [r for s in SPLITS for r in rows(s)]
        assert len(rr) == len(h)
        assert all((r[:2] == (i, allrows[i]['id']) and float(np.float32(r[2]) - np.float32(r[3])) == r[4] for i, r in enumerate(rr)))
        m = np.array([r[4] for r in rr])
        pd.DataFrame([dict(id=r[1], logp_positive=r[2], logp_negative=r[3], margin=r[4]) for r in rr]).to_csv(folder / 'sentiment_scores.csv', index=False)
        hb = np.asarray(h, dtype=np.float64)
        arrays = {'sentiment_only': m[:, None]}
        ranks = {}
        basis_info = {}
        bdir = folder / 'bases'
        bdir.mkdir(exist_ok=True)
        for name, stage, fn in [('raw', 'stage_b_raw_vectors', 'emotion_directions_raw.float32.npy'), ('denoised', 'stage_b_final_vectors', 'emotion_directions_final.float32.npy')]:
            v = np.asarray(np.load(ROOT / 'artifacts' / stage / key / fn)[:, cfg['layer'] - 1], dtype=np.float64)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            b, sing = basis(v, protocol['rank_relative_tolerance'])
            ranks[name] = len(b)
            basis_info[name] = dict(rank=len(b), singular_values=sing.tolist(), relative_tolerance=protocol['rank_relative_tolerance'])
            arrsave(bdir / (name + '_unit_vectors.npy'), v)
            arrsave(bdir / (name + '_orthonormal_basis.npy'), b)
            arrays[name] = np.column_stack([m, hb @ v.T])
            arrays['orthonormal_' + name] = np.column_stack([m, hb @ b.T])

        def rand(width, k, seed):
            q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(width, k)), mode='reduced')
            q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
            return q.T
        for seed in protocol['random8_seeds']:
            v = rand(cfg['width'], 8, seed)
            name = f'random8_seed_{seed}'
            arrsave(bdir / (name + '.npy'), v)
            arrays[name] = np.column_stack([m, hb @ v.T])
        for rank in sorted(set(ranks.values())):
            for seed in protocol['random_rank_seeds']:
                name = f'random_rank{rank}_seed_{seed}'
                v = rand(cfg['width'], rank, seed)
                arrsave(bdir / (name + '.npy'), v)
                arrays[name] = np.column_stack([m, hb @ v.T])
            pca = PCA(n_components=rank, svd_solver='randomized', random_state=42, n_oversamples=16, iterated_power=4, whiten=False).fit(hb[offset['train']])
            joblib.dump(pca, bdir / f'pca_rank{rank}.joblib')
            arrays[f'pca_rank{rank}'] = np.column_stack([m, pca.transform(hb)])
            save(bdir / f'pca_rank{rank}.json', dict(fit_split='train', n_training=14186, rank=rank, explained_variance_ratio=pca.explained_variance_ratio_.tolist(), mean_sha256=hashlib.sha256(pca.mean_.tobytes()).hexdigest()))
        fdir = folder / 'features'
        fdir.mkdir(exist_ok=True)
        for name, a in arrays.items():
            assert np.isfinite(a).all()
            arrsave(fdir / (name + '.npy'), a)
        pd.DataFrame([{k: v for k, v in r.items() if k != 'text'} for r in allrows]).to_csv(folder / 'feature_ids.csv', index=False)
        meta = dict(protocol_sha256=sha(OUT / 'protocol.json'), ranks=ranks, rank_audit=basis_info, offset={s: [v.start, v.stop] for s, v in offset.items()}, methods=list(arrays), feature_order='margin then named emotion scores or basis coordinates', sha256={str(p.relative_to(folder)): sha(p) for p in [*bdir.rglob('*'), *fdir.rglob('*'), folder / 'feature_ids.csv', folder / 'sentiment_scores.csv'] if p.is_file()})
        save(folder / 'features_metadata.json', meta)
        info[key] = meta
        log(f'Features {key}: ranks={ranks}; {len(arrays)} classifier methods; 10 random8 and 50 rank-matched seeds; PCA fit train only')
    return info

def fit(protocol, info):
    selected = {}
    frozen = {}
    for key in KEYS:
        folder = OUT / key
        selected[key] = {}
        y = np.array([r['label'] for r in rows('train')])
        yv = np.array([r['label'] for r in rows('validation')])
        meta = info[key]
        for name in meta['methods']:
            dest = folder / 'classifiers' / name
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'metadata.json').exists():
                md = js(dest / 'metadata.json')
                assert sha(dest / 'model.joblib') == md['model_sha256']
                selected[key][name] = md
                continue
            a = np.load(folder / 'features' / (name + '.npy'), mmap_mode='r')
            x = a[slice(*meta['offset']['train'])]
            xv = a[slice(*meta['offset']['validation'])]
            search = []
            candidates = []
            for c in GRID:
                p = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(C=c, solver='lbfgs', fit_intercept=True, class_weight=None, tol=1e-05, max_iter=3000, random_state=42, l1_ratio=0.0))])
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    p.fit(x, y)
                v = p.decision_function(xv)
                mm = met(yv, (v > 0).astype(int), v)
                search.append(dict(C=c, validation_balanced_accuracy=mm['balanced_accuracy'], validation_macro_f1=mm['macro_f1'], validation_auroc=mm['auroc'], n_iter=int(p.named_steps['classifier'].n_iter_[0])))
                candidates.append(p)
            best = max(range(len(GRID)), key=lambda i: (search[i]['validation_balanced_accuracy'], search[i]['validation_macro_f1'], -GRID[i]))
            p = candidates[best]
            assert int(p.named_steps['scaler'].n_samples_seen_) == 14186 and np.allclose(p.named_steps['scaler'].mean_, x.mean(0), atol=1e-10)
            joblib.dump(p, dest / 'model.joblib')
            md = dict(**search[best], model_sha256=sha(dest / 'model.joblib'), feature_sha256=sha(folder / 'features' / (name + '.npy')), coefficients=p.named_steps['classifier'].coef_.tolist(), intercept=p.named_steps['classifier'].intercept_.tolist(), scaler_mean=p.named_steps['scaler'].mean_.tolist(), scaler_scale=p.named_steps['scaler'].scale_.tolist(), scaler_fit_n=14186, fit_split='train', threshold_decision_score=0.0, refit_validation=False)
            save(dest / 'metadata.json', md)
            pd.DataFrame(search).to_csv(dest / 'C_search.csv', index=False)
            selected[key][name] = md
            log(f'Fit {key}/{name}: C={md['C']}; validation balanced accuracy={md['validation_balanced_accuracy']:.5f}')
        frozen[key] = {name: md['model_sha256'] for name, md in selected[key].items()}
    save(OUT / 'frozen_before_test.json', dict(protocol_sha256=sha(OUT / 'protocol.json'), classifiers=frozen, selected=selected, features_metadata_sha256={k: sha(OUT / k / 'features_metadata.json') for k in KEYS}, script_sha256=sha(Path(__file__)), all_classifiers_frozen_before_any_test_metrics=True))
    save(OUT / 'state.json', dict(complete=False, phase='all_classifiers_frozen'))
    return selected
BOOTMET = ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc')

def weighted(y, p, weight, groups):
    tn = float(weight[(y == 0) & (p == 0)].sum())
    fp = float(weight[(y == 0) & (p == 1)].sum())
    fn = float(weight[(y == 1) & (p == 0)].sum())
    tp = float(weight[(y == 1) & (p == 1)].sum())
    n0 = tn + fp
    n1 = tp + fn
    u, inv = groups
    neg = np.bincount(inv[y == 0], weights=weight[y == 0], minlength=len(u))
    pos = np.bincount(inv[y == 1], weights=weight[y == 1], minlength=len(u))
    auc = float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (n0 * n1))
    return np.array([(tn + tp) / (n0 + n1), 0.5 * (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn)), 0.5 * (tn / n0 + tp / n1), auc])

def bootstrap(key, y, pred, score, point, ranks):
    dest = OUT / key / 'bootstrap'
    dest.mkdir(exist_ok=True)
    if (dest / 'paired_differences.csv').exists():
        return pd.read_csv(dest / 'paired_differences.csv').to_dict('records')
    names = list(score)
    groups = {name: np.unique(score[name], return_inverse=True) for name in names}
    rng = np.random.default_rng(47)
    cls = [np.flatnonzero(y == c) for c in (0, 1)]
    draws = {name: np.empty((2000, 4)) for name in names}
    for b in range(2000):
        idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in cls])
        w = np.bincount(idx, minlength=len(y)).astype(float)
        for name in names:
            draws[name][b] = weighted(y, pred[name], w, groups[name])
        if b < 3:
            for name in ('original', 'raw', 'denoised', 'sentiment_only'):
                truth = met(y[idx], pred[name][idx], score[name][idx])
                assert np.allclose(draws[name][b], [truth[m] for m in BOOTMET], atol=1e-12)
        if (b + 1) % 250 == 0:
            log(f'Bootstrap {key}: {b + 1}/2000 paired stratified draws')
    pts = {name: np.array([point[name][m] for m in BOOTMET]) for name in names}
    ensembles = {'random8_mean': [name for name in names if name.startswith('random8_seed_')]}
    for rank in sorted(set(ranks.values())):
        ensembles[f'random_rank{rank}_mean'] = [name for name in names if name.startswith(f'random_rank{rank}_seed_')]
    for group, nn in ensembles.items():
        draws[group] = np.mean([draws[name] for name in nn], axis=0)
        pts[group] = np.mean([pts[name] for name in nn], axis=0)
    comparisons = []
    for name in names:
        for ref in ('original', 'sentiment_only'):
            if ref != name:
                comparisons.append((name, ref))
    for name in ('raw', 'denoised', 'orthonormal_raw', 'orthonormal_denoised'):
        rank = ranks['denoised' if 'denoised' in name else 'raw']
        comparisons.extend(((name, ref) for ref in ('random8_mean', f'random_rank{rank}_mean', f'pca_rank{rank}')))
    comparisons.append(('denoised', 'raw'))
    results = []
    absolute = []
    for name, a in draws.items():
        for j, m in enumerate(BOOTMET):
            lo, hi = np.quantile(a[:, j], [0.025, 0.975])
            absolute.append(dict(model=key, method=name, metric=m, value=float(pts[name][j]), ci_low=float(lo), ci_high=float(hi), draws=2000, seed=47))
    for name, ref in comparisons:
        delta = draws[name] - draws[ref]
        for j, m in enumerate(BOOTMET):
            lo, hi = np.quantile(delta[:, j], [0.025, 0.975])
            results.append(dict(model=key, method=name, reference=ref, metric=m, difference=float(pts[name][j] - pts[ref][j]), ci_low=float(lo), ci_high=float(hi), draws=2000, seed=47, uncertainty='paired examples conditional on frozen models and fixed seeds; seed SD separate'))
    pd.DataFrame(results).to_csv(dest / 'paired_differences.csv', index=False)
    pd.DataFrame(absolute).to_csv(dest / 'absolute_metrics_95ci.csv', index=False)
    np.savez_compressed(dest / 'draws.npz', **draws)
    save(dest / 'verification.json', dict(weighted_auc_tie_handling_matches_sklearn=True, validated_draws=3, paired=True, class_stratified=True, n_draws=2000, seed=47))
    return results

def evaluate(protocol, info, selected):
    freeze = js(OUT / 'frozen_before_test.json')
    assert freeze['protocol_sha256'] == sha(OUT / 'protocol.json')
    for key in KEYS:
        for name, h in freeze['classifiers'][key].items():
            assert sha(OUT / key / 'classifiers' / name / 'model.joblib') == h
    metrics = []
    changes = []
    confusions = {}
    summaries = []
    ci = []
    results = {}
    for key in KEYS:
        folder = OUT / key
        meta = info[key]
        models = {name: joblib.load(folder / 'classifiers' / name / 'model.joblib') for name in meta['methods']}
        results[key] = {}
        for split in SPLITS:
            rr = rows(split)
            y = np.array([r['label'] for r in rr])
            sl = slice(*meta['offset'][split])
            m = np.load(folder / 'features/sentiment_only.npy', mmap_mode='r')[sl, 0]
            scores = {'original': m}
            pred = {'original': (m > 0).astype(int)}
            for name, p in models.items():
                x = np.load(folder / 'features' / (name + '.npy'), mmap_mode='r')[sl]
                scores[name] = p.decision_function(x)
                pred[name] = p.predict(x)
                assert np.array_equal(pred[name], scores[name] > 0)
            results[key][split] = {name: met(y, pred[name], s) for name, s in scores.items()}
            frame = pd.DataFrame(dict(id=[r['id'] for r in rr], label=y, source_row_index=[r['source_row_index'] for r in rr], margin=m))
            for name, res in results[key][split].items():
                metrics.append(dict(model=key, split=split, method=name, selected_C=selected[key][name]['C'] if name != 'original' else None, **{k: v for k, v in res.items() if k != 'confusion_matrix'}))
                confusions[f'{key}/{split}/{name}'] = res['confusion_matrix']
                frame[name + '_prediction'] = pred[name]
                frame[name + '_decision_score'] = scores[name]
                for ref in ('original', 'sentiment_only'):
                    fixed = (pred[ref] != y) & (pred[name] == y)
                    broken = (pred[ref] == y) & (pred[name] != y)
                    changes.append(dict(model=key, split=split, method=name, reference=ref, incorrect_corrected=int(fixed.sum()), correct_made_incorrect=int(broken.sum())))
            frame.to_csv(folder / (split + '_predictions.csv'), index=False)
            groups = {'random8': [name for name in scores if name.startswith('random8_seed_')]}
            for rank in sorted(set(meta['ranks'].values())):
                groups[f'random_rank{rank}'] = [name for name in scores if name.startswith(f'random_rank{rank}_seed_')]
            for group, names in groups.items():
                for mm in ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc', 'negative_recall', 'positive_recall'):
                    vals = np.array([results[key][split][name][mm] for name in names])
                    summaries.append(dict(model=key, split=split, method=group, metric=mm, mean=float(vals.mean()), sd=float(vals.std(ddof=1)), seeds=len(names), minimum=float(vals.min()), maximum=float(vals.max())))
            if split == 'test':
                ci.extend(bootstrap(key, y, pred, scores, results[key][split], meta['ranks']))
        save(folder / 'all_split_metrics.json', results[key])
    pd.DataFrame(metrics).to_csv(OUT / 'metrics_all_methods.csv', index=False)
    pd.DataFrame(changes).to_csv(OUT / 'prediction_changes.csv', index=False)
    pd.DataFrame(ci).to_csv(OUT / 'paired_differences_95ci.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'random_control_mean_sd.csv', index=False)
    save(OUT / 'confusion_matrices.json', confusions)
    return (results, metrics, changes, summaries, ci)

def report(protocol, info, selected, results, metrics, changes, summaries, ci):
    audit = js(OUT / 'data_usage_audit.json')
    lines = ['Stage I: SST-2 Sentiment Correction', '', 'New internal held-out subset of official SST-2 training data; not official test or an independent external dataset.', 'Train 14,186 (7,093/class); validation 2,000 (1,000/class); test 5,000 (2,500/class).', f'Previously examined official train IDs excluded: {audit['previous_train_union_count']}; official validation excluded:872; additional exclusions:{audit['exclusions']}.', f'Clean available counts:{audit['clean_available']}; samples reused from earlier experiments:0.', 'Frozen SST-2 T1 prompt, candidate-token margin readout; FP16 CUDA without quantization. Layers14/24 and18/36.', 'Same-pass selected post-block text-token means; raw and denoised unit-direction projections, without per-example normalization.', 'Classifier fits/scalers/PCA trained on train only; C selected by validation balanced accuracy, macro-F1 then smallest C; no refit. Both models and all methods frozen before any test evaluation.', '', '| Method | 0.5B Accuracy | 0.5B AUROC | 3B Accuracy | 3B AUROC |', '|---|---:|---:|---:|---:|']

    def stat(key, group, metric):
        return next((r for r in summaries if r['model'] == key and r['split'] == 'test' and (r['method'] == group) and (r['metric'] == metric)))
    methods = [('original', 'Original Qwen'), ('sentiment_only', 'Balanced sentiment-only'), ('raw', 'Raw emotion-guided'), ('denoised', 'Denoised emotion-guided'), ('random8', 'Random features (10 seeds)'), ('random_rank', 'Rank-matched random (50 seeds)'), ('pca', 'PCA rank-matched'), ('orthonormal_raw', 'Orthonormal raw emotion'), ('orthonormal_denoised', 'Orthonormal denoised emotion')]
    table = []
    for name, label in methods:
        values = []
        row = {'method': label}
        for key in KEYS:
            rank = info[key]['ranks']['raw']
            group = f'random_rank{rank}' if name == 'random_rank' else name
            for mm in ('accuracy', 'auroc'):
                if name in ('random8', 'random_rank'):
                    s = stat(key, group, mm)
                    v = f'{s['mean']:.2%} +/- {s['sd']:.2%}' if mm == 'accuracy' else f'{s['mean']:.4f} +/- {s['sd']:.4f}'
                    row[key + '_' + mm] = s['mean']
                    row[key + '_' + mm + '_seed_sd'] = s['sd']
                else:
                    actual = f'pca_rank{rank}' if name == 'pca' else name
                    r = results[key]['test'][actual]
                    v = f'{r[mm]:.2%}' if mm == 'accuracy' else f'{r[mm]:.4f}'
                    row[key + '_' + mm] = r[mm]
                values.append(v)
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
        table.append(row)
    pd.DataFrame(table).to_csv(OUT / 'summary_table.csv', index=False)
    comparison = []
    tweet = pd.read_csv(ROOT / 'artifacts/stage_i_emotion_guided_correction/metrics_all_methods.csv')
    for key in KEYS:
        lines.extend(['', f'Model {key}: measured raw/denoised ranks {info[key]['ranks']}.', f'Sentiment-only selected C={selected[key]['sentiment_only']['C']}.'])
        for name in ('raw', 'denoised', 'orthonormal_raw', 'orthonormal_denoised'):
            mm = results[key]['test'][name]
            ch = next((r for r in changes if r['model'] == key and r['split'] == 'test' and (r['method'] == name) and (r['reference'] == 'sentiment_only')))
            lines.append(f'{name}: C={selected[key][name]['C']}; accuracy={mm['accuracy']:.4%}; macro-F1={mm['macro_f1']:.6f}; balanced accuracy={mm['balanced_accuracy']:.6f}; AUROC={mm['auroc']:.6f}; negative/positive recall={mm['negative_recall']:.6f}/{mm['positive_recall']:.6f}; confusion {mm['confusion_matrix']}; fixed {ch['incorrect_corrected']}, broke {ch['correct_made_incorrect']}.')
            rank = info[key]['ranks']['denoised' if 'denoised' in name else 'raw']
            for ref in ('sentiment_only', 'random8_mean', f'random_rank{rank}_mean', f'pca_rank{rank}'):
                for rr in ci:
                    if rr['model'] == key and rr['method'] == name and (rr['reference'] == ref):
                        lines.append(f'  vs {ref} {rr['metric']}: difference {rr['difference']:+.6f}; paired95% CI [{rr['ci_low']:+.6f},{rr['ci_high']:+.6f}].')
            if name in ('raw', 'denoised'):
                base = results[key]['test']['sentiment_only']
                random_auc = stat(key, f'random_rank{rank}', 'auroc')
                pc = results[key]['test'][f'pca_rank{rank}']
                lines.append(f'  Generic controls: emotion AUROC {mm['auroc']:.6f}, rank-random mean {random_auc['mean']:.6f}, PCA {pc['auroc']:.6f}. Emotion specificity is not established solely by beating the margin baseline; assess paired control comparisons above.')
                lines.append(f'  SST-2 emotion-minus-sentiment accuracy {(mm['accuracy'] - base['accuracy']) * 100:+.3f} percentage points; AUROC {mm['auroc'] - base['auroc']:+.6f}.')
        for name in ('original', 'sentiment_only', 'raw', 'denoised'):
            t = tweet[(tweet.model == key) & (tweet.split == 'test') & (tweet.method == name)].iloc[0]
            s = results[key]['test'][name]
            comparison.append(dict(model=key, method=name, sst2_accuracy=s['accuracy'], sst2_auroc=s['auroc'], tweet_accuracy=float(t.accuracy), tweet_auroc=float(t.auroc)))
            lines.append(f'Dataset comparison {name}: SST-2 accuracy/AUROC {s['accuracy']:.4%}/{s['auroc']:.6f}; saved TweetEval {t.accuracy:.4%}/{t.auroc:.6f}.')
    pd.DataFrame(comparison).to_csv(OUT / 'sst2_vs_tweeteval.csv', index=False)
    lines += ['', 'Selected C values (all fixed seeds, no winner selection):', pd.DataFrame([dict(model=k, method=n, C=v['C']) for k in KEYS for n, v in selected[k].items()]).to_string(index=False), '', 'Interpretation and limitations:', 'This experiment measures supervised predictive information beyond a frozen next-token sentiment margin. It is not a causal intervention and is not Stage E margin-free probing or Stage A generated-label accuracy. Read the actual differences and intervals, including null or negative findings, rather than assuming correction improves.', 'SST-2 uses new balanced splits from official training, with unseen row IDs and normalized texts; TweetEval uses balanced training but unbalanced official validation/test (1,131/6,347), whose test examples were already examined. SST-2 T1 is binary; the frozen TweetEval prompt retains three-label wording, with binary readout. The datasets differ in text domain, class balance, prompt and test status, so numerical differences are not a controlled dataset effect.', 'SST-2 training examples include sentence fragments with shared review origins. Exact and normalized duplicates are removed, but unavailable original-review grouping means review-level independence cannot be guaranteed. No external test-set independence is claimed.', 'Stage B metadata lacks historical checkpoint commits; vector hashes, model IDs, shapes and downstream pinned commits are verified, but historical checkpoint equivalence cannot be established.', 'Bootstrap: 2,000 paired class-stratified test-example draws, seed47; percentile nominal95% CIs conditional on frozen classifiers and fixed directions. Seed SD is separate variability, not an example CI or hierarchical uncertainty. No multiplicity correction; findings remain exploratory across many comparisons.', 'Raw and denoised eight coordinates are correlated rank-seven spans; orthonormal versions can differ through scaler/L2 geometry. PCA is fitted on training only. Random8 has eight dimensions, rank-random matches the measured ranks; all seeds reported, none selected using test.']
    text = '\n'.join(lines) + '\n'
    (OUT / 'results_summary.txt').write_text(text, encoding='utf-8')
    (OUT / 'RESULTS_REPORT.md').write_text(text, encoding='utf-8')
    log('\n' + text)
    (OUT / 'README.md').write_text('Stage I SST-2 correction\n\nAll outputs isolated here; earlier experiments remain read-only.\n\n```powershell\npython -B WORK_DIR\\artifacts\\stage_i_sst2\\stage_i_sst2.py --phase all --batch-size 8\n```\n\nPhases: prepare, smoke, extract, fit, evaluate, all. `all` runs training-only smoke checks before extraction, freezes every classifier before test metrics and verifies prior artifact hashes. SQLite row commits follow flushed hidden-state writes; restart resumes completed extraction rows and fitted classifiers. Bootstrap can resume at completed model boundaries. Do not alter protocol or manifests to resume. Evaluation can be reproduced with `--phase evaluate` using frozen features/models, without Qwen passes.\n\nSee protocol.json, repository_preflight.json, data_usage_audit.json, splits/, frozen_before_test.json, verification.json, metrics_all_methods.csv, paired_differences_95ci.csv and results_summary.txt. Test is internal official-training subset. No quantization.\n', encoding='utf-8')

def finish(results):
    verify_sources()
    checks = {}
    for key in KEYS:
        for split in SPLITS:
            f = pd.read_csv(OUT / key / (split + '_predictions.csv'), float_precision='round_trip')
            rr = rows(split)
            assert f.id.tolist() == [r['id'] for r in rr] and f.label.tolist() == [r['label'] for r in rr]
            y = f.label.to_numpy()
            for name, res in results[key][split].items():
                m = met(y, f[name + '_prediction'].to_numpy(), f[name + '_decision_score'].to_numpy())
                assert m == res, 'Saved predictions metrics differ'
            checks[key + '/' + split] = True
    save(OUT / 'verification.json', dict(complete=True, old_sources_unchanged=True, saved_predictions_metrics_verified=checks, splits_disjoint=True, reused_samples=0, train_only_scaler_and_pca=True, all_models_frozen_before_test=True, convergence_verified=True))
    save(OUT / 'state.json', dict(complete=True, phase='complete'))
    log('COMPLETE: all metrics and prior-source preservation verified.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', 'prepare', 'smoke', 'extract', 'fit', 'evaluate'], default='all')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    protocol = prepare()
    if args.phase == 'prepare':
        return
    if args.phase == 'smoke':
        extract(protocol, True, args.batch_size)
        return
    if args.phase == 'all':
        extract(protocol, True, args.batch_size)
    if args.phase in ('all', 'extract'):
        extract(protocol, False, args.batch_size)
    if args.phase == 'extract':
        return
    info = features(protocol)
    if args.phase in ('all', 'fit'):
        selected = fit(protocol, info)
    else:
        selected = js(OUT / 'frozen_before_test.json')['selected']
    if args.phase == 'fit':
        return
    results, metrics, changes, summaries, ci = evaluate(protocol, info, selected)
    report(protocol, info, selected, results, metrics, changes, summaries, ci)
    finish(results)
if __name__ == '__main__':
    main()
