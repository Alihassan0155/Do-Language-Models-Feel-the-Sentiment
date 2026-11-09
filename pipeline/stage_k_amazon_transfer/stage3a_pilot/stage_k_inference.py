"""Stage K shared frozen feature extractor. No classifier predictions or outcomes."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os, sys, ast, json, csv, hashlib, time, gc, sqlite3, argparse
from pathlib import Path
sys.dont_write_bytecode = True
R = WORKSPACE
K = R / 'artifacts/stage_k_amazon_transfer'
D = K / 'stage2_dataset'
P = K / 'stage3a_pilot'
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[var] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = str(R / '.cache/huggingface')
import numpy as np, joblib, torch, psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
EMOTIONS = ['joy', 'love', 'excitement', 'gratitude', 'sadness', 'anger', 'fear', 'disgust']

def dataset_sha():
    return js(D / 'dataset_metadata.json')['manifest_sha256']
TOL = {'margin': {'atol': 0.0625, 'rtol': 0.001}, 'pooled': {'atol': 0.025, 'rtol': 0.005}, 'raw': {'atol': 0.05, 'rtol': 0.005}, 'denoised': {'atol': 0.05, 'rtol': 0.005}, 'pca_rank7': {'atol': 0.05, 'rtol': 0.005}, 'random_rank7': {'atol': 0.05, 'rtol': 0.005}}

def log(s):
    print(time.strftime('%H:%M:%S') + ' ' + s, flush=True)

def js(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def sha(p):
    p = source_path(p)
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    with t.open('w', encoding='utf-8') as f:
        f.write(json.dumps(v, indent=2, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)

def atomic_npz(p, arrays):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    with t.open('wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)

def cpu_peak():
    m = psutil.Process().memory_info()
    return int(getattr(m, 'peak_wset', m.rss))

def settings():
    return js(K / 'transfer_compatibility.json')['source_settings']

def verify_dataset():
    assert sha(D / 'amazon_test_20000.jsonl') == dataset_sha(), 'Manifest hash mismatch'
    meta = js(D / 'dataset_metadata.json')
    assert meta['dataset_revision'] == js(D / 'dataset_revision.json')['revision'] and meta['inference_ready']
    counts = [0, 0]
    out = []
    for line in (D / 'amazon_test_20000.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        assert r['label'] in (0, 1)
        counts[r['label']] += 1
        assert digest(r['text']) == r['text_sha256']
        out.append({k: r[k] for k in ('evaluation_id', 'dataset_row_index', 'text', 'text_sha256')})
    assert counts == [10000, 10000] and len(out) == 20000 and (len({r['evaluation_id'] for r in out}) == 20000)
    assert [r['dataset_row_index'] for r in out] == sorted((r['dataset_row_index'] for r in out))
    return out

def mask_reference():
    text = source_path(R / 'stage_d.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'token_batch'))
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(R / 'stage_d.py'), 'exec'), namespace)
    return (namespace['token_batch'], digest(ast.get_source_segment(text, node)))

class Extractor:

    def __init__(self, key):
        assert torch.cuda.is_available(), 'CUDA required; CPU fallback prohibited'
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        np.random.seed(42)
        self.key = key
        self.cfg = next((s['identity'] for s in settings() if s['model'] == key))
        c = self.cfg
        cfg = AutoConfig.from_pretrained(c['model_id'], revision=c['checkpoint_commit'], local_files_only=True)
        assert cfg._commit_hash == c['checkpoint_commit']
        assert digest(json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False)) == c['config_sha256']
        snap = next((Path(s['snapshot']) for s in settings() if s['model'] == key))
        identity = js(R / 'artifacts/stage_i_sst2' / key / 'model_identity.json')
        weights = {p.name: sha(p) for p in snap.glob('*.safetensors')}
        assert weights == identity['weights'], 'Weight identity mismatch'
        self.tok = AutoTokenizer.from_pretrained(c['model_id'], revision=c['checkpoint_commit'], use_fast=True, local_files_only=True)
        assert digest(self.tok.backend_tokenizer.to_str()) == c['tokenizer_sha256']
        self.tok.padding_side = 'left'
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        assert {n: self.tok.encode(n, add_special_tokens=False) for n in ('positive', 'negative')} == {'positive': [30487], 'negative': [42224]}
        self.token_batch, self.mask_sha = mask_reference()
        log(f'Loading {key} CUDA FP16, revision {c['checkpoint_commit']}, layer {c['layer']}')
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(c['model_id'], revision=c['checkpoint_commit'], dtype=torch.float16, local_files_only=True, attn_implementation='sdpa').eval().to('cuda')
        self.model.requires_grad_(False)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        assert next(self.model.parameters()).dtype == torch.float16 and next(self.model.parameters()).device.type == 'cuda'
        assert len(self.model.model.layers) == c['total_layers'] and self.model.config.hidden_size == c['width'] and (not getattr(self.model, 'is_quantized', False))
        self.cache = {}
        self.mask = None
        self.hook = self.model.model.layers[c['layer'] - 1].register_forward_hook(self.capture)

    def capture(self, module, args, out):
        residual = out[0] if isinstance(out, tuple) else out
        assert residual.shape[-1] == self.cfg['width']
        self.cache['pooled'] = ((residual.float() * self.mask.unsqueeze(-1)).sum(1) / self.mask.sum(1).unsqueeze(-1)).cpu().numpy()
        if self.cache.get('verify'):
            self.cache['residual'] = residual.detach().clone()

    def batch(self, rows, prompt, verify=False):
        assert all(('label' not in r and 'label_name' not in r for r in rows)), 'Labels prohibited in inference'
        inputs, self.mask, counts, truncated = self.token_batch(self.tok, self.key, prompt, rows)
        assert not any(truncated), 'Truncation prohibited'
        self.cache['verify'] = verify
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=False, output_hidden_states=verify)
            pos = torch.arange(inputs['input_ids'].shape[1], device='cuda').expand_as(inputs['input_ids'])
            last = pos.masked_fill(~inputs['attention_mask'].bool(), -1).max(1).values
            lp = torch.log_softmax(output.logits[torch.arange(len(rows), device='cuda'), last].float(), -1)[:, [30487, 42224]].cpu().numpy()
            h = self.cache['pooled'].copy()
            m = (lp[:, 0] - lp[:, 1]).astype(np.float64)
            if verify:
                assert torch.equal(self.cache['residual'], output.hidden_states[self.cfg['layer']])
                manual = ((output.hidden_states[self.cfg['layer']].float() * self.mask.unsqueeze(-1)).sum(1) / self.mask.sum(1).unsqueeze(-1)).cpu().numpy()
                assert np.array_equal(manual, h)
                del output
                self.hook.remove()
                try:
                    plain = self.model(**inputs, use_cache=False)
                    lp2 = torch.log_softmax(plain.logits[torch.arange(len(rows), device='cuda'), last].float(), -1)[:, [30487, 42224]].cpu().numpy()
                    assert np.array_equal(lp, lp2), 'Hook changed logits'
                    del plain
                finally:
                    self.hook = self.model.model.layers[self.cfg['layer'] - 1].register_forward_hook(self.capture)
                self.cache.pop('residual', None)
            else:
                del output
        result = dict(margin=m, log_probabilities=lp, pooled=h, text_token_counts=np.asarray(counts, dtype=np.int64), input_lengths=inputs['attention_mask'].sum(1).cpu().numpy())
        self.cache.clear()
        self.mask = None
        del inputs
        assert h.shape == (len(rows), self.cfg['width']) and np.isfinite(h).all() and np.isfinite(m).all()
        return result

    def close(self):
        self.hook.remove()
        del self.model, self.tok, self.cache
        gc.collect()
        torch.cuda.empty_cache()

class Transforms:

    def __init__(self, s):
        self.s = s
        self.base = R / 'artifacts' / ('stage_i_sst2' if s['source'] == 'sst2' else 'stage_i_tweeteval_matched_controls') / s['model']
        self.paths = {}
        self.unit = {}
        for name in ('raw', 'denoised'):
            p = self.base / 'bases' / (name + '_unit_vectors.npy')
            self.unit[name] = np.load(p)
            self.paths[str(p)] = sha(p)
            vp = Path(s['vectors'][name])
            expected = s['identity']['raw_sha256' if name == 'raw' else 'final_sha256']
            assert sha(vp) == expected
            v = np.asarray(np.load(vp)[:, s['identity']['layer'] - 1], dtype=np.float64)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            assert np.array_equal(v, self.unit[name])
        p = self.base / 'bases/pca_rank7.joblib'
        self.pca = joblib.load(p)
        self.paths[str(p)] = sha(p)
        assert self.pca.n_samples_ == 14186 and (not self.pca.whiten) and (self.pca.components_.shape == (7, s['identity']['width']))
        random = []
        for seed in range(2000, 2050):
            p = self.base / 'bases' / f'random_rank7_seed_{seed}.npy'
            v = np.load(p)
            assert v.shape == (7, s['identity']['width']) and np.allclose(v @ v.T, np.eye(7), atol=1e-12)
            random.append(v)
            self.paths[str(p)] = sha(p)
        self.random = np.stack(random)

    def apply(self, m, h):
        hb = np.asarray(h, dtype=np.float64)
        m = np.asarray(m, dtype=np.float64)
        out = {'sentiment_only': m[:, None]}
        for name in ('raw', 'denoised'):
            out[name] = np.column_stack([m, hb @ self.unit[name].T])
        out['pca_rank7'] = np.column_stack([m, self.pca.transform(hb)])
        out['random_rank7'] = np.stack([np.column_stack([m, hb @ v.T]) for v in self.random], axis=1)
        assert all((np.isfinite(a).all() for a in out.values()))
        return out

    def signature(self, e):
        return dict(manifest_sha256=dataset_sha(), source=self.s['source'], model=self.s['model'], identity=self.s['identity'], prompt=self.s['prompt'], mask_function_sha256=e.mask_sha, transform_sha256=self.paths, precision='FP16 SDPA forward; float32 log_softmax and subtraction; float32 mean; float64 projections', emotion_order=EMOTIONS, random_seeds=list(range(2000, 2050)), torch=torch.__version__, transformers=__import__('transformers').__version__, source_limit=1024, scoring_candidate_ids={'positive': 30487, 'negative': 42224})

def verify_classifiers(s, features):
    matches = [r for r in csv.DictReader((K / 'source_model_inventory.csv').open(encoding='utf-8')) if r['source'] == s['source'] and r['model'] == s['model']]
    assert len(matches) == 54
    for r in matches:
        assert sha(r['classifier_path']) == r['sha256']
        p = joblib.load(r['classifier_path'])
        name = r['method']
        x = features['random_rank7'][:, int(name.rsplit('_', 1)[1]) - 2000, :] if name.startswith('random_rank7_seed_') else features[name]
        assert x.shape[1] == int(r['feature_count']) == p.n_features_in_ and int(p.named_steps['scaler'].n_samples_seen_) == 14186
        z = p.named_steps['scaler'].transform(x)
        assert z.shape == x.shape and np.isfinite(z).all()
        assert np.array_equal(p.named_steps['classifier'].classes_, [0, 1])
        assert p.named_steps['classifier'].C == float(r['C'])
    return dict(count=54, passed=True, predictions_computed=False, feature_order={'sentiment_only': ['margin'], 'raw': ['margin', *EMOTIONS], 'denoised': ['margin', *EMOTIONS], 'pca_rank7': ['margin', *[f'pc{i}' for i in range(1, 8)]], 'random_rank7': ['margin', *[f'coordinate{i}' for i in range(1, 8)]]})

def run_chunks(e, s, rows, folder, batch_size=4, chunk_size=8):
    folder.mkdir(parents=True, exist_ok=True)
    tf = Transforms(s)
    sig = tf.signature(e)
    sigsha = digest(json.dumps(sig, sort_keys=True))
    protocol = folder / 'protocol.json'
    if protocol.exists():
        assert js(protocol) == sig, 'Resume protocol mismatch'
    else:
        atomic_json(protocol, sig)
    chunks = folder / 'chunks'
    chunks.mkdir(exist_ok=True)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    oom = 0
    active = batch_size
    collected = []
    new = 0
    ntokens = 0
    forward_seconds = 0.0
    for offset in range(0, len(rows), chunk_size):
        block = rows[offset:offset + chunk_size]
        file = chunks / f'{offset:06d}.npz'
        marker = file.with_suffix('.json')
        ids = [r['evaluation_id'] for r in block]
        if marker.exists():
            md = js(marker)
            assert md['signature_sha256'] == sigsha and md['sha256'] == sha(file) and (md['ids'] == ids)
            with np.load(file, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            assert a['ids'].tolist() == ids
            assert a['pooled'].shape == (len(block), e.cfg['width'])
            assert all((np.isfinite(v).all() for k, v in a.items() if k != 'ids'))
            collected.append(a)
            log(f'Resume verified {s['source']}/{s['model']} {offset + len(block)}/{len(rows)}')
            continue
        pieces = []
        j = 0
        while j < len(block):
            subset = block[j:j + active]
            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                a = e.batch(subset, s['prompt'])
                torch.cuda.synchronize()
                forward_seconds += time.perf_counter() - t
            except torch.cuda.OutOfMemoryError:
                oom += 1
                e.cache.clear()
                e.mask = None
                gc.collect()
                torch.cuda.empty_cache()
                if active == 1:
                    raise
                active = max(1, active // 2)
                log(f'CUDA OOM retry unsaved rows at batch{active}')
                continue
            a.update(tf.apply(a['margin'], a['pooled']))
            a['ids'] = np.array([r['evaluation_id'] for r in subset])
            pieces.append(a)
            new += len(subset)
            ntokens += int(a['input_lengths'].sum())
            j += len(subset)
        a = {k: np.concatenate([v[k] for v in pieces], axis=0) for k in pieces[0]}
        assert a['ids'].tolist() == ids
        atomic_npz(file, a)
        atomic_json(marker, dict(signature_sha256=sigsha, sha256=sha(file), ids=ids, n=len(block), offset=offset))
        collected.append(a)
        elapsed = time.perf_counter() - start
        free, total = torch.cuda.mem_get_info()
        log(f'{s['source']}/{s['model']}: saved {offset + len(block)}/{len(rows)}; batch{active}; elapsed{elapsed:.1f}s; GPU allocated{torch.cuda.memory_allocated() / 2 ** 30:.2f}GiB free{free / 2 ** 30:.2f}GiB')
    arrays = {k: np.concatenate([v[k] for v in collected], axis=0) for k in collected[0]}
    assert arrays['ids'].tolist() == [r['evaluation_id'] for r in rows] and len(set(arrays['ids'].tolist())) == len(rows)
    check = verify_classifiers(s, {k: arrays[k][:32] for k in ('sentiment_only', 'raw', 'denoised', 'pca_rank7', 'random_rank7')})
    if new == 0 and (folder / 'completion.json').exists():
        previous = js(folder / 'completion.json')
        assert previous['signature_sha256'] == sigsha and previous['n'] == len(rows)
        assert previous['ids_sha256'] == digest('\n'.join(arrays['ids'].tolist()))
        assert previous['benchmark']['chunk_size'] == chunk_size and previous['benchmark']['requested_batch_size'] == batch_size, 'Resume chunk/batch settings changed'
        log('All chunks verified; preserving original benchmark measurements')
        return (arrays, previous['benchmark'])
    elapsed = time.perf_counter() - start
    bench = dict(configuration=s['source'] + '_' + s['model'], model_load_seconds=e.load_seconds, load_shared_between_two_prompts=True, n=len(rows), newly_processed=new, total_wall_seconds=elapsed, forward_and_tokenization_seconds=forward_seconds, seconds_per_example=forward_seconds / new if new else None, input_tokens=int(arrays['input_lengths'].sum()), input_tokens_per_second=ntokens / forward_seconds if forward_seconds else None, requested_batch_size=batch_size, final_batch_size=active, chunk_size=chunk_size, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(), process_lifetime_peak_cpu_ram_bytes=cpu_peak(), oom_events=oom, cuda_device=torch.cuda.get_device_name(0), classifier_check=check, complete=True)
    atomic_json(folder / 'completion.json', dict(n=len(rows), ids_sha256=digest('\n'.join(arrays['ids'].tolist())), signature_sha256=sigsha, benchmark=bench))
    return (arrays, bench)
