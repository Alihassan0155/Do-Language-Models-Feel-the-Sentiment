"""Stage H: fixed-layer sentiment ablation with paired performance/damage CIs."""
from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent) if (Path(__file__).parent / 'stage_e_common.py').exists() else str(CODE_ROOT))
import argparse, csv, json, hashlib, sqlite3, gc
import numpy as np
from stage_e_common import ROOT, ART, E, MODELS, DATASETS, EMOTIONS, atomic_json, sha
from stage_e_pipeline import assert_protocol, emotion_inputs, load_hidden, fit_pca, random_directions
from stage_f import basis
from stage_g_confirmation import bootstrap_weights
PRIMARY = ('0.5b', '3b', 'bert_large_uncased')
PRECISION = {'0.5b': 'float16', '1.5b': 'float32', '3b': 'float16', 'bert_large_uncased': 'float16'}

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def prepare(args):
    p = json.loads((E / 'protocol.json').read_text(encoding='utf-8'))
    assert_protocol(p)
    source = ART / 'stage_g_confirmation' / 'samples.json'
    samples = json.loads(source.read_text(encoding='utf-8'))
    for dataset, rows in samples.items():
        if len(rows) != 600 or len({r['id'] for r in rows}) != 600 or {r['label'] for r in rows} != {0, 1}:
            raise ValueError('Expected frozen 600-example binary sample.')
    gate = ART / 'stage_g_confirmation' / 'positive_control' / 'gate.json'
    control = json.loads(gate.read_text(encoding='utf-8'))
    config = dict(schema=1, stage_e_protocol_sha256=sha(E / 'protocol.json'), sample_sha256=sha(source), stage_g_control_sha256=sha(gate), stage_g_1_5b_control_passed=control['passed'], primary_models=list(PRIMARY), exploratory_models=['1.5b'], models=p['models'], precision=PRECISION, batch_size=args.batch_size, random_seeds=list(range(2000, 2000 + args.random_count)), bootstrap_draws=args.bootstrap, bootstrap_seed=44, centering=args.centering, projection='h - ((h-anchor) @ U) @ U.T; text-token post-block residuals only', anchor='zero for primary uncentered removal; training pooled-state mean for optional train-centered sensitivity analysis', rank_tolerance=1e-05, controls='50+ random rank-7 and rank-1 bases; PCA rank-7/rank-1 fit on sentiment training states only', prompts='fixed T1', margin='full-vocabulary logP(positive)-logP(negative); exact Stage A readout position', classification='candidate-label argmax; TweetEval neutral predictions count as errors on the binary subset; macro-F1 over negative/positive gold classes', auroc='binary positive vs negative, ranked by sentiment margin M', bootstrap='paired, class-stratified example resampling; random controls conditional on sampled directions', caution='Ablation damage is specific to this intervention/site; higher distortion is a confound for emotion-specific necessity. Previously observed test/proxy data.')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('protocol.json', config), ('samples.json', samples)]:
        path = args.output / name
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Stage H settings changed; use a separate --output folder.')
        if not path.exists():
            atomic_json(path, value)
    return (p, config, samples)

def build_bases(p, key, dataset, config, folder):
    target = folder / key / dataset
    target.mkdir(parents=True, exist_ok=True)
    source = args_root = Path(config['_root'])
    cache = source / key / dataset / 'bases.npz'
    metadata_path = source / key / dataset / 'bases_metadata.json'
    if cache.exists():
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if meta['protocol_sha256'] != sha(source / 'protocol.json') or meta['bases_sha256'] != sha(cache):
            raise ValueError('Basis cache changed.')
        with np.load(cache) as saved:
            values = {n: saved[n].copy() for n in saved.files if n != 'anchor'}
            anchor = saved['anchor'].copy()
        return (values, anchor, meta)
    _, info = emotion_inputs(key)
    layer = info['layer'] - 1
    raw = np.asarray(np.load(ART / 'stage_b_raw_vectors' / key / 'emotion_directions_raw.float32.npy')[:, layer], dtype=np.float64)
    final = np.asarray(np.load(ART / 'stage_b_final_vectors' / key / 'emotion_directions_final.float32.npy')[:, layer], dtype=np.float64)
    h, y, _, train_meta = load_hidden(p, key, dataset, 'train')
    sentiment = h[y == 1].mean(axis=0, dtype=np.float64) - h[y == 0].mean(axis=0, dtype=np.float64)
    if not np.allclose(sentiment, np.load(ART / 'stage_f' / key / dataset / 'sentiment_direction.npy'), atol=1e-10, rtol=1e-10):
        raise ValueError('Direct sentiment source mismatch.')
    values = {}
    for name, v in [('emotion_raw', raw), ('emotion_denoised', final), ('valence_raw', (raw[:4].mean(axis=0) - raw[4:].mean(axis=0))[None]), ('valence_denoised', (final[:4].mean(axis=0) - final[4:].mean(axis=0))[None]), ('direct_sentiment', sentiment[None])]:
        u, _ = basis(v)
        values[name] = u.T.astype(np.float32)
    if values['emotion_raw'].shape[0] != 7 or values['emotion_denoised'].shape[0] != 7:
        raise ValueError('Expected emotion rank 7; revise matched controls if ranks change.')
    pca = fit_pca(h, 42)
    values['pca_rank7'] = pca.components_[:7].astype(np.float32)
    values['pca_rank1'] = pca.components_[:1].astype(np.float32)
    for seed in config['random_seeds']:
        random = random_directions(info['width'], seed, 7).astype(np.float32)
        values[f'random_rank7_seed_{seed}'] = random
        values[f'random_rank1_seed_{seed}'] = random[:1]
    anchor = np.zeros(info['width'], dtype=np.float32) if config['centering'] == 'none' else h.mean(axis=0, dtype=np.float64).astype(np.float32)
    for v in values.values():
        if not np.allclose(v @ v.T, np.eye(len(v)), atol=2e-05):
            raise ValueError('Nonorthogonal ablation basis.')
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, anchor=anchor, **values)
    meta = dict(protocol_sha256=sha(source / 'protocol.json'), bases_sha256=sha(cache), train_hidden_sha256=train_meta['hidden_sha256'], ranks={n: len(v) for n, v in values.items()}, layer=info['layer'], model_role='primary' if key in PRIMARY else 'exploratory', failed_stage_g_control=key == '1.5b' and (not config['stage_g_1_5b_control_passed']))
    atomic_json(metadata_path, meta)
    return (values, anchor, meta)

def remove(output, mask, basis_rows, anchor):
    h = output[0] if isinstance(output, tuple) else output
    projected = (h.float() - anchor) @ basis_rows.T @ basis_rows
    changed = (h.float() - projected * mask.unsqueeze(-1)).to(h.dtype)
    return (changed, *output[1:]) if isinstance(output, tuple) else changed

def database(folder):
    db = sqlite3.connect(folder / 'scores.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS scores(model TEXT,dataset TEXT,method TEXT,example_index INTEGER,predicted INTEGER,logp_positive REAL,logp_negative REAL,logp_neutral REAL,margin REAL,token_removed_energy_fraction REAL,token_relative_change REAL,pooled_cosine REAL,pooled_norm_ratio REAL,PRIMARY KEY(model,dataset,method,example_index))')
    return db

def execute(p, config, samples, args, keys, smoke=False):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
    from stage_d import token_batch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback.')
    folder = args.output / 'smoke' if smoke else args.output
    folder.mkdir(parents=True, exist_ok=True)
    db = database(folder)
    try:
        for key in keys:
            info = p['models'][key]
            tok = AutoTokenizer.from_pretrained(MODELS[key], use_fast=True, local_files_only=True, revision=model_revision(MODELS[key]))
            if hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest() != info['tokenizer_sha256']:
                raise ValueError('Tokenizer changed.')
            tok.padding_side = 'right' if key == 'bert_large_uncased' else 'left'
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            ids = {}
            for label in ('positive', 'negative'):
                tokens = tok.encode(label, add_special_tokens=False)
                if len(tokens) != 1:
                    raise ValueError('Expected single-token candidates.')
                ids[label] = tokens[0]
            loader = AutoModelForMaskedLM if key == 'bert_large_uncased' else AutoModelForCausalLM
            model = loader.from_pretrained(MODELS[key], dtype=getattr(torch, PRECISION[key]), local_files_only=True, revision=model_revision(MODELS[key])).eval().to('cuda')
            blocks = model.bert.encoder.layer if key == 'bert_large_uncased' else model.model.layers
            if len(blocks) != info['total_layers'] or model.config.hidden_size != info['width']:
                raise ValueError('Physical model mismatch.')
            block = blocks[info['layer'] - 1]

            def score(rows, dataset, v=None, anchor=None):
                inputs, mask, _, clipped = token_batch(tok, key, info['prompts'][dataset], rows)
                if any(clipped):
                    raise ValueError('Text truncation forbidden.')
                distortion = []
                handle = None
                if v is not None:
                    u = torch.tensor(v, device='cuda', dtype=torch.float32)
                    mu = torch.tensor(anchor, device='cuda', dtype=torch.float32)

                    def hook(module, arguments, output):
                        original = output[0] if isinstance(output, tuple) else output
                        result = remove(output, mask, u, mu)
                        changed = result[0] if isinstance(result, tuple) else result
                        a = original.float()
                        b = changed.float()
                        delta = b - a
                        weights = mask.unsqueeze(-1)
                        counts = mask.sum(dim=1).clamp_min(1)
                        energies = delta.square().sum(dim=-1) / a.square().sum(dim=-1).clamp_min(1e-20)
                        fraction = (energies * mask).sum(dim=1) / counts
                        relative = (energies.sqrt() * mask).sum(dim=1) / counts
                        pooled_a = (a * weights).sum(dim=1) / counts.unsqueeze(-1)
                        pooled_b = (b * weights).sum(dim=1) / counts.unsqueeze(-1)
                        co = F.cosine_similarity(pooled_a, pooled_b, dim=-1, eps=1e-12)
                        ratio = pooled_b.norm(dim=-1) / pooled_a.norm(dim=-1).clamp_min(1e-12)
                        distortion.append(torch.stack([fraction, relative, co, ratio], dim=1).cpu().numpy())
                        return result
                    handle = block.register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        kwargs = dict(inputs)
                        if key != 'bert_large_uncased':
                            kwargs['use_cache'] = False
                        output = model(**kwargs)
                        if key == 'bert_large_uncased':
                            positions = inputs['input_ids'].eq(tok.mask_token_id)
                            if not torch.all(positions.sum(dim=1) == 1):
                                raise ValueError('Expected one BERT mask.')
                            logits = output.logits[positions].float()
                        else:
                            positions = torch.arange(inputs['input_ids'].shape[1], device='cuda').expand_as(inputs['input_ids'])
                            last = positions.masked_fill(~inputs['attention_mask'].bool(), -1).max(dim=1).values
                            logits = output.logits[torch.arange(len(rows), device='cuda'), last].float()
                        labels = ['negative', 'positive']
                        pred = logits[:, [ids[n] for n in labels]].argmax(dim=-1).cpu().tolist()
                        pred = [{'negative': 0, 'positive': 1, 'neutral': 2}[labels[i]] for i in pred]
                        lp = torch.log_softmax(logits, dim=-1)[:, [ids[n] for n in ('positive', 'negative')]].cpu().numpy()
                        if not np.isfinite(lp).all():
                            raise ValueError('Nonfinite scores.')
                        dist = distortion[0] if distortion else np.tile([0.0, 0.0, 1.0, 1.0], (len(rows), 1))
                        return (lp, pred, dist)
                finally:
                    if handle is not None:
                        handle.remove()
            try:
                for dataset in DATASETS:
                    values, anchor, meta = build_bases(p, key, dataset, config, folder)
                    names = ['emotion_raw', 'emotion_denoised', 'valence_raw', 'valence_denoised', 'direct_sentiment', 'pca_rank7', f'random_rank7_seed_{config['random_seeds'][0]}', f'random_rank1_seed_{config['random_seeds'][0]}'] if smoke else list(values)
                    rows = samples[dataset]
                    n = min(8, len(rows)) if smoke else len(rows)
                    for method in ['baseline'] + names:
                        done = {r[0] for r in db.execute('SELECT example_index FROM scores WHERE model=? AND dataset=? AND method=?', (key, dataset, method))}
                        pending = [i for i in range(n) if i not in done]
                        for chunk in sorted({i // args.batch_size for i in pending}):
                            start = chunk * args.batch_size
                            batch_indices = list(range(start, min(start + args.batch_size, len(rows))))
                            lp, pred, dist = score([rows[i] for i in batch_indices], dataset, None if method == 'baseline' else values[method], anchor)
                            indices = [i for i in batch_indices if i < n and i not in done]
                            records = [(key, dataset, method, i, pred[i - start], float(lp[i - start, 0]), float(lp[i - start, 1]), None, float(lp[i - start, 0] - lp[i - start, 1]), *[float(x) for x in dist[i - start]]) for i in indices]
                            with db:
                                db.executemany('INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', records)
                        print(f'H {key}/{dataset}/{method}: {n} saved', flush=True)
                    a, _, _ = score(rows[:1], dataset)
                    b, _, _ = score(rows[:1], dataset, np.empty((0, info['width']), dtype=np.float32), anchor)
                    if not np.array_equal(a, b):
                        raise ValueError('Empty-subspace hook differs from baseline.')
                    target = folder / key / dataset
                    target.mkdir(parents=True, exist_ok=True)
                    atomic_json(target / 'checks.json', dict(empty_subspace_equal=True, candidate_ids=ids, physical_block_0_based=info['layer'] - 1, model_role=meta['model_role'], smoke=smoke))
            finally:
                del model, blocks, block, tok
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        db.close()

def metrics(y, pred, margin, w):
    accuracy = w @ (pred == y)
    f1 = []
    for label in (0, 1):
        tp = w @ ((pred == label) & (y == label))
        den = w @ (pred == label) + w @ (y == label)
        f1.append(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0))
    order = np.argsort(margin, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(margin[order]) != 0) + 1]
    positive = np.add.reduceat(w[:, order] * (y[order] == 1), starts, axis=1)
    negative = np.add.reduceat(w[:, order] * (y[order] == 0), starts, axis=1)
    before = negative.cumsum(axis=1) - negative
    auc = (positive * (before + 0.5 * negative)).sum(axis=1) / (positive.sum(axis=1) * negative.sum(axis=1))
    return np.stack([accuracy, (f1[0] + f1[1]) / 2, auc], axis=1)

def interval(x):
    return [float(v) for v in np.quantile(x, [0.025, 0.975])]

def report(p, config, samples, args):
    db = database(args.output)
    db.row_factory = sqlite3.Row
    performance = []
    comparisons = []
    states = {}
    metric_names = ['accuracy', 'macro_f1', 'auroc']
    for key in MODELS:
        for dataset in DATASETS:
            rows = samples[dataset]
            n = len(rows)
            y = np.array([r['label'] for r in rows])
            w = bootstrap_weights(y, args.bootstrap, 44)
            point = np.full((1, n), 1 / n)

            def get(name):
                records = [dict(r) for r in db.execute('SELECT * FROM scores WHERE model=? AND dataset=? AND method=? ORDER BY example_index', (key, dataset, name))]
                return records if [r['example_index'] for r in records] == list(range(n)) else None
            baseline = get('baseline')
            states[f'{key}/{dataset}'] = 'not run'
            if baseline is None:
                continue

            def compute(records):
                pred = np.array([r['predicted'] for r in records])
                m = np.array([r['margin'] for r in records])
                return (metrics(y, pred, m, point)[0], metrics(y, pred, m, w), m)
            bp, bb, bm = compute(baseline)
            values, _, meta = build_bases(p, key, dataset, config, args.output)
            damage_boot = {}
            damage_point = {}
            all_done = True
            for method in ['baseline'] + list(values):
                records = get(method)
                if records is None:
                    all_done = False
                    continue
                mp, mb, mm = compute(records)
                damage = bp - mp
                boot_damage = bb - mb
                damage_boot[method] = boot_damage
                damage_point[method] = damage
                delta = mm - bm
                mean_delta_ci = interval(w @ delta)
                role = meta['model_role']
                for j, name in enumerate(metric_names):
                    ci = interval(boot_damage[:, j])
                    pci = interval(mb[:, j])
                    performance.append(dict(model=key, model_role=role, dataset=dataset, method=method, rank=0 if method == 'baseline' else len(values[method]), n=n, metric=name, baseline=float(bp[j]), ablated=float(mp[j]), ablated_ci_low=pci[0], ablated_ci_high=pci[1], damage=float(damage[j]), damage_ci_low=ci[0], damage_ci_high=ci[1], mean_delta_margin=float(delta.mean()), delta_margin_ci_low=mean_delta_ci[0], delta_margin_ci_high=mean_delta_ci[1], mean_removed_token_energy=float(np.mean([r['token_removed_energy_fraction'] for r in records])), mean_relative_token_change=float(np.mean([r['token_relative_change'] for r in records])), mean_pooled_cosine=float(np.mean([r['pooled_cosine'] for r in records]))))
            states[f'{key}/{dataset}'] = 'complete' if all_done else 'partial'
            for method in ('emotion_raw', 'emotion_denoised', 'valence_raw', 'valence_denoised', 'direct_sentiment', 'pca_rank7', 'pca_rank1'):
                if method not in damage_boot:
                    continue
                rank = len(values[method])
                names = [n for n in values if n.startswith(f'random_rank{rank}_')]
                if any((n not in damage_boot for n in names)):
                    continue
                null = np.stack([damage_boot[n] for n in names], axis=1)
                null_point = np.stack([damage_point[n] for n in names])
                for j, name in enumerate(metric_names):
                    random_mean_boot = null[:, :, j].mean(axis=1)
                    random_max_boot = null[:, :, j].max(axis=1)
                    excess = damage_boot[method][:, j] - random_mean_boot
                    strongest = damage_boot[method][:, j] - random_max_boot
                    ci = interval(excess)
                    sci = interval(strongest)
                    rci = interval(random_mean_boot)
                    rmci = interval(random_max_boot)
                    comparisons.append(dict(model=key, model_role=meta['model_role'], dataset=dataset, method=method, rank=rank, metric=name, random_controls=len(names), damage=float(damage_point[method][j]), mean_random_damage=float(null_point[:, j].mean()), mean_random_damage_ci_low=rci[0], mean_random_damage_ci_high=rci[1], max_random_damage_ci_low=rmci[0], max_random_damage_ci_high=rmci[1], excess_damage=float(damage_point[method][j] - null_point[:, j].mean()), excess_ci_low=ci[0], excess_ci_high=ci[1], max_random_damage=float(null_point[:, j].max()), excess_over_max_ci_low=sci[0], excess_over_max_ci_high=sci[1]))
    db.close()
    write_csv(args.output / 'performance_damage_paired_ci.csv', performance)
    write_csv(args.output / 'damage_vs_random_paired_ci.csv', comparisons)
    primary_complete = all((states[f'{k}/{d}'] == 'complete' for k in PRIMARY for d in DATASETS))
    all_complete = all((v == 'complete' for v in states.values()))
    atomic_json(args.output / 'state.json', dict(primary_complete=primary_complete, all_models_complete=all_complete, model_datasets=states))
    lines = [f'Stage H primary complete={primary_complete}; all including exploratory 1.5B complete={all_complete}.', 'Damage=baseline minus ablated metric; positive means worse. AUROC uses positive-vs-negative margin; macro-F1 uses both gold classes.', 'Uncentered removal unless protocol specifies train-centered sensitivity. All controls rank matched; PCA/direct sentiment fitted on training only.', 'Paired stratified example bootstrap 95% CIs; random intervals conditional on sampled bases; no multiple-testing correction.', 'Representation distortion must be considered: disproportionate damage under a larger perturbation does not uniquely establish emotion necessity.', '1.5B exploratory because its Stage G positive control failed. Source data were previously observed; not a new untouched benchmark.']
    for r in performance:
        if r['method'] in ('baseline', 'emotion_raw', 'emotion_denoised', 'valence_raw', 'valence_denoised', 'direct_sentiment', 'pca_rank7'):
            lines.append(f'{r['model']} ({r['model_role']}) {r['dataset']} {r['method']} {r['metric']}: baseline={r['baseline']:.4f}, ablated={r['ablated']:.4f}, damage={r['damage']:+.4f} CI=[{r['damage_ci_low']:+.4f},{r['damage_ci_high']:+.4f}], token energy removed={r['mean_removed_token_energy']:.4f}.')
    lines += ['', 'RANK-MATCHED RANDOM DAMAGE COMPARISONS', 'All numbers below are percentage points (accuracy, macro-F1, AUROC). Positive damage means worse; negative means improved.', 'Random mean/max refer to the same rank and examples. Excess is method damage minus mean random damage.', 'CI above zero supports excess damage over those sampled controls; crossing zero is inconclusive. These are pointwise paired bootstrap intervals.']
    for key in MODELS:
        for dataset in DATASETS:
            group = [r for r in comparisons if r['model'] == key and r['dataset'] == dataset]
            lines.append(f'\n{key} ({('primary' if key in PRIMARY else 'exploratory')}) / {dataset}:')
            for rank in (7, 1):
                for metric in metric_names:
                    refs = [r for r in group if r['rank'] == rank and r['metric'] == metric]
                    if not refs:
                        lines.append(f'  rank-{rank} {metric}: comparison unavailable; not all random controls/methods complete.')
                        continue
                    r = refs[0]
                    lines.append(f'  RANDOM rank-{rank} {metric}, {r['random_controls']} controls: mean damage={100 * r['mean_random_damage']:+.3f} pp CI=[{100 * r['mean_random_damage_ci_low']:+.3f},{100 * r['mean_random_damage_ci_high']:+.3f}]; max observed damage={100 * r['max_random_damage']:+.3f} pp.')
                    for r in refs:
                        verdict = 'exceeds random mean' if r['excess_ci_low'] > 0 else 'less damage than random mean' if r['excess_ci_high'] < 0 else 'inconclusive vs random mean'
                        strongest = 'exceeds strongest sampled random control' if r['excess_over_max_ci_low'] > 0 else 'not established above strongest sampled random control'
                        lines.append(f'    {r['method']}: damage={100 * r['damage']:+.3f} pp; excess vs random mean={100 * r['excess_damage']:+.3f} pp CI=[{100 * r['excess_ci_low']:+.3f},{100 * r['excess_ci_high']:+.3f}]; excess vs max CI=[{100 * r['excess_over_max_ci_low']:+.3f},{100 * r['excess_over_max_ci_high']:+.3f}]; {verdict}; {strongest}.')
    (args.output / 'stage_h_results_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved Stage H summary; primary complete={primary_complete}; all complete={all_complete}.')

def self_test():
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    h = torch.tensor([[[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]]])
    mask = torch.tensor([[True, False]])
    u = torch.tensor([[1.0, 0.0, 0.0]])
    anchor = torch.zeros(3)
    changed = remove(h, mask, u, anchor)
    assert torch.equal(changed[0, 0], torch.tensor([0.0, 4.0, 5.0])) and torch.equal(changed[~mask], h[~mask])
    assert torch.equal(remove(changed, mask, u, anchor), changed)
    assert torch.equal(remove(h, mask, torch.empty(0, 3), anchor), h)
    y = np.array([0, 0, 1, 1, 1, 0])
    pred = np.array([0, 2, 1, 0, 1, 1])
    m = np.array([0.0, 0.5, 0.5, 1.0, 2.0, 0.0])
    w = np.full((1, 6), 1 / 6)
    measured = metrics(y, pred, m, w)[0]
    expected = [accuracy_score(y, pred), f1_score(y, pred, labels=[0, 1], average='macro'), roc_auc_score(y, m)]
    assert np.allclose(measured, expected)
    reps = np.array([0, 1, 1, 2, 4, 5])
    weights = np.bincount(reps, minlength=6)[None] / 6
    assert np.allclose(metrics(y, pred, m, weights)[0], [accuracy_score(y[reps], pred[reps]), f1_score(y[reps], pred[reps], labels=[0, 1], average='macro'), roc_auc_score(y[reps], m[reps])])
    print('Projection/idempotence/masking/zero-basis and weighted paired metrics with ties/neutral predictions passed.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'run', 'report', 'all'], default='prepare')
    parser.add_argument('--model', choices=[*MODELS, 'primary', 'all'], default='all')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--random-count', type=int, default=50)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--centering', choices=['none', 'train'], default='none')
    parser.add_argument('--output', type=Path, default=ART / 'stage_h')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.batch_size < 1 or args.random_count < 50 or args.bootstrap < 100:
        parser.error('Use positive batch size, 50+ controls and 100+ bootstrap draws.')
    args.output = args.output.resolve()
    p, config, samples = prepare(args)
    config = dict(config, _root=str(args.output))
    keys = list(MODELS) if args.model == 'all' else list(PRIMARY) if args.model == 'primary' else [args.model]
    if args.phase == 'smoke':
        execute(p, config, samples, args, keys, True)
    if args.phase in ('run', 'all'):
        execute(p, config, samples, args, keys)
    if args.phase in ('report', 'all'):
        report(p, config, samples, args)
    print(f'Stage H ready: 600 examples/dataset; {7 + 2 * args.random_count} ablations plus baseline; 1.5B exploratory.')
if __name__ == '__main__':
    main()
