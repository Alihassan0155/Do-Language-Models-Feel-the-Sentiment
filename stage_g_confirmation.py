"""Stage G: frozen-layer, norm-matched sentiment steering with resumable SQLite output."""
from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
from pathlib import Path
if (Path(__file__).parent / 'stage_e_common.py').exists():
    sys.path.insert(0, str(Path(__file__).parent))
else:
    sys.path.insert(0, str(CODE_ROOT))
import argparse, csv, json, hashlib, sqlite3, gc
import numpy as np
from stage_e_common import ROOT, ART, E, MODELS, DATASETS, EMOTIONS, atomic_json, read_rows, sha
from stage_e_pipeline import assert_protocol, emotion_inputs, load_hidden, random_directions
OUT = ART / 'stage_g_confirmation'
ALPHAS = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)

def pilot_directions(p, key, dataset):
    _, info = emotion_inputs(key)
    layer = info['layer'] - 1
    raw = np.asarray(np.load(ART / 'stage_b_raw_vectors' / key / 'emotion_directions_raw.float32.npy')[:, layer], dtype=np.float64)
    final = np.asarray(np.load(ART / 'stage_b_final_vectors' / key / 'emotion_directions_final.float32.npy')[:, layer], dtype=np.float64)
    h, y, _, meta = load_hidden(p, key, dataset, 'train')
    sentiment = h[y == 1].mean(axis=0, dtype=np.float64) - h[y == 0].mean(axis=0, dtype=np.float64)
    if not np.allclose(sentiment, np.load(ART / 'stage_f' / key / dataset / 'sentiment_direction.npy'), atol=1e-10, rtol=1e-10):
        raise ValueError('Stage F sentiment source mismatch.')
    values = {}
    signs = {}
    scale = float(np.median(np.linalg.norm(raw, axis=1)))
    for kind, array in [('raw', raw), ('denoised', final)]:
        for i, emotion in enumerate(EMOTIONS):
            name = f'emotion_{kind}:{emotion}'
            values[name] = array[i]
            signs[name] = 1 if i < 4 else -1
        name = f'valence_{kind}'
        values[name] = array[:4].mean(axis=0) - array[4:].mean(axis=0)
        signs[name] = 1
    values['direct_sentiment'] = sentiment
    signs['direct_sentiment'] = 1
    for seed in range(10):
        name = f'random_seed_{seed}'
        values[name] = random_directions(info['width'], seed, 1)[0]
        signs[name] = 0
    norms = {name: float(np.linalg.norm(v)) for name, v in values.items()}
    if any((n <= 0 for n in norms.values())) or any((not np.isfinite(v).all() for v in values.values())):
        raise ValueError('Invalid direction.')
    values = {name: np.asarray(v / norms[name] * scale, dtype=np.float32) for name, v in values.items()}
    return (values, dict(reference_norm=scale, original_norms=norms, expected_signs=signs, layer=info['layer'], raw_sha256=info['raw_sha256'], final_sha256=info['final_sha256'], sentiment_train_hidden_sha256=meta['hidden_sha256']))

def intervene(output, mask, delta):
    h = output[0] if isinstance(output, tuple) else output
    changed = h + mask.unsqueeze(-1).to(h.dtype) * delta.to(h.device, h.dtype)
    return (changed, *output[1:]) if isinstance(output, tuple) else changed

def database():
    db = sqlite3.connect(OUT / 'scores.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS scores(model TEXT,dataset TEXT,direction TEXT,alpha REAL,example_index INTEGER,predicted TEXT,logp_positive REAL,logp_negative REAL,logp_neutral REAL,margin REAL,PRIMARY KEY(model,dataset,direction,alpha,example_index))')
    return db

def run(p, samples, args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
    from stage_d import token_batch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback.')
    db = database()
    try:
        for key in list(MODELS) if args.model == 'all' else [args.model]:
            info = p['models'][key]
            tok = AutoTokenizer.from_pretrained(MODELS[key], use_fast=True, local_files_only=True, revision=model_revision(MODELS[key]))
            if hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest() != info['tokenizer_sha256']:
                raise ValueError('Tokenizer changed.')
            tok.padding_side = 'right' if key == 'bert_large_uncased' else 'left'
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            ids = {}
            for label in ('positive', 'negative', 'neutral'):
                tokens = tok.encode(label, add_special_tokens=False)
                if len(tokens) != 1:
                    raise ValueError('Expected single-token sentiment candidate.')
                ids[label] = tokens[0]
            loader = AutoModelForMaskedLM if key == 'bert_large_uncased' else AutoModelForCausalLM
            model = loader.from_pretrained(MODELS[key], dtype=getattr(torch, FORWARD_DTYPES[key]), local_files_only=True, revision=model_revision(MODELS[key])).eval().to('cuda')
            blocks = model.bert.encoder.layer if key == 'bert_large_uncased' else model.model.layers
            if len(blocks) != info['total_layers'] or model.config.hidden_size != info['width']:
                raise ValueError('Model dimensions changed.')
            block = blocks[info['layer'] - 1]

            def score(rows, dataset, vector=None, alpha=0.0):
                inputs, mask, _, clipped = token_batch(tok, key, info['prompts'][dataset], rows)
                if any(clipped):
                    raise ValueError('No silent text truncation.')
                handle = None
                if vector is not None and alpha != 0:
                    delta = torch.as_tensor(vector * alpha, device='cuda')
                    handle = block.register_forward_hook(lambda module, arguments, output: intervene(output, mask, delta))
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
                        labels = ['negative', 'positive'] if dataset == 'sst2' else ['negative', 'neutral', 'positive']
                        lp = torch.log_softmax(logits, dim=-1)[:, [ids[x] for x in ('positive', 'negative', 'neutral')]].cpu().numpy()
                        predictions = logits[:, [ids[x] for x in labels]].argmax(dim=-1).cpu().tolist()
                        if not np.isfinite(lp).all():
                            raise ValueError('Nonfinite scores.')
                        return (lp, [labels[i] for i in predictions])
                finally:
                    if handle is not None:
                        handle.remove()
            try:
                for dataset in DATASETS:
                    values, meta = directions(p, key, dataset)
                    target = OUT / key / dataset
                    target.mkdir(parents=True, exist_ok=True)
                    atomic_json(target / 'directions_metadata.json', meta)
                    np.savez(target / 'directions.npz', **values)
                    names = [args.direction] if args.direction else list(values)
                    if any((name not in values for name in names)):
                        raise ValueError('Unknown --direction.')
                    rows = samples[dataset][:args.limit] if args.limit else samples[dataset]
                    jobs = [('_baseline', 0.0, None)] + [(name, a, values[name]) for name in names for a in ALPHAS if a != 0]
                    for name, alpha, vector in jobs:
                        done = {r[0] for r in db.execute('SELECT example_index FROM scores WHERE model=? AND dataset=? AND direction=? AND alpha=?', (key, dataset, name, alpha))}
                        pending = [i for i in range(len(rows)) if i not in done]
                        for chunk in sorted({i // args.batch_size for i in pending}):
                            start = chunk * args.batch_size
                            batch_indices = list(range(start, min(start + args.batch_size, len(samples[dataset]))))
                            lp, pred = score([samples[dataset][i] for i in batch_indices], dataset, vector, alpha)
                            indices = [i for i in batch_indices if i < len(rows) and i not in done]
                            records = [(key, dataset, name, alpha, i, pred[i - start], float(lp[i - start, 0]), float(lp[i - start, 1]), float(lp[i - start, 2]) if dataset != 'sst2' else None, float(lp[i - start, 0] - lp[i - start, 1])) for i in indices]
                            with db:
                                db.executemany('INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?)', records)
                        print(f'G {key}/{dataset}/{name} alpha={alpha:g}: {len(rows)} saved', flush=True)
                    a, _ = score(rows[:1], dataset)
                    b, _ = score(rows[:1], dataset, np.zeros(info['width'], dtype=np.float32), 1.0)
                    if not np.array_equal(a, b):
                        raise ValueError('Zero hook changes baseline.')
                    atomic_json(target / 'smoke_checks.json', dict(zero_hook_equal=True, candidate_ids=ids, physical_block_0_based=info['layer'] - 1, rows_this_run=len(rows)))
            finally:
                del model, blocks, block, tok
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        db.close()

def csv_write(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
'Fixed-layer Stage G confirmation: token-relative doses, fresh samples, paired CIs.'
REFERENCE_FRACTION = 0.1
FORWARD_DTYPES = {'0.5b': 'float16', '1.5b': 'float32', '3b': 'float16', 'bert_large_uncased': 'float16'}
CALIBRATION = {}
ACTIVE_MODEL = None
DIRECTION_CACHE = {}

def normalized_text(text):
    return ' '.join(text.lower().split())

def balanced_unique(rows, count, excluded_ids, excluded_texts, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    selected = []
    labels_by_text = {}
    for row in rows:
        labels_by_text.setdefault(normalized_text(row['text']), set()).add(row['label'])
    seen = set(excluded_texts)
    for label in (0, 1):
        candidates = []
        for i in order:
            row = rows[int(i)]
            text = normalized_text(row['text'])
            if row['label'] != label or row['id'] in excluded_ids or text in seen or (len(labels_by_text[text]) != 1):
                continue
            seen.add(text)
            candidates.append(dict(row, stage_e_row_index=int(i)))
            if len(candidates) == count:
                break
        if len(candidates) != count:
            raise ValueError(f'Only {len(candidates)} unique eligible class {label} examples; requested {count}.')
        selected += candidates
    rng.shuffle(selected)
    return selected

def prepare(args):
    global REFERENCE_FRACTION
    p = json.loads((E / 'protocol.json').read_text(encoding='utf-8'))
    assert_protocol(p)
    pilot_path = ART / 'stage_g' / 'samples.json'
    pilot = json.loads(pilot_path.read_text(encoding='utf-8'))
    samples = {}
    development = {}
    for dataset in DATASETS:
        ids = {r['id'] for r in pilot[dataset]}
        texts = {normalized_text(r['text']) for r in pilot[dataset]}
        rows, _ = read_rows(dataset, 'test')
        samples[dataset] = balanced_unique(rows, args.examples // 2, ids, texts, args.seed)
        ids |= {r['id'] for r in samples[dataset]}
        texts |= {normalized_text(r['text']) for r in samples[dataset]}
        rows, _ = read_rows(dataset, 'validation')
        development[dataset] = balanced_unique(rows, args.control_examples // 2, ids, texts, args.seed + 1)
        assert not {normalized_text(r['text']) for r in samples[dataset]} & {normalized_text(r['text']) for r in development[dataset]}
    config = dict(schema=2, stage_e_protocol_sha256=sha(E / 'protocol.json'), pilot_samples_sha256=sha(pilot_path), models=p['models'], examples=args.examples, control_examples=args.control_examples, seed=args.seed, batch_size=args.batch_size, alphas=list(ALPHAS), reference_fraction=args.reference_fraction, random_seeds=list(range(1000, 1000 + args.random_count)), bootstrap_draws=args.bootstrap, bootstrap_seed=args.seed + 2, direction_rule='unit directions; same norm at a given token/alpha within model', intervention='delta(token)=alpha*reference_fraction*norm(original post-block token residual)*unit_direction; text tokens only', prompt='template_1', layers='fixed Stage C selection; no layer re-selection', precision=FORWARD_DTYPES, labels='candidate-token argmax; full vocabulary Stage A margin; TweetEval neutral gold excluded but neutral predictions retained', sample_rule='unique normalized texts; exclude pilot IDs/texts; balanced Stage E test/proxy pool; previously observed in earlier stages, fresh to Stage G only', positive_control='1.5B development-only direct sentiment; at alpha +/-0.5 require CI(delta+)>0, CI(delta-)<0, CI(C)>0 on both datasets', samples_digest=hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(), development_digest=hashlib.sha256(json.dumps(development, sort_keys=True).encode()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, value in [('protocol.json', config), ('samples.json', samples), ('development_samples.json', development)]:
        path = args.output / filename
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Frozen confirmation settings changed; choose a new --output folder.')
        if not path.exists():
            atomic_json(path, value)
    REFERENCE_FRACTION = args.reference_fraction
    return (p, config, samples, development)

def directions(p, key, dataset):
    cache_key = (key, dataset)
    if cache_key not in DIRECTION_CACHE:
        values, meta = pilot_directions(p, key, dataset)
        values = {n: v for n, v in values.items() if not n.startswith('random_')}
        signs = {n: v for n, v in meta['expected_signs'].items() if not n.startswith('random_')}
        for seed in RANDOM_SEEDS:
            name = f'random_seed_{seed}'
            values[name] = random_directions(p['models'][key]['width'], seed, 1)[0]
            signs[name] = 0
        values = {n: np.asarray(v / np.linalg.norm(v.astype(np.float64)), dtype=np.float32) for n, v in values.items()}
        meta.update(reference_norm=1.0, expected_signs=signs, intervention_scale='relative to each pre-intervention token residual norm', reference_fraction=REFERENCE_FRACTION, forward_dtype=FORWARD_DTYPES[key], legacy_median_raw_norm=meta['reference_norm'])
        DIRECTION_CACHE[cache_key] = (values, meta)
    return DIRECTION_CACHE[cache_key]

def intervene(output, mask, delta):
    h = output[0] if isinstance(output, tuple) else output
    norms = h.float().norm(dim=-1, keepdim=True)
    offset = (norms * delta.to(h.device).float() * REFERENCE_FRACTION * mask.unsqueeze(-1)).to(h.dtype)
    changed = h + offset
    expected = float(delta.float().norm().item()) * REFERENCE_FRACTION
    if expected > 0:
        measured = (changed.float() - h.float()).norm(dim=-1)[mask] / norms.squeeze(-1)[mask].clamp_min(1e-12)
        if not measured.isfinite().all():
            raise ValueError('Nonfinite relative perturbation.')
        stats = CALIBRATION.setdefault((ACTIVE_MODEL, round(expected, 8)), dict(tokens=0, sum_ratio=0.0, min_ratio=float('inf'), max_ratio=0.0))
        stats['tokens'] += measured.numel()
        stats['sum_ratio'] += float(measured.sum().item())
        stats['min_ratio'] = min(stats['min_ratio'], float(measured.min().item()))
        stats['max_ratio'] = max(stats['max_ratio'], float(measured.max().item()))
    return (changed, *output[1:]) if isinstance(output, tuple) else changed

def execute(p, samples, args, folder, keys, direction=None):
    global OUT, ACTIVE_MODEL
    from types import SimpleNamespace
    OUT = folder
    OUT.mkdir(parents=True, exist_ok=True)
    for key in keys:
        ACTIVE_MODEL = key
        run(p, samples, SimpleNamespace(model=key, batch_size=args.batch_size, limit=None, direction=direction))
    calibration = []
    for (key, expected), stats in CALIBRATION.items():
        calibration.append(dict(model=key, expected_relative_norm=expected, mean_actual_relative_norm=stats['sum_ratio'] / stats['tokens'], min_actual=stats['min_ratio'], max_actual=stats['max_ratio'], text_tokens=stats['tokens']))
    csv_write('relative_perturbation_checks.csv', calibration)

def bootstrap_weights(labels, draws, seed):
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, len(labels)), dtype=np.float64)
    for label in (0, 1):
        indices = np.flatnonzero(np.asarray(labels) == label)
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=draws)
    return weights / len(labels)

def interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]

def read_grid(db, key, dataset, name, n):
    base = db.execute('SELECT example_index,margin,predicted FROM scores WHERE model=? AND dataset=? AND direction="_baseline" ORDER BY example_index', (key, dataset)).fetchall()
    if [r[0] for r in base] != list(range(n)):
        return None
    m = np.empty((n, 7))
    predictions = {0.0: [r[2] for r in base]}
    m[:, 3] = [r[1] for r in base]
    for j, alpha in enumerate(ALPHAS):
        if alpha == 0:
            continue
        rows = db.execute('SELECT example_index,margin,predicted FROM scores WHERE model=? AND dataset=? AND direction=? AND alpha=? ORDER BY example_index', (key, dataset, name, alpha)).fetchall()
        if [r[0] for r in rows] != list(range(n)):
            return None
        m[:, j] = [r[1] for r in rows]
        predictions[alpha] = [r[2] for r in rows]
    return (m, predictions)

def gate(args, config, development):
    folder = args.output / 'positive_control'
    path = folder / 'scores.sqlite3'
    if not path.exists():
        raise RuntimeError('Run --phase control first; 1.5B gate not evaluated.')
    db = sqlite3.connect(path)
    results = []
    for dataset in DATASETS:
        rows = development[dataset]
        grid = read_grid(db, '1.5b', dataset, 'direct_sentiment', len(rows))
        if grid is None:
            raise RuntimeError('1.5B positive control incomplete.')
        m, _ = grid
        w = bootstrap_weights([r['label'] for r in rows], args.bootstrap, args.seed + 2)
        plus = m[:, -1] - m[:, 3]
        minus = m[:, 0] - m[:, 3]
        c = (m[:, -1] - m[:, 0]) / 2
        cp = interval(w @ plus)
        cm = interval(w @ minus)
        cc = interval(w @ c)
        passed = cp[0] > 0 and cm[1] < 0 and (cc[0] > 0)
        results.append(dict(dataset=dataset, n=len(rows), alpha=0.5, mean_delta_plus=float(plus.mean()), delta_plus_ci=cp, mean_delta_minus=float(minus.mean()), delta_minus_ci=cm, mean_C=float(c.mean()), C_ci=cc, passed=passed))
    db.close()
    result = dict(complete=True, passed=all((r['passed'] for r in results)), development_only=True, protocol_sha256=sha(args.output / 'protocol.json'), datasets=results)
    atomic_json(folder / 'gate.json', result)
    lines = [f'1.5B direct sentiment control passed={result['passed']}; fixed layer 27, float32, 5% maximum text-token perturbation.']
    lines += [f'{r['dataset']}: delta(+.5)={r['mean_delta_plus']:+.5f} CI={r['delta_plus_ci']}; delta(-.5)={r['mean_delta_minus']:+.5f} CI={r['delta_minus_ci']}; C={r['mean_C']:+.5f} CI={r['C_ci']}; passed={r['passed']}' for r in results]
    (folder / 'positive_control_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result

def confirmation_report(args, config, samples):
    global OUT
    OUT = args.output
    if not (OUT / 'scores.sqlite3').exists():
        raise RuntimeError('No confirmation scores yet.')
    db = sqlite3.connect(OUT / 'scores.sqlite3')
    delta_rows = []
    c_rows = []
    mono_rows = []
    random_rows = []
    states = {}
    for key in MODELS:
        for dataset in DATASETS:
            rows = samples[dataset]
            n = len(rows)
            path = OUT / key / dataset / 'directions_metadata.json'
            if not path.exists():
                states[f'{key}/{dataset}'] = 'not run'
                continue
            meta = json.loads(path.read_text())
            w = bootstrap_weights([r['label'] for r in rows], args.bootstrap, args.seed + 2)
            grids = {name: read_grid(db, key, dataset, name, n) for name in meta['expected_signs']}
            states[f'{key}/{dataset}'] = 'complete' if all((v is not None for v in grids.values())) else 'partial'
            random = {a: [] for a in (0.1, 0.25, 0.5)}
            for name, grid in grids.items():
                if grid is None:
                    continue
                m, predictions = grid
                sign = meta['expected_signs'][name]
                delta = m - m[:, [3]]
                boot_delta = w @ delta
                for j, alpha in enumerate(ALPHAS):
                    ci = interval(boot_delta[:, j])
                    flips = np.mean(np.array(predictions[alpha]) != np.array(predictions[0.0]))
                    delta_rows.append(dict(model=key, dataset=dataset, direction=name, alpha=alpha, n=n, mean_delta_margin=float(delta[:, j].mean()), ci_low=ci[0], ci_high=ci[1], label_flip_rate=float(flips)))
                for a in (0.1, 0.25, 0.5):
                    positive = ALPHAS.index(a)
                    negative = ALPHAS.index(-a)
                    c = (m[:, positive] - m[:, negative]) / 2
                    bc = w @ c
                    ci = interval(bc)
                    c_rows.append(dict(model=key, dataset=dataset, direction=name, alpha=a, n=n, expected_sign=sign, mean_C=float(c.mean()), ci_low=ci[0], ci_high=ci[1]))
                    if name.startswith('random_'):
                        random[a].append((float(c.mean()), bc))
                mono_rows.append(dict(model=key, dataset=dataset, direction=name, n=n, expected_sign=sign, mean_slope=float(np.mean(m @ np.asarray(ALPHAS) / np.dot(ALPHAS, ALPHAS))), fraction_expected_monotone=float(np.mean(np.all(np.diff(m, axis=1) * sign >= -1e-05, axis=1))) if sign else ''))
            for row in [r for r in c_rows if r['model'] == key and r['dataset'] == dataset and (r['expected_sign'] != 0)]:
                controls = random[row['alpha']]
                if len(controls) != args.random_count:
                    continue
                grid = grids[row['direction']][0]
                a = row['alpha']
                c = (grid[:, ALPHAS.index(a)] - grid[:, ALPHAS.index(-a)]) / 2
                max_control = float(np.max(np.abs([v[0] for v in controls])))
                boot_max = np.max(np.abs(np.stack([v[1] for v in controls], axis=1)), axis=1)
                excess_ci = interval(row['expected_sign'] * (w @ c) - boot_max)
                random_rows.append(dict(model=key, dataset=dataset, direction=row['direction'], alpha=a, random_directions=len(controls), signed_C=row['expected_sign'] * row['mean_C'], max_abs_random_C=max_control, excess_ci_low=excess_ci[0], excess_ci_high=excess_ci[1]))
    db.close()
    csv_write('delta_margin_paired_ci.csv', delta_rows)
    csv_write('two_sided_response_paired_ci.csv', c_rows)
    csv_write('monotonicity.csv', mono_rows)
    csv_write('random_comparison_paired_ci.csv', random_rows)
    control_path = OUT / 'positive_control' / 'gate.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    excluded = ['1.5b'] if control and (not control['passed']) else []
    complete = all((v == 'complete' for k, v in states.items() if k.split('/')[0] not in excluded)) and all((f'{k}/{d}' in states for k in MODELS if k not in excluded for d in DATASETS))
    atomic_json(OUT / 'state.json', dict(complete_for_eligible_models=complete, excluded_models=excluded, model_datasets=states, protocol_sha256=sha(OUT / 'protocol.json')))
    lines = [f'Stage G confirmation complete for eligible models={complete}; excluded after failed development control={excluded}.', 'Fixed layers/T1; each text-token residual receives a 1%, 2.5%, or 5% norm-relative perturbation; unit directions norm matched.', 'Fresh to Stage G: pilot IDs/normalized texts excluded. Earlier-stage test/proxy observations remain; not an untouched benchmark.', 'Paired class-stratified bootstrap CIs across examples; random comparison conditional on the 50 fixed sampled directions; no multiple-testing correction.', 'Only fully scored directions appear in CI tables. Label flips are candidate-token flips, not free-generation outcomes.']
    lines += [f'{r['model']} {r['dataset']} {r['direction']} alpha={r['alpha']}: C={r['mean_C']:+.5f} CI=[{r['ci_low']:+.5f},{r['ci_high']:+.5f}]' for r in c_rows if r['direction'] in ('valence_raw', 'valence_denoised', 'direct_sentiment') and r['alpha'] == 0.5]
    (OUT / 'stage_g_confirmation_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)

def self_test():
    import torch
    h = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]])
    mask = torch.tensor([[True, False]])
    delta = torch.tensor([0.25, 0.0, 0.0])
    changed = intervene(h, mask, delta)
    assert torch.equal(changed[~mask], h[~mask])
    assert torch.allclose((changed - h)[mask].norm(dim=-1) / h[mask].norm(dim=-1), torch.tensor([0.025]), atol=1e-06)
    assert torch.equal(intervene(h, mask, delta * 0), h)
    rows = [dict(id=str(i), text='same' if i < 2 else str(i), label=i % 2) for i in range(20)]
    sample = balanced_unique(rows, 3, {'3'}, {'4'}, 42)
    assert len(sample) == 6 and len({normalized_text(r['text']) for r in sample}) == 6
    assert all((r['id'] != '3' and r['text'] not in ('same', '4') for r in sample))
    labels = np.array([0] * 10 + [1] * 10)
    w = bootstrap_weights(labels, 200, 42)
    assert np.allclose(w.sum(axis=1), 1) and np.allclose(w[:, :10].sum(axis=1), 0.5)
    assert np.allclose(w @ np.ones(20), 1)
    print('Token-relative norm, masking, zero hook, disjoint sampling and paired stratified bootstrap tests passed.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'control', 'smoke', 'run', 'report', 'all'], default='prepare')
    parser.add_argument('--model', choices=[*MODELS, 'all'], default='all')
    parser.add_argument('--examples', type=int, default=600)
    parser.add_argument('--control-examples', type=int, default=100)
    parser.add_argument('--random-count', type=int, default=50)
    parser.add_argument('--reference-fraction', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--output', type=Path, default=ART / 'stage_g_confirmation')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.examples < 2 or args.examples % 2 or args.control_examples < 2 or args.control_examples % 2 or (args.random_count < 50) or (args.batch_size < 1) or (args.bootstrap < 100) or (args.reference_fraction <= 0):
        parser.error('Use even sample sizes, 50+ random controls, positive scale/counts and 100+ bootstrap draws.')
    args.output = args.output.resolve()
    global RANDOM_SEEDS
    RANDOM_SEEDS = list(range(1000, 1000 + args.random_count))
    p, config, samples, development = prepare(args)
    if args.phase in ('control', 'all'):
        execute(p, development, args, args.output / 'positive_control', ['1.5b'], 'direct_sentiment')
        gate(args, config, development)
    if args.phase == 'smoke':
        small = {d: rows[:8] for d, rows in development.items()}
        execute(p, small, args, args.output / 'smoke', list(MODELS) if args.model == 'all' else [args.model], 'valence_denoised')
    if args.phase in ('run', 'all'):
        control = gate(args, config, development)
        keys = list(MODELS) if args.model == 'all' else [args.model]
        if not control['passed'] and '1.5b' in keys:
            if args.model == '1.5b':
                raise RuntimeError('1.5B direct-sentiment control failed; full confirmation disabled for this model.')
            keys.remove('1.5b')
            print('Skipping 1.5B: failed direct-sentiment positive control. Fixed layer unchanged.', flush=True)
        execute(p, samples, args, args.output, keys)
    if args.phase in ('report', 'all'):
        confirmation_report(args, config, samples)
    print(f'Confirmation protocol ready: {args.examples} unique examples/dataset; {19 + args.random_count} directions; {args.random_count} random controls.', flush=True)
if __name__ == '__main__':
    main()
