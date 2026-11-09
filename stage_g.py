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
OUT = ART / 'stage_g'
ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)

def setup(args):
    p = json.loads((E / 'protocol.json').read_text(encoding='utf-8'))
    assert_protocol(p)
    samples = {}
    for dataset in DATASETS:
        rows, entry = read_rows(dataset, 'test')
        rng = np.random.default_rng(args.seed)
        selected = []
        for label in (0, 1):
            indices = [i for i, r in enumerate(rows) if r['label'] == label]
            if len(indices) < args.per_class:
                raise ValueError('Sample exceeds available class.')
            selected += rng.choice(indices, args.per_class, replace=False).tolist()
        rng.shuffle(selected)
        samples[dataset] = [dict(rows[i], stage_e_test_row_index=i) for i in selected]
    config = dict(stage_e_protocol_sha256=sha(E / 'protocol.json'), models=p['models'], per_class=args.per_class, seed=args.seed, batch_size=args.batch_size, alphas=list(ALPHAS), prompt='template_1', sample_digest=hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(), intervention='post selected block; all valid raw-text tokens; exclude prompt/padding/special tokens', norm='all directions scaled to median raw Stage B emotion norm at selected layer, within each model', labels='candidate-token argmax, not Stage A free-generation predictions; TweetEval candidates include neutral', rows='balanced Stage E test/proxy samples; TweetEval neutral gold rows excluded; earlier observed data, exploratory evaluation', direction_types='8 raw emotions, 8 denoised emotions, raw/denoised valence, train-only direct sentiment, 10 random seeds')
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, value in [('protocol.json', config), ('samples.json', samples)]:
        path = OUT / filename
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Frozen Stage G inputs changed.')
        if not path.exists():
            atomic_json(path, value)
    return (p, config, samples)

def directions(p, key, dataset):
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
            model = loader.from_pretrained(MODELS[key], dtype=torch.float16, local_files_only=True, revision=model_revision(MODELS[key])).eval().to('cuda')
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

def report(samples):
    db = database()
    db.row_factory = sqlite3.Row
    summary = []
    monotonic = []
    complete = {}
    try:
        for key in MODELS:
            for dataset in DATASETS:
                path = OUT / key / dataset / 'directions_metadata.json'
                complete[f'{key}/{dataset}'] = False
                if not path.exists():
                    continue
                meta = json.loads(path.read_text())
                n = len(samples[dataset])
                base = {r['example_index']: dict(r) for r in db.execute('SELECT * FROM scores WHERE model=? AND dataset=? AND direction=?', (key, dataset, '_baseline'))}
                all_done = len(base) == n
                for name, sign in meta['expected_signs'].items():
                    values = {0.0: base}
                    for alpha in ALPHAS:
                        if alpha != 0:
                            values[alpha] = {r['example_index']: dict(r) for r in db.execute('SELECT * FROM scores WHERE model=? AND dataset=? AND direction=? AND alpha=?', (key, dataset, name, alpha))}
                    if any((set(v) != set(range(n)) for v in values.values())):
                        all_done = False
                        continue
                    margins = np.array([[values[a][i]['margin'] for a in ALPHAS] for i in range(n)])
                    for j, alpha in enumerate(ALPHAS):
                        delta = margins[:, j] - margins[:, 3]
                        flips = [values[alpha][i]['predicted'] != base[i]['predicted'] for i in range(n)]
                        accuracy = np.mean([values[alpha][i]['predicted'] == ('positive' if samples[dataset][i]['label'] else 'negative') for i in range(n)])
                        summary.append(dict(model=key, dataset=dataset, direction=name, alpha=alpha, n=n, expected_sign=sign, mean_delta_margin=float(delta.mean()), median_delta_margin=float(np.median(delta)), label_flip_rate=float(np.mean(flips)), candidate_accuracy=float(accuracy)))
                    slopes = margins @ np.array(ALPHAS) / np.dot(ALPHAS, ALPHAS)
                    monotonic.append(dict(model=key, dataset=dataset, direction=name, n=n, expected_sign=sign, mean_slope=float(slopes.mean()), fraction_expected_monotone=float(np.mean(np.all(np.diff(margins, axis=1) * sign >= -1e-05, axis=1))) if sign else '', fraction_positive_slope=float(np.mean(slopes > 0)), fraction_negative_slope=float(np.mean(slopes < 0))))
                complete[f'{key}/{dataset}'] = all_done
    finally:
        db.close()
    csv_write('steering_summary.csv', summary)
    csv_write('monotonicity.csv', monotonic)
    controls = []
    for r in summary:
        if r['direction'].startswith('random_') or r['alpha'] == 0:
            continue
        random = [x['mean_delta_margin'] for x in summary if x['model'] == r['model'] and x['dataset'] == r['dataset'] and (x['alpha'] == r['alpha']) and x['direction'].startswith('random_')]
        if len(random) == 10:
            controls.append(dict(model=r['model'], dataset=r['dataset'], direction=r['direction'], alpha=r['alpha'], mean_delta_margin=r['mean_delta_margin'], random_mean=float(np.mean(random)), random_seed_sd=float(np.std(random, ddof=1)), excess_absolute_effect=abs(r['mean_delta_margin']) - float(np.mean(np.abs(random)))))
    csv_write('random_comparison.csv', controls)
    finished = all(complete.values())
    atomic_json(OUT / 'state.json', dict(complete=finished, model_datasets=complete))
    lines = [f'Stage G complete={finished}; summaries only include fully evaluated direction/strength/example grids.', 'Fixed T1; frozen Stage C layers; text-token steering; all directions norm matched.', 'Candidate-token label flips; Qwen free-generation invalid-output behavior is not measured.', 'TweetEval neutral gold excluded, neutral predictions allowed. Previously observed test/proxy sample; exploratory.', 'Random SD is variation across seeds, not an example-level confidence interval.', 'Steering effects establish sufficiency under these interventions, not unique emotion mediation.']
    lines += [f'{r['model']} {r['dataset']} {r['direction']} alpha=+1: delta M={r['mean_delta_margin']:+.4f}, flips={100 * r['label_flip_rate']:.2f}%' for r in summary if r['alpha'] == 1 and (not r['direction'].startswith('random_'))]
    (OUT / 'stage_g_results_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Stage G complete={finished}; results: {OUT}')

def test():
    import torch
    h = torch.ones(2, 3, 4)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    delta = torch.arange(4.0)
    changed = intervene(h, mask, delta)
    assert torch.equal(changed[~mask], h[~mask]) and torch.equal(changed[mask], h[mask] + delta)
    second = torch.tensor([2.0])
    out = intervene((h, second), mask, delta)
    assert out[1] is second and torch.equal(out[0], changed)
    assert torch.equal(intervene(h, mask, delta * 0), h)
    print('Masking, output-structure and zero-intervention checks passed.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run', 'report', 'all'], default='prepare')
    parser.add_argument('--model', choices=[*MODELS, 'all'], default='all')
    parser.add_argument('--per-class', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, help='Smoke only: first N examples per dataset.')
    parser.add_argument('--direction', help='Smoke/targeted filter, e.g. emotion_denoised:joy')
    parser.add_argument('--output', type=Path, default=ART / 'stage_g', help='Use a separate folder for a different sample size/configuration.')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        test()
        return
    if min(args.per_class, args.batch_size) < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Counts must be positive.')
    global OUT
    OUT = args.output.resolve()
    if args.limit:
        OUT = OUT.with_name(OUT.name + '_smoke')
    p, config, samples = setup(args)
    if args.phase in ('run', 'all'):
        run(p, samples, args)
    if args.phase in ('report', 'all'):
        report(samples)
    print(f'Samples: {args.per_class} per gold class/dataset. 29 directions; seven fixed strengths; resumable.')
if __name__ == '__main__':
    main()
