from __future__ import annotations
'Stage D: observe emotion directions during sentiment classification; no intervention.'
from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import argparse
import csv
import gc
import hashlib
import json
import os
import sqlite3
from pathlib import Path
ROOT = WORKSPACE
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import numpy as np
import torch
import transformers
import nnsight
from datasets import load_from_disk
from nnsight import NNsight
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from build_stage_b_vectors import EMOTIONS, extract_pooled as extract_qwen
from build_stage_b_bert_vectors import extract_pooled as extract_bert
from build_stage_b_final_vectors import MODEL_IDS
from prompt_templates import PROMPT_TEMPLATES, QWEN3B_SST2_T4_PROMPT
from bert_stage_a import TEMPLATES as BERT_TEMPLATES
OUT = ROOT / 'artifacts' / 'stage_d'
LABELS = {'sst2': ('negative', 'positive'), 'tweet_eval_sentiment': ('negative', 'neutral', 'positive')}
DB_MODELS = {'0.5b': 'qwen2.5-0.5b', '1.5b': 'qwen2.5-1.5b', '3b': 'qwen2.5-3b', 'bert_large_uncased': 'bert-large-uncased'}

def sha(path):
    path = source_path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def frozen_rows():
    """Bind text and source provenance to the completed four-model Stage A report."""
    source = ROOT / 'artifacts' / 'stage_a_four_models' / 'scores.sqlite3'
    if not source.exists():
        raise FileNotFoundError('Transfer artifacts/stage_a_four_models/scores.sqlite3 first.')
    conn = sqlite3.connect(f'file:{source.as_posix()}?mode=ro', uri=True)
    try:
        records = conn.execute("SELECT dataset,row_index,source_split,source_row_index,gold_label\n            FROM scores WHERE model=? AND template_id='template_1'\n            ORDER BY dataset,row_index", (DB_MODELS['0.5b'],)).fetchall()
    finally:
        conn.close()
    rows = []
    for dataset, count in (('sst2', 12872), ('tweet_eval_sentiment', 12284)):
        path = ROOT / 'artifacts' / 'datasets' / dataset
        if not path.exists():
            raise FileNotFoundError(f'Transfer the frozen dataset folder first: {path}')
        data = load_from_disk(str(path))
        part = [r for r in records if r[0] == dataset]
        if len(part) != count or [r[1] for r in part] != list(range(count)):
            raise ValueError(f'Stage A row count/order mismatch: {dataset}')
        for _, index, split, source_index, gold in part:
            item = data[split][source_index]
            if LABELS[dataset][item['label']] != gold:
                raise ValueError('Stage A gold label disagrees with source dataset.')
            text = item['sentence' if dataset == 'sst2' else 'text']
            rows.append(dict(dataset=dataset, row_index=index, source_split=split, source_row_index=source_index, gold_label=gold, text=text))
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = ''.join((json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
    path = OUT / 'examples.jsonl'
    if path.exists() and path.read_text(encoding='utf-8') != encoded:
        raise ValueError('Frozen Stage D examples changed; refusing to mix results.')
    if not path.exists():
        path.write_text(encoded, encoding='utf-8')
    return rows

def prompt_for(key, dataset, template):
    if key == 'bert_large_uncased':
        return BERT_TEMPLATES[template]
    if key == '3b' and dataset == 'sst2' and (template == 'template_4'):
        return QWEN3B_SST2_T4_PROMPT
    return PROMPT_TEMPLATES[dataset][template]

def token_batch(tokenizer, key, pattern, rows):
    """Locate the raw text span in the full Stage A prompt; mask prompt/special/pad tokens."""
    marker = '__STAGE_D_TEXT_SPAN_7ef8492c__'
    strings, spans, lengths = ([], [], [])
    for row in rows:
        text = row['text']
        skeleton = pattern.format(text=marker)
        if key != 'bert_large_uncased':
            skeleton = tokenizer.apply_chat_template([{'role': 'user', 'content': skeleton}], tokenize=False, add_generation_prompt=True)
        prefix, suffix = skeleton.split(marker)
        rendered = prefix + text + suffix
        strings.append(rendered)
        spans.append((len(prefix), len(prefix) + len(text)))
        lengths.append(len(tokenizer(rendered, add_special_tokens=key == 'bert_large_uncased')['input_ids']))
    max_length = 256 if key == 'bert_large_uncased' else 1024
    batch = tokenizer(strings, padding=True, truncation=True, max_length=max_length, add_special_tokens=key == 'bert_large_uncased', return_offsets_mapping=True, return_tensors='pt')
    offsets = batch.pop('offset_mapping')
    mask = torch.zeros_like(batch['attention_mask'], dtype=torch.bool)
    for i, (start, end) in enumerate(spans):
        a, b = (offsets[i, :, 0], offsets[i, :, 1])
        mask[i] = (a < end) & (b > start) & (b > a)
    mask &= batch['attention_mask'].bool()
    for token_id in tokenizer.all_special_ids:
        mask &= batch['input_ids'].ne(token_id)
    counts = mask.sum(dim=1).tolist()
    if min(counts) == 0:
        raise ValueError('A prompted example has no valid text tokens after truncation.')
    return (batch.to('cuda'), mask.to('cuda'), counts, [n > max_length for n in lengths])

def vectors(key):
    folder = ROOT / 'artifacts' / 'stage_b_final_vectors' / key
    path = folder / 'emotion_directions_final.float32.npy'
    meta = json.loads((folder / 'final_vectors_metadata.json').read_text(encoding='utf-8'))
    values = np.load(path).astype(np.float32)
    expected = (8, meta['layers'], meta['hidden_size'])
    if not meta.get('complete') or meta['model_id'] != MODEL_IDS[key] or tuple(meta['emotion_order']) != tuple(EMOTIONS) or (values.shape != expected):
        raise ValueError(f'Final vector metadata mismatch: {key}')
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError('Non-finite or zero emotion vectors.')
    return (values / norms, sha(path))

def run(key, all_rows, args):
    unit, vector_hash = vectors(key)
    layers = unit.shape[1]
    model = wrapped = tokenizer = None
    try:
        for template in args.templates:
            for dataset in LABELS:
                rows = [r for r in all_rows if r['dataset'] == dataset]
                folder = OUT / key / template / dataset
                folder.mkdir(parents=True, exist_ok=True)
                pattern = prompt_for(key, dataset, template)
                signature = dict(model_id=MODEL_IDS[key], source_sha256=sha(OUT / 'examples.jsonl'), vector_sha256=vector_hash, prompt=pattern, template=template, dataset=dataset, layers=layers, examples=len(rows), score='cosine(mean(text-token block outputs),final emotion vector)', text_mask='tokens overlapping raw-text span; exclude special and padding', max_length=256 if key == 'bert_large_uncased' else 1024, model_dtype='float16', nnsight=nnsight.__version__, torch=torch.__version__, transformers=transformers.__version__)
                settings = folder / 'signature.json'
                if settings.exists() and json.loads(settings.read_text(encoding='utf-8')) != signature:
                    raise ValueError(f'Checkpoint signature changed: {folder}')
                write_json(settings, signature)
                conn = sqlite3.connect(folder / 'progress.sqlite3')
                try:
                    conn.execute('CREATE TABLE IF NOT EXISTS completed(row_index INTEGER PRIMARY KEY, text_tokens INTEGER, truncated INTEGER)')
                    done = {r[0] for r in conn.execute('SELECT row_index FROM completed')}
                    path = folder / 'cosine_scores.float32.npy'
                    if path.exists():
                        scores = np.load(path, mmap_mode='r+')
                        if scores.shape != (len(rows), layers, 8) or scores.dtype != np.float32:
                            raise ValueError('Score shape/dtype mismatch.')
                    else:
                        if done:
                            raise ValueError('Completed checkpoint rows have no score array.')
                        scores = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), layers, 8))
                    limit = min(args.limit, len(rows)) if args.limit else len(rows)
                    pending = [i for i in range(limit) if i not in done]
                    print(f'{key}/{template}/{dataset}: {len(done)}/{len(rows)} saved; {len(pending)} pending', flush=True)
                    if pending and model is None:
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[key], use_fast=True, local_files_only=True, revision=model_revision(MODEL_IDS[key]))
                        tokenizer.padding_side = 'right' if key == 'bert_large_uncased' else 'left'
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        loader = AutoModel if key == 'bert_large_uncased' else AutoModelForCausalLM
                        model = loader.from_pretrained(MODEL_IDS[key], dtype=torch.float16, local_files_only=True, revision=model_revision(MODEL_IDS[key])).eval().to('cuda')
                        actual_layers = len(model.encoder.layer) if key == 'bert_large_uncased' else len(model.model.layers)
                        if actual_layers != layers or model.config.hidden_size != unit.shape[2]:
                            raise ValueError('Loaded model dimensions differ from vectors.')
                        wrapped = NNsight(model)
                    for start in range(0, len(pending), args.batch_size):
                        indices = pending[start:start + args.batch_size]
                        inputs, mask, counts, clipped = token_batch(tokenizer, key, pattern, [rows[i] for i in indices])
                        extractor = extract_bert if key == 'bert_large_uncased' else extract_qwen
                        h = extractor(wrapped, inputs, mask, layers).astype(np.float32)
                        norms = np.linalg.norm(h, axis=-1, keepdims=True)
                        if np.any(norms <= 0):
                            raise ValueError('Zero pooled activation norm.')
                        values = np.einsum('blh,elh->ble', h / norms, unit)
                        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0001:
                            raise ValueError('Invalid cosine scores.')
                        scores[indices] = values
                        scores.flush()
                        with conn:
                            conn.executemany('INSERT INTO completed VALUES (?,?,?)', [(i, c, int(t)) for i, c, t in zip(indices, counts, clipped)])
                        if start % (args.batch_size * 25) == 0 or start + len(indices) == len(pending):
                            print(f'  saved {len(done) + start + len(indices)}/{len(rows)}', flush=True)
                    total, truncated = conn.execute('SELECT COUNT(*),COALESCE(SUM(truncated),0) FROM completed').fetchone()
                    write_json(folder / 'results_metadata.json', dict(**signature, complete=total == len(rows), completed_rows=total, truncated_examples=truncated, emotion_order=EMOTIONS))
                    del scores
                finally:
                    conn.close()
    finally:
        del wrapped, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

def effect(a, b):
    """Cohen's d, group A minus group B, with pooled within-group SD."""
    va, vb = (a.var(axis=0, ddof=1), b.var(axis=0, ddof=1))
    sd = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return np.divide(a.mean(axis=0) - b.mean(axis=0), sd, out=np.full_like(sd, np.nan), where=sd > 0)

def analyze(all_rows, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    selection = json.loads((ROOT / 'artifacts' / 'stage_c_empathetic_dialogues' / 'selected_intervention_layers.json').read_text(encoding='utf-8'))['models']
    records, summaries, descriptions = ([], [], [])
    for key in MODEL_IDS:
        if key not in selection:
            continue
        chosen = selection[key]['selected']['layer_1_based'] - 1
        for template in args.templates:
            for dataset, classes in LABELS.items():
                folder = OUT / key / template / dataset
                meta_path = folder / 'results_metadata.json'
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                if not meta['complete']:
                    continue
                scores = np.load(folder / 'cosine_scores.float32.npy', mmap_mode='r')
                rows = [r for r in all_rows if r['dataset'] == dataset]
                groups = [('combined', np.ones(len(rows), dtype=bool))]
                if dataset == 'sst2':
                    groups += [(s, np.array([r['source_split'] == s for r in rows])) for s in ('train', 'validation')]
                for split, keep in groups:
                    values = np.asarray(scores[keep], dtype=np.float64)
                    gold = np.array([r['gold_label'] for i, r in enumerate(rows) if keep[i]])
                    figdir = ROOT / 'visualizations' / 'stage_d' / key / template / dataset / split
                    figdir.mkdir(parents=True, exist_ok=True)
                    for label in classes:
                        x = values[gold == label]
                        for layer in range(values.shape[1]):
                            for e, emotion in enumerate(EMOTIONS):
                                descriptions.append((key, template, dataset, split, label, layer + 1, emotion, len(x), x[:, layer, e].mean(), x[:, layer, e].std(ddof=1)))
                    contrasts = [('positive', 'negative')]
                    if 'neutral' in classes:
                        contrasts += [('positive', 'neutral'), ('neutral', 'negative')]
                    for label_a, label_b in contrasts:
                        a, b = (values[gold == label_a], values[gold == label_b])
                        d = effect(a, b)
                        rng = np.random.default_rng(args.seed)
                        boot = np.empty((args.bootstrap, 8))
                        aa, bb = (a[:, chosen, :], b[:, chosen, :])
                        for j in range(args.bootstrap):
                            boot[j] = effect(aa[rng.integers(len(aa), size=len(aa))], bb[rng.integers(len(bb), size=len(bb))])
                        low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
                        for layer in range(values.shape[1]):
                            for e, emotion in enumerate(EMOTIONS):
                                records.append((key, template, dataset, split, label_a, label_b, layer + 1, emotion, len(a), len(b), d[layer, e]))
                        for e, emotion in enumerate(EMOTIONS):
                            summaries.append((key, template, dataset, split, chosen + 1, emotion, label_a, label_b, d[chosen, e], low[e], high[e]))
                        fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
                        vmax = max(0.1, float(np.nanmax(np.abs(d))))
                        im = ax.imshow(d.T, aspect='auto', origin='upper', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                        ax.set_yticks(range(8), EMOTIONS)
                        ticks = np.unique(np.linspace(0, d.shape[0] - 1, 7).astype(int))
                        ax.set_xticks(ticks, ticks + 1)
                        ax.axvline(chosen, color='black', linestyle='--', linewidth=1)
                        ax.set_xlabel('Layer (dashed: Stage C selection)')
                        ax.set_title(f'{key} / {template} / {dataset} / {split}: {label_a} minus {label_b}')
                        fig.colorbar(im, ax=ax, label="Cohen's d")
                        for ext in ('png', 'pdf'):
                            fig.savefig(figdir / f'effect_{label_a}_vs_{label_b}.{ext}', dpi=300)
                        plt.close(fig)
                    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
                    colors = {'negative': '#D55E00', 'neutral': '#777777', 'positive': '#0072B2'}
                    for e, ax in enumerate(axes.flat):
                        for label in classes:
                            ax.hist(values[gold == label, chosen, e], bins=35, density=True, histtype='step', linewidth=1.5, color=colors[label], label=label)
                        ax.set_title(EMOTIONS[e])
                        ax.set_xlabel('Cosine similarity')
                    axes.flat[0].legend(frameon=False)
                    fig.suptitle(f'{key} / {template} / {dataset} / {split} / layer {chosen + 1}')
                    for ext in ('png', 'pdf'):
                        fig.savefig(figdir / f'selected_layer_distributions.{ext}', dpi=300)
                    plt.close(fig)
    tables = [('layer_effect_sizes.csv', ('model', 'template', 'dataset', 'split', 'class_a', 'class_b', 'layer', 'emotion', 'n_a', 'n_b', 'cohens_d'), records), ('selected_layer_bootstrap.csv', ('model', 'template', 'dataset', 'split', 'layer', 'emotion', 'class_a', 'class_b', 'cohens_d', 'ci95_low', 'ci95_high'), summaries), ('class_score_summary.csv', ('model', 'template', 'dataset', 'split', 'gold_label', 'layer', 'emotion', 'n', 'mean_cosine', 'sd_cosine'), descriptions)]
    for filename, header, data in tables:
        with (OUT / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
    lines = ['STAGE D — OBSERVATIONAL SENTIMENT EMOTION PROFILES', 'Cosine scores: text-token mean activation vs existing final emotion vectors.', 'All layers analyzed descriptively; 95% percentile bootstrap CIs at Stage C-selected layers only.', f'Bootstrap replicates: {args.bootstrap}; seed: {args.seed}.', 'SST-2 pooled analysis includes training examples; train/validation also reported separately.', 'CIs are pointwise, with no multiple-comparison correction. No causal claim or intervention.', 'Detailed results: class_score_summary.csv, layer_effect_sizes.csv, selected_layer_bootstrap.csv.', 'Plots: visualizations/stage_d/.', '']
    for r in summaries:
        key, template, dataset, split, layer, emotion, ca, cb, d, low, high = r
        lines.append(f'{key} {template} {dataset}/{split} L{layer} {emotion} {ca}-{cb}: d={d:+.3f} [{low:+.3f},{high:+.3f}]')
    (OUT / 'stage_d_results.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved Stage D results: {OUT}\nPlots: {ROOT / 'visualizations' / 'stage_d'}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=[*MODEL_IDS, 'qwen', 'all'], default='qwen')
    parser.add_argument('--templates', nargs='+', choices=list(BERT_TEMPLATES), default=['template_1'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, help='First N examples per dataset/template; resumable smoke test.')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap < 100 or (args.limit is not None and args.limit < 1):
        parser.error('Use batch-size >= 1, bootstrap >= 100, and limit >= 1.')
    rows = frozen_rows()
    if not args.analyze_only:
        if not torch.cuda.is_available():
            parser.error('CUDA is required for extraction.')
        print(f'CUDA: {torch.cuda.get_device_name(0)}', flush=True)
        keys = list(MODEL_IDS) if args.model == 'all' else [k for k in MODEL_IDS if k != 'bert_large_uncased'] if args.model == 'qwen' else [args.model]
        for key in keys:
            run(key, rows, args)
    if args.limit is None:
        analyze(rows, args)
if __name__ == '__main__':
    main()
