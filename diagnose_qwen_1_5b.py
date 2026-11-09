"""Audit Qwen 1.5B and run isolated all-layer/fresh-neutral diagnostics."""
from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
import time
from datetime import datetime
import numpy as np
from stage_e_common import ROOT, ART, E, MODELS, DATASETS, atomic_json, sha
from stage_e_pipeline import assert_protocol, load_hidden, emotion_inputs
from stage_f import geometry, cosine
KEY = '1.5b'
OUT = ART / 'stage_f_1_5b_diagnostics'

def inputs():
    p = json.loads((E / 'protocol.json').read_text(encoding='utf-8'))
    assert_protocol(p)
    _, info = emotion_inputs(KEY)
    raw = np.load(ART / 'stage_b_raw_vectors' / KEY / 'emotion_directions_raw.float32.npy')
    final = np.load(ART / 'stage_b_final_vectors' / KEY / 'emotion_directions_final.float32.npy')
    assert raw.shape == final.shape == (8, 28, 1536)
    return (p, info, raw, final)

def audit():
    p, info, raw, final = inputs()
    folder = ART / 'stage_b_final_vectors' / KEY
    details = []
    for l in range(28):
        components = np.asarray(np.load(folder / 'pca_components' / f'layer_{l + 1:02d}.float32.npy'), dtype=np.float64)
        r, f = (raw[:, l].astype(np.float64), final[:, l].astype(np.float64))
        expected = r - r @ components.T @ components
        error = float(np.max(np.abs(f - expected)))
        if not np.allclose(f, expected, atol=2e-05, rtol=2e-05):
            raise ValueError(f'PCA reconstruction mismatch at layer {l + 1}: {error}')
        orth_error = float(np.max(np.abs(components @ components.T - np.eye(len(components)))))
        if orth_error > 2e-05:
            raise ValueError('Nonorthogonal PCA components.')
        stats = json.loads((folder / f'layer_{l + 1:02d}_pca.json').read_text())
        for e in range(8):
            details.append(dict(layer=l + 1, emotion=p['emotion_order'][e], neutral_pcs=len(components), variance_explained=stats['explained_variance'], raw_denoised_cosine=cosine(r[e], f[e]), norm_fraction_removed=float(1 - np.linalg.norm(f[e]) / np.linalg.norm(r[e])), reconstruction_max_error=error, pc_orthogonality_max_error=orth_error))
    neutral = np.load(ART / 'stage_b_neutral_activations' / KEY / 'pooled.float16.npy', mmap_mode='r')
    layer = info['layer'] - 1
    xmean = np.zeros(1536, dtype=np.float64)
    for start in range(0, len(neutral), 128):
        xmean += np.asarray(neutral[start:start + 128, layer], dtype=np.float64).sum(axis=0)
    xmean /= len(neutral)
    saved_mean = np.load(folder / 'neutral_pca_means.float32.npy')[layer]
    pc = np.asarray(np.load(folder / 'pca_components' / f'layer_{layer + 1:02d}.float32.npy'), dtype=np.float64)
    total = projected = 0.0
    for start in range(0, len(neutral), 128):
        x = np.asarray(neutral[start:start + 128, layer], dtype=np.float64) - xmean
        total += np.sum(x * x)
        projected += np.sum((x @ pc.T) ** 2)
    explained = projected / total
    if not np.allclose(xmean, saved_mean, atol=0.0001, rtol=0.0001):
        raise ValueError('Neutral PCA mean mismatch.')
    if not np.isclose(explained, details[layer * 8]['variance_explained'], atol=2e-05):
        raise ValueError('Neutral PCA variance mismatch.')
    for dataset in DATASETS:
        h, y, _, meta = load_hidden(p, KEY, dataset, 'train')
        assert meta['layer'] == info['layer'] == 27
        fdir = np.load(ART / 'stage_f' / KEY / dataset / 'sentiment_direction.npy')
        rebuilt = h[y == 1].mean(axis=0, dtype=np.float64) - h[y == 0].mean(axis=0, dtype=np.float64)
        if not np.allclose(fdir, rebuilt, atol=1e-10, rtol=1e-10):
            raise ValueError('Stage F direction does not match E training means.')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'denoising_all_layers.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(details[0]))
        w.writeheader()
        w.writerows(details)
    report = dict(complete=True, shape=list(final.shape), model_id=MODELS[KEY], selected_layer_1_based=27, physical_block_0_based=26, activation_site='post decoder block model.layers[26].output; no final RMSNorm', raw_sha256=sha(ART / 'stage_b_raw_vectors' / KEY / 'emotion_directions_raw.float32.npy'), final_sha256=sha(folder / 'emotion_directions_final.float32.npy'), selected_layer_neutral_mean_max_error=float(np.max(np.abs(xmean - saved_mean))), selected_layer_neutral_variance_independent=float(explained), all_layer_projection_formula_verified=True, pooling='B/C: bare text, right padding; E/F: fixed T1 chat context, text-span mask, left padding. Same within each stage across all Qwens; context differs between stages.', normalization='B PCA acts on unnormalized raw directions; E/C normalize afterward; F valence uses original magnitudes.')
    atomic_json(OUT / 'audit.json', report)
    selected = details[26 * 8:27 * 8]
    lines = ['Qwen 1.5B audit: indexing, shapes/hashes, PCA formula and F reconstruction PASS.', 'Physical block: model.layers[26].output = layer 27/28 in B/C/E/F.', f'Layer 27 neutral PCs: {len(pc)}; independently verified variance: {explained:.6%}.', f'Layer 27 mean raw-to-denoised cosine: {np.mean([r['raw_denoised_cosine'] for r in selected]):.6f}.', f'Layer 27 mean norm fraction removed: {np.mean([r['norm_fraction_removed'] for r in selected]):.6%}.', report['pooling'], report['normalization'], 'All-layer sentiment coverage and fresh-neutral regeneration require separate CUDA passes; neither follows from this audit.']
    (OUT / 'audit_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)

def coverage(batch_size):
    import torch
    from nnsight import NNsight
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from stage_d import token_batch
    p, info, raw, final = inputs()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required.')
    tok = AutoTokenizer.from_pretrained(MODELS[KEY], use_fast=True, local_files_only=True, revision=model_revision(MODELS[KEY]))
    import hashlib
    if hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest() != p['models'][KEY]['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from Stage E.')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODELS[KEY], dtype=torch.float16, local_files_only=True, revision=model_revision(MODELS[KEY])).eval().to('cuda')
    assert len(base.model.layers) == 28 and base.config.hidden_size == 1536
    wrapped = NNsight(base)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in DATASETS:
        h, y, rows, meta = load_hidden(p, KEY, dataset, 'train')
        checkpoint = OUT / f'{dataset}_class_sums.npz'
        signature = dict(protocol_sha256=sha(E / 'protocol.json'), source_sha256=meta['source_sha256'], batch_size=batch_size)
        signature_path = OUT / f'{dataset}_coverage_signature.json'
        if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
            raise ValueError('Diagnostic resume settings changed.')
        atomic_json(signature_path, signature)
        sums = np.zeros((2, 28, 1536), dtype=np.float64)
        counts = np.zeros(2, dtype=np.int64)
        start_row = 0
        max_error = 0.0
        if checkpoint.exists():
            with np.load(checkpoint) as saved:
                sums = saved['sums'].copy()
                counts = saved['counts'].copy()
                start_row = int(saved['next_row'])
                max_error = float(saved['selected_layer_max_abs_error'])
        for start in range(start_row, len(rows), batch_size):
            indices = range(start, min(start + batch_size, len(rows)))
            inputs_, mask, _, clipped = token_batch(tok, KEY, p['models'][KEY]['prompts'][dataset], [rows[i] for i in indices])
            if any(clipped):
                raise ValueError('Text truncation forbidden.')
            weights = mask.unsqueeze(-1)
            denom = weights.sum(dim=1)
            with torch.inference_mode():
                with wrapped.trace(inputs_['input_ids'], attention_mask=inputs_['attention_mask'], use_cache=False):
                    pooled = []
                    for l in range(28):
                        residual = wrapped.model.layers[l].output
                        pooled.append((residual.float() * weights).sum(dim=1) / denom)
                    saved = torch.stack(pooled, dim=1).cpu().save()
            values = saved.numpy()
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite activations.')
            error = float(np.max(np.abs(values[:, 26] - np.asarray(h[start:start + len(values)]))))
            max_error = max(max_error, error)
            for label in (0, 1):
                selected = values[y[start:start + len(values)] == label]
                sums[label] += selected.sum(axis=0, dtype=np.float64)
                counts[label] += len(selected)
            temp = checkpoint.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f, sums=sums, counts=counts, next_row=start + len(values), selected_layer_max_abs_error=max_error)
            for attempt in range(30):
                try:
                    temp.replace(checkpoint)
                    break
                except PermissionError:
                    if attempt == 29:
                        raise
                    time.sleep(0.2)
            if start % (batch_size * 100) == 0:
                print(f'All layers {dataset}: {start + len(values)}/{len(rows)}', flush=True)
        directions = sums[1] / counts[1] - sums[0] / counts[0]
        np.save(OUT / f'{dataset}_sentiment_all_layers.npy', directions)
        stage_e_sent = h[y == 1].mean(axis=0, dtype=np.float64) - h[y == 0].mean(axis=0, dtype=np.float64)
        for l in range(28):
            for space, v in [('raw', raw), ('denoised', final)]:
                metrics, *_ = geometry(directions[l], v[:, l])
                results.append(dict(dataset=dataset, layer=l + 1, space=space, **metrics))
        atomic_json(OUT / f'{dataset}_selected_layer_parity.json', dict(max_example_abs_error=max_error, sentiment_cosine=cosine(directions[26], stage_e_sent), mean_direction_relative_error=float(np.linalg.norm(directions[26] - stage_e_sent) / np.linalg.norm(stage_e_sent)), note='Batch differences may cause floating-point variation; report actual parity, do not silently infer equality.'))
    with (OUT / 'coverage_all_layers.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'visualizations' / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = ROOT / 'visualizations' / 'stage_f_1_5b_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        for space in ('raw', 'denoised'):
            r = [x for x in results if x['dataset'] == dataset and x['space'] == space]
            ax.plot([x['layer'] for x in r], [x['explained_percent'] for x in r], label=space)
        ax.axvline(27, color='gray', ls='--', label='Frozen selected layer')
        ax.set_title(dataset)
        ax.set_xlabel('Transformer block (1-based)')
        ax.legend(frameon=False)
    axes[0].set_ylabel('Sentiment-direction coverage (%)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'coverage_all_28_layers.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('All-layer diagnostic complete; selection unchanged.', flush=True)

def fresh(batch_size):
    import torch
    import build_stage_b_final_vectors as b
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required.')
    target = OUT / ('fresh_b7_b9_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    target.mkdir(parents=True, exist_ok=False)
    data, digest, manifest = b.neutral_training()
    signature = b.neutral_signature(KEY, digest, manifest, b.model_dimensions(KEY), len(data))
    complete = b.extract_neutral(KEY, data, signature, target / 'neutral', batch_size, None)
    if not complete:
        raise RuntimeError('Fresh neutral extraction incomplete.')
    b.denoise(KEY, signature, target / 'neutral', target / 'final')
    original = ART / 'stage_b_final_vectors' / KEY / 'emotion_directions_final.float32.npy'
    rebuilt = target / 'final' / 'emotion_directions_final.float32.npy'
    old, new = (np.load(original), np.load(rebuilt))
    delta = new.astype(np.float64) - old.astype(np.float64)
    result = dict(complete=True, fresh_directory=str(target), neutral_cache_reused=False, original_sha256=sha(original), fresh_sha256=sha(rebuilt), bitwise_equal=bool(np.array_equal(old, new)), max_abs_error=float(np.max(np.abs(delta))), relative_frobenius_error=float(np.linalg.norm(delta) / np.linalg.norm(old)), min_vector_cosine=min((cosine(old[e, l], new[e, l]) for e in range(8) for l in range(28))), note='Hash differences alone do not establish a geometry difference; CUDA/eigendecomposition floating-point variation is possible.')
    atomic_json(target / 'comparison.json', result)
    (target / 'comparison_summary.txt').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['audit', 'coverage', 'fresh', 'all'], default='audit')
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('Batch size must be positive.')
    if a.phase in ('audit', 'all'):
        audit()
    if a.phase in ('coverage', 'all'):
        coverage(a.batch_size)
    if a.phase in ('fresh', 'all'):
        fresh(a.batch_size)
if __name__ == '__main__':
    main()
