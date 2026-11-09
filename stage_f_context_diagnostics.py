"""Cheap Stage F diagnostics: neutral removal and selected-layer bare-text geometry."""
from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import csv
import json
import numpy as np
from stage_e_common import ROOT, ART, E, MODELS, DATASETS, atomic_json, sha
from stage_e_pipeline import assert_protocol, emotion_inputs, load_hidden
from stage_f import geometry, cosine
OUT = ART / 'stage_f_context_diagnostics'

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def protocol():
    p = json.loads((E / 'protocol.json').read_text(encoding='utf-8'))
    assert_protocol(p)
    return p

def removal():
    p = protocol()
    rows = []
    for key in MODELS:
        _, info = emotion_inputs(key)
        pc = np.asarray(np.load(ART / 'stage_b_final_vectors' / key / 'pca_components' / f'layer_{info['layer']:02d}.float32.npy'), dtype=np.float64)
        if not np.allclose(pc @ pc.T, np.eye(len(pc)), atol=2e-05, rtol=2e-05):
            raise ValueError('Nonorthogonal neutral PCs.')
        for dataset in DATASETS:
            h, y, _, _ = load_hidden(p, key, dataset, 'train')
            sent = h[y == 1].mean(axis=0, dtype=np.float64) - h[y == 0].mean(axis=0, dtype=np.float64)
            prior = np.load(ART / 'stage_f' / key / dataset / 'sentiment_direction.npy')
            if not np.allclose(sent, prior, atol=1e-10, rtol=1e-10):
                raise ValueError('Stage F sentiment source mismatch.')
            removed = sent @ pc.T @ pc
            fraction = float(np.dot(removed, removed) / np.dot(sent, sent))
            if not -1e-06 <= fraction <= 1 + 1e-05:
                raise ValueError('Invalid fraction.')
            rows.append(dict(model=key, dataset=dataset, layer=info['layer'], neutral_pcs=len(pc), removed_sentiment_fraction=fraction, removed_sentiment_percent=100 * fraction, retained_sentiment_percent=100 * (1 - fraction)))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'neutral_removed_sentiment.csv', rows)
    lines = ['Neutral PCA sentiment removal: saved T1 training directions only; no forward passes.']
    lines += [f'{r['model']} {r['dataset']} layer {r['layer']}: {r['removed_sentiment_percent']:.2f}% removed ({r['neutral_pcs']} PCs)' for r in rows]
    (OUT / 'neutral_removed_sentiment_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_json(OUT / 'neutral_removal_metadata.json', dict(protocol_sha256=sha(E / 'protocol.json'), split='train', formula='norm(PC.T @ PC @ v_sent)^2 / norm(v_sent)^2', complete=True))
    print('\n'.join(lines), flush=True)

def bare(keys, batch_size):
    import torch
    from nnsight import NNsight
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    from build_stage_b_vectors import tokenized_batch as qwen_tokens
    from build_stage_b_bert_vectors import tokenized_batch as bert_tokens
    import gc
    p = protocol()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required.')
    OUT.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for key in keys:
        _, info = emotion_inputs(key)
        tok = AutoTokenizer.from_pretrained(MODELS[key], local_files_only=True, revision=model_revision(MODELS[key]))
        tok.padding_side = 'right'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        loader = AutoModel if key == 'bert_large_uncased' else AutoModelForCausalLM
        base = loader.from_pretrained(MODELS[key], dtype=torch.float16, local_files_only=True, revision=model_revision(MODELS[key])).eval().to('cuda')
        wrapped = NNsight(base)
        for dataset in DATASETS:
            h, y, rows, meta = load_hidden(p, key, dataset, 'train')
            target = OUT / key / dataset
            target.mkdir(parents=True, exist_ok=True)
            sig = dict(protocol_sha256=sha(E / 'protocol.json'), train_source_sha256=meta['source_sha256'], layer=info['layer'], batch_size=batch_size, format='exact Stage B/C bare-text tokenizer/mask, right padding', storage='float32 pooled states')
            sig_path = target / 'signature.json'
            if sig_path.exists() and json.loads(sig_path.read_text()) != sig:
                raise ValueError('Bare-text resume settings changed.')
            atomic_json(sig_path, sig)
            state = target / 'progress.json'
            start_row = json.loads(state.read_text())['completed_rows'] if state.exists() else 0
            path = target / 'pooled.float32.npy'
            a = np.load(path, mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), info['width']))
            if a.shape != (len(rows), info['width']):
                raise ValueError('Invalid bare hidden shape.')
            for start in range(start_row, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                tokens, mask, _ = (bert_tokens if key == 'bert_large_uncased' else qwen_tokens)(tok, batch)
                weights = mask.unsqueeze(-1)
                denom = weights.sum(dim=1)
                kwargs = dict(attention_mask=tokens['attention_mask'])
                if key == 'bert_large_uncased':
                    kwargs['token_type_ids'] = tokens.get('token_type_ids')
                else:
                    kwargs['use_cache'] = False
                with torch.inference_mode():
                    with wrapped.trace(tokens['input_ids'], **kwargs):
                        residual = wrapped.encoder.layer[info['layer'] - 1].nns_output[0] if key == 'bert_large_uncased' else wrapped.model.layers[info['layer'] - 1].output
                        pooled = ((residual.float() * weights).sum(dim=1) / denom).cpu().save()
                values = pooled.numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite bare states.')
                a[start:start + len(values)] = values
                a.flush()
                atomic_json(state, dict(completed_rows=start + len(values), complete=start + len(values) == len(rows)))
                if start % (batch_size * 100) == 0:
                    print(f'Bare text {key}/{dataset}: {start + len(values)}/{len(rows)}', flush=True)
            sent = a[y == 1].mean(axis=0, dtype=np.float64) - a[y == 0].mean(axis=0, dtype=np.float64)
            prompted = h[y == 1].mean(axis=0, dtype=np.float64) - h[y == 0].mean(axis=0, dtype=np.float64)
            np.save(target / 'sentiment_direction.npy', sent)
            for space, folder, filename in [('raw', 'stage_b_raw_vectors', 'emotion_directions_raw.float32.npy'), ('denoised', 'stage_b_final_vectors', 'emotion_directions_final.float32.npy')]:
                v = np.load(ART / folder / key / filename)[:, info['layer'] - 1]
                bare_metrics, *_ = geometry(sent, v)
                prompt_metrics, *_ = geometry(prompted, v)
                comparisons.append(dict(model=key, dataset=dataset, layer=info['layer'], space=space, bare_coverage_percent=bare_metrics['explained_percent'], prompted_coverage_percent=prompt_metrics['explained_percent'], bare_valence_cosine=bare_metrics['cosine_valence'], prompted_valence_cosine=prompt_metrics['cosine_valence'], bare_prompted_sentiment_cosine=cosine(sent, prompted)))
            atomic_json(target / 'metadata.json', dict(complete=True, signature=sig, hidden_sha256=sha(path), n_negative=int(sum(y == 0)), n_positive=int(sum(y == 1))))
            del a, h
        del wrapped, base, tok
        gc.collect()
        torch.cuda.empty_cache()
    csv_path = OUT / 'bare_vs_prompted_geometry.csv'
    if csv_path.exists():
        old = list(csv.DictReader(csv_path.open(encoding='utf-8')))
        comparisons = [r for r in old if r['model'] not in keys] + comparisons
    write_csv(csv_path, comparisons)
    lines = ['Selected-layer context diagnostic; identical Stage E training rows, no validation/test data.']
    lines += [f'{r['model']} {r['dataset']} {r['space']}: prompted coverage={float(r['prompted_coverage_percent']):.2f}%, bare={float(r['bare_coverage_percent']):.2f}%; sentiment cosine={float(r['bare_prompted_sentiment_cosine']):.4f}' for r in comparisons]
    (OUT / 'bare_vs_prompted_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['removal', 'bare', 'all'], default='removal')
    p.add_argument('--model', choices=[*MODELS, 'all'], default='1.5b')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('Batch size must be positive.')
    if args.phase in ('removal', 'all'):
        removal()
    if args.phase in ('bare', 'all'):
        bare(list(MODELS) if args.model == 'all' else [args.model], args.batch_size)
if __name__ == '__main__':
    main()
