"""Generate local TweetEval binary margins for the extracted correction pipeline.

This is new orchestration, not a claim to recover historical private predictions.
The FP16 forward, last-attended readout and float32 subtraction match the
source correction protocol. Run only after Stage E extraction is complete.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from release_paths import WORKSPACE, source_path

def sha(path):
    return hashlib.sha256(source_path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if not args.execute or args.batch_size < 1:
        parser.error('Explicit --execute and a positive batch size are required')
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from stage_d import token_batch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no precision-changing CPU fallback')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stage_e = WORKSPACE / 'artifacts/stage_e'
    out = WORKSPACE / 'artifacts/stage_i_sentiment_correction/reused_inputs'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh workspace; margin preparation does not overwrite inputs')
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((stage_e / 'protocol.json').read_text())
    data = {}
    for split in ('train', 'validation', 'test'):
        source = stage_e / 'data/tweet_eval_sentiment' / (split + '.jsonl')
        data[split] = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        (out / source.name).write_bytes(source.read_bytes())
    db = sqlite3.connect(out / 'scores.sqlite3')
    db.execute('CREATE TABLE scores (model TEXT, split TEXT, example_index INTEGER, id TEXT, label INTEGER, logp_positive REAL, logp_negative REAL, margin REAL, PRIMARY KEY(model,split,example_index))')
    identities = {}
    try:
        for key in ('0.5b', '3b'):
            cfg = protocol['models'][key]
            prompt = cfg['prompts']['tweet_eval_sentiment']
            tok = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['checkpoint_commit'], use_fast=True, local_files_only=True)
            assert hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest() == cfg['tokenizer_sha256']
            tok.padding_side = 'left'
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            assert tok.encode('positive', add_special_tokens=False) == [30487]
            assert tok.encode('negative', add_special_tokens=False) == [42224]
            model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], revision=cfg['checkpoint_commit'], dtype=torch.float16, attn_implementation='sdpa', local_files_only=True).eval().to('cuda')
            model.requires_grad_(False)
            identities[key] = {field: cfg[field] for field in ('model_id', 'checkpoint_commit', 'tokenizer_sha256', 'config_sha256')}
            identities[key]['prompt'] = prompt
            for split, rows in data.items():
                for start in range(0, len(rows), args.batch_size):
                    block = rows[start:start + args.batch_size]
                    inputs, _, _, truncated = token_batch(tok, key, prompt, block)
                    if any(truncated):
                        raise ValueError('Truncation is prohibited')
                    with torch.inference_mode():
                        logits = model(**inputs, use_cache=False).logits
                        pos = torch.arange(inputs['input_ids'].shape[1], device='cuda').expand_as(inputs['input_ids'])
                        last = pos.masked_fill(~inputs['attention_mask'].bool(), -1).max(1).values
                        lp = torch.log_softmax(logits[torch.arange(len(block), device='cuda'), last].float(), -1)[:, [30487, 42224]].cpu().numpy()
                    margin = (lp[:, 0] - lp[:, 1]).astype(np.float64)
                    db.executemany('INSERT INTO scores VALUES (?,?,?,?,?,?,?,?)', [(key, split, start + i, r['id'], r['label'], float(lp[i, 0]), float(lp[i, 1]), float(margin[i])) for i, r in enumerate(block)])
                    db.commit()
            del model, tok
            torch.cuda.empty_cache()
    finally:
        db.close()
    scoring = {'models': identities, 'source_sha256': {'artifacts/stage_e/protocol.json': sha(stage_e / 'protocol.json')}, 'precision': 'FP16 CUDA SDPA; float32 log_softmax and subtraction; binary margin>0; ties negative'}
    (out / 'original_scoring_protocol.json').write_text(json.dumps(scoring, indent=2))
    hashes = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'preservation_manifest.json').write_text(json.dumps({'preserved_sha256': hashes}, indent=2))
if __name__ == '__main__':
    main()
