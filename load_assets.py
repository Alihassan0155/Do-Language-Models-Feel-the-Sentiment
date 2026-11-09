"""Explicit asset acquisition and baseline helpers; importing performs no I/O."""
import argparse
import json
from release_paths import WORKSPACE, model_revision
MODEL_IDS = {'0.5b': 'Qwen/Qwen2.5-0.5B-Instruct', '1.5b': 'Qwen/Qwen2.5-1.5B-Instruct'}

def load_sentiment_datasets():
    from datasets import load_dataset, load_from_disk
    specs = {'sst2': ('stanfordnlp/sst2', None), 'tweet_eval_sentiment': ('cardiffnlp/tweet_eval', 'sentiment')}
    result = {}
    for name, (identifier, config) in specs.items():
        path = WORKSPACE / 'artifacts/datasets' / name
        if path.exists():
            result[name] = load_from_disk(str(path))
        else:
            result[name] = load_dataset(identifier, config)
            path.parent.mkdir(parents=True, exist_ok=True)
            result[name].save_to_disk(str(path))
    return result

def load_model(model_size, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    identifier = MODEL_IDS[model_size]
    tok = AutoTokenizer.from_pretrained(identifier, use_fast=True, local_files_only=True, revision=model_revision(identifier))
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(identifier, dtype=torch.float16 if device == 'cuda' else torch.float32, local_files_only=True, revision=model_revision(identifier)).eval().to(device)
    return (model, tok)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', action='store_true')
    parser.add_argument('--model', choices=['0.5b', '1.5b', '3b', 'bert_large_uncased'])
    parser.add_argument('--revision', help='Explicit provider revision; required with --model')
    args = parser.parse_args()
    if not args.datasets and (not args.model):
        parser.error('Choose --datasets or --model')
    if args.model and (not args.revision):
        parser.error('--model requires --revision')
    if args.datasets:
        load_sentiment_datasets()
    if args.model:
        from huggingface_hub import snapshot_download
        names = dict(MODEL_IDS, **{'3b': 'Qwen/Qwen2.5-3B-Instruct', 'bert_large_uncased': 'google-bert/bert-large-uncased'})
        snapshot = snapshot_download(names[args.model], revision=args.revision, cache_dir=str(WORKSPACE / '.cache/huggingface/hub'))
        path = WORKSPACE / 'model_revisions.json'
        revisions = json.loads(path.read_text()) if path.exists() else {}
        identifier = names[args.model]
        resolved = __import__('pathlib').Path(snapshot).name
        if identifier in revisions and revisions[identifier] != resolved:
            raise ValueError('Model revision is already frozen; use a fresh workspace')
        revisions[identifier] = resolved
        path.write_text(json.dumps(revisions, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
