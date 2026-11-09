"""Shared Stage E paths, frozen binary rows and selected-layer provenance."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import hashlib
import json
import os
from pathlib import Path
ROOT = WORKSPACE
ART = ROOT / 'artifacts'
E = ART / 'stage_e'
EMOTIONS = ('joy', 'love', 'excitement', 'gratitude', 'sadness', 'anger', 'fear', 'disgust')
MODELS = {'0.5b': 'Qwen/Qwen2.5-0.5B-Instruct', '1.5b': 'Qwen/Qwen2.5-1.5B-Instruct', '3b': 'Qwen/Qwen2.5-3B-Instruct', 'bert_large_uncased': 'google-bert/bert-large-uncased'}
DATASETS = ('sst2', 'tweet_eval_sentiment')
SPLITS = ('train', 'validation', 'test')
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))

def sha(path):
    path = source_path(path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    tmp.replace(path)

def read_rows(dataset, split):
    path = E / 'data' / dataset / (split + '.jsonl')
    manifest = json.loads((E / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    entry = manifest['datasets'][dataset]['splits'][split]
    if sha(path) != entry['sha256']:
        raise ValueError(f'Frozen Stage E data changed: {path}')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) != entry['n'] or {r['label'] for r in rows} != {0, 1}:
        raise ValueError(f'Invalid binary split: {dataset}/{split}')
    return (rows, entry)

def selection(key):
    path = ART / 'stage_c_empathetic_dialogues' / 'selected_intervention_layers.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    model = data['models'][key]
    if model['model_id'] != MODELS[key]:
        raise ValueError('Stage C model differs from Stage E.')
    return (int(model['selected']['layer_1_based']), sha(path))
