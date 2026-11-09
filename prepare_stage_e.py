"""Freeze binary Stage E fit/tuning/test-proxy splits without touching earlier stages."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
import numpy as np
from datasets import load_from_disk
from stage_e_common import ART, E, DATASETS, SPLITS, atomic_json, sha

def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

def record(dataset, source_split, index, text, label):
    return dict(id=f'{dataset}:{source_split}:{index}', dataset=dataset, source_split=source_split, source_row_index=int(index), label=int(label), text=text, text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())

def grouped_split(rows, fraction, seed):
    """Stratify normalized-text groups by majority class; keep duplicates together."""
    groups = defaultdict(list)
    for row in rows:
        groups[normalized(row['text'])].append(row)
    buckets = defaultdict(list)
    for text, members in groups.items():
        counts = Counter((r['label'] for r in members))
        majority = min(counts, key=lambda x: (-counts[x], x))
        buckets[majority].append(text)
    rng = np.random.default_rng(seed)
    held = set()
    for label in sorted(buckets):
        items = sorted(buckets[label])
        rng.shuffle(items)
        n = max(1, min(len(items) - 1, round(len(items) * fraction)))
        held.update(items[:n])
    train = [r for r in rows if normalized(r['text']) not in held]
    val = [r for r in rows if normalized(r['text']) in held]
    return (train, val)

def remove_overlap(rows, forbidden):
    kept = [r for r in rows if normalized(r['text']) not in forbidden]
    return (kept, len(rows) - len(kept))

def prepare(seed=42, val_fraction=0.2):
    source = ART / 'stage_d' / 'examples.jsonl'
    if not source.exists():
        raise FileNotFoundError('Stage D examples.jsonl is required to preserve the existing SST-2 sample.')
    old = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    sst = [r for r in old if r['dataset'] == 'sst2']
    sampled = [record('sst2', r['source_split'], r['source_row_index'], r['text'], int(r['gold_label'] == 'positive')) for r in sst if r['source_split'] == 'train']
    test = [record('sst2', r['source_split'], r['source_row_index'], r['text'], int(r['gold_label'] == 'positive')) for r in sst if r['source_split'] == 'validation']
    if len(sampled) != 12000 or len(test) != 872:
        raise ValueError('Unexpected frozen SST-2 sample size.')
    sampled, dropped = remove_overlap(sampled, {normalized(r['text']) for r in test})
    train, val = grouped_split(sampled, val_fraction, seed)
    datasets = {'sst2': dict(train=train, validation=val, test=test)}
    drops = {'sst2': {'train_rows_matching_test_proxy': dropped}}
    tweet_path = ART / 'datasets' / 'tweet_eval_sentiment'
    tweets = load_from_disk(str(tweet_path))
    datasets['tweet_eval_sentiment'] = {}
    for split in SPLITS:
        datasets['tweet_eval_sentiment'][split] = [record('tweet_eval_sentiment', split, i, r['text'], int(r['label'] == 2)) for i, r in enumerate(tweets[split]) if r['label'] in (0, 2)]
    parts = datasets['tweet_eval_sentiment']
    parts['validation'], drop_val = remove_overlap(parts['validation'], {normalized(r['text']) for r in parts['test']})
    forbidden = {normalized(r['text']) for s in ('validation', 'test') for r in parts[s]}
    parts['train'], drop_train = remove_overlap(parts['train'], forbidden)
    drops['tweet_eval_sentiment'] = {'validation_rows_matching_test': drop_val, 'train_rows_matching_validation_or_test': drop_train}
    info = dict(seed=seed, sst2_validation_fraction=val_fraction, binary_labels={'0': 'negative', '1': 'positive'}, neutral_policy='excluded', stage_d_examples_sha256=sha(source), policy='Fixed Stage D SST-2 training sample; grouped stratified train/tuning split. TweetEval official splits.', sst2_test_policy='Official validation is a labeled test proxy, previously observed in Stages A/D; official test gold labels are unavailable. Not a new unbiased benchmark.', duplicate_policy='Normalized exact texts never cross fit/tuning/test; within-split duplicates retained. This does not guarantee absence of SST phrase/review overlap.', removed_cross_split_rows=drops, datasets={})
    old_path = E / 'data' / 'manifest.json'
    prepared = {}
    for dataset in DATASETS:
        parts = datasets[dataset]
        sets = {s: {normalized(r['text']) for r in parts[s]} for s in SPLITS}
        for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            if sets[a] & sets[b]:
                raise ValueError(f'Normalized exact-text leakage: {dataset}/{a}/{b}')
        info['datasets'][dataset] = dict(splits={})
        for split, rows in parts.items():
            if {r['label'] for r in rows} != {0, 1} or len({r['id'] for r in rows}) != len(rows):
                raise ValueError('Each partition requires unique source IDs and both classes.')
            text = ''.join((json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in rows))
            path = E / 'data' / dataset / f'{split}.jsonl'
            prepared[path] = text
            info['datasets'][dataset]['splits'][split] = dict(n=len(rows), counts=dict(Counter((str(r['label']) for r in rows))), sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(), source_splits=sorted({r['source_split'] for r in rows}))
    if old_path.exists() and json.loads(old_path.read_text(encoding='utf-8')) != info:
        raise ValueError('Existing Stage E split manifest differs; refusing to overwrite frozen data.')
    for path, text in prepared.items():
        if path.exists() and path.read_text(encoding='utf-8') != text:
            raise ValueError(f'Existing frozen rows differ: {path}')
    for path, text in prepared.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = text.encode('utf-8')
        if not path.exists() or path.read_bytes() != encoded:
            path.write_bytes(encoded)
    atomic_json(old_path, info)
    for dataset in DATASETS:
        print(dataset, {s: info['datasets'][dataset]['splits'][s]['n'] for s in SPLITS})
    print(f'Saved frozen splits: {old_path}')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sst2-validation-fraction', type=float, default=0.2)
    args = p.parse_args()
    if not 0 < args.sst2_validation_fraction < 1:
        p.error('Validation fraction must be between zero and one.')
    prepare(args.seed, args.sst2_validation_fraction)
