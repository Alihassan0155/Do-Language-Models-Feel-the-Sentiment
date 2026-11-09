"""E2: frozen fresh-row SST-2 extension; historical inputs remain read-only."""
from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import unicodedata
OUT = WORKSPACE / 'artifacts/e2_sst2_distortion_matched_20261018'
ROOT = WORKSPACE
ART = ROOT / 'artifacts'
sys.path.insert(0, str(CODE_ROOT))
import numpy as np
import stage_h_distortion_matched as M
M.DATASET = 'sst2'
KEYS = ('0.5b', '3b')
SEED = 20261018

def sha(path):
    path = source_path(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(name, value):
    M.atomic_json(OUT / name, value)

def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

def arrow_rows(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    return ipc.open_stream(pa.memory_map(str(path), 'r')).read_all().to_pylist()

def audit_and_freeze():
    if (OUT / 'protocol.json').exists():
        verify_inputs()
        return (read(ART / 'stage_e/protocol.json'), read(OUT / 'protocol.json'), read(OUT / 'samples.json'))
    used, forbidden, scanned, failures = (set(), set(), [], [])

    def record(row):
        if not isinstance(row, dict):
            return
        ident = str(row.get('id', ''))
        is_sst = ident.startswith('sst2:train:')
        if is_sst:
            used.add(int(ident.rsplit(':', 1)[1]))
        if row.get('dataset') == 'sst2' and row.get('source_split', row.get('split')) == 'train':
            index = row.get('source_row_index', row.get('row_index'))
            if index is not None:
                used.add(int(index))
            is_sst = True
        if is_sst and isinstance(row.get('text'), str):
            forbidden.add(normalized(row['text']))
    paths = sorted((p for p in ART.rglob('*') if p.is_file() and OUT not in p.parents))
    for path in paths:
        if path.suffix not in ('.sqlite3', '.jsonl', '.csv') and (not path.name.endswith('samples.json')) and (path.name != 'sst2_train_12000_sample.json'):
            continue
        scanned.append(str(path.relative_to(ROOT)))
        try:
            if path.suffix == '.sqlite3':
                wal = Path(str(path) + '-wal')
                if wal.exists() and wal.stat().st_size:
                    raise ValueError('Nonempty WAL: cannot audit immutable snapshot')
                db = sqlite3.connect('file:' + path.as_posix() + '?mode=ro&immutable=1', uri=True)
                try:
                    for table, in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                        qt = '"' + table.replace('"', '""') + '"'
                        cols = {r[1] for r in db.execute('PRAGMA table_info(' + qt + ')')}
                        if {'dataset', 'source_split', 'source_row_index'} <= cols:
                            used.update((int(r[0]) for r in db.execute('SELECT DISTINCT source_row_index FROM ' + qt + " WHERE dataset='sst2' AND source_split='train'")))
                        elif {'dataset', 'split', 'row_index'} <= cols:
                            used.update((int(r[0]) for r in db.execute('SELECT DISTINCT row_index FROM ' + qt + " WHERE dataset='sst2' AND split='train'")))
                        if 'id' in cols:
                            used.update((int(r[0].rsplit(':', 1)[1]) for r in db.execute('SELECT DISTINCT id FROM ' + qt + " WHERE id LIKE 'sst2:train:%'")))
                        if path.name == 'sst2_train_12000_cuda.sqlite3' and 'row_index' in cols:
                            used.update((int(r[0]) for r in db.execute('SELECT DISTINCT row_index FROM ' + qt)))
                finally:
                    db.close()
            elif path.suffix == '.jsonl':
                for line in path.open(encoding='utf-8-sig'):
                    if line.strip():
                        record(json.loads(line))
            elif path.suffix == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    for row in csv.DictReader(handle):
                        record(row)
            else:
                obj = read(path)
                if path.name == 'sst2_train_12000_sample.json':
                    used.update(obj['row_indices'])
                seq = obj if isinstance(obj, list) else obj.get('samples', obj)
                if isinstance(seq, dict):
                    seq = [r for group in seq.values() if isinstance(group, list) for r in group]
                for row in seq:
                    record(row)
        except Exception as exc:
            failures.append({'path': str(path), 'error': str(exc)})
    if failures:
        save('audit_failure.json', failures)
        raise RuntimeError('Incomplete provenance scan; stop before selecting data.')
    train_path = ART / 'datasets/sst2/train/data-00000-of-00001.arrow'
    validation_path = ART / 'datasets/sst2/validation/data-00000-of-00001.arrow'
    train, validation = (arrow_rows(train_path), arrow_rows(validation_path))
    assert len(train) == 67349 and len(validation) == 872
    assert all((0 <= i < len(train) for i in used))
    forbidden.update((normalized(train[i]['sentence']) for i in used))
    forbidden.update((normalized(r['sentence']) for r in validation))
    additional = []
    for pool in ('vector_train', 'pca_neutral_train', 'emotion_validation'):
        path = ART / 'goemotions_b2' / pool / 'data-00000-of-00001.arrow'
        forbidden.update((normalized(r['text']) for r in arrow_rows(path)))
        additional.append(path)
    ed_path = ART / 'stage_c_empathetic_dialogues/sample.jsonl'
    forbidden.update((normalized(json.loads(line)['text']) for line in ed_path.read_text(encoding='utf-8').splitlines()))
    labels = collections.defaultdict(set)
    for row in train:
        labels[normalized(row['sentence'])].add(row['label'])
    pools, seen = ({0: [], 1: []}, set())
    for i, row in enumerate(train):
        text = normalized(row['sentence'])
        if i in used or text in forbidden or len(labels[text]) != 1 or (text in seen):
            continue
        seen.add(text)
        pools[row['label']].append(i)
    if min(map(len, pools.values())) < 300:
        raise RuntimeError('Not enough unused unique texts; stop without outcomes.')
    rng = np.random.default_rng(SEED)
    selected = {label: rng.choice(pools[label], 300, replace=False).tolist() for label in (0, 1)}
    indices = []
    for start in range(0, 300, 4):
        chunk = selected[0][start:start + 4] + selected[1][start:start + 4]
        rng.shuffle(chunk)
        indices.extend(chunk)
    rows = [dict(id=f'sst2:train:{i}', dataset='sst2', source_split='train', source_row_index=i, label=int(train[i]['label']), text=train[i]['sentence']) for i in indices]
    assert len(set(indices)) == len({normalized(r['text']) for r in rows}) == 600
    assert not set(indices) & used and (not {normalized(r['text']) for r in rows} & forbidden)
    p = read(ART / 'stage_e/protocol.json')
    M.assert_protocol(p)
    source = read(ART / 'stage_h/protocol.json')
    assert source['centering'] == 'none' and source['batch_size'] == 8
    assert source['random_seeds'] == list(range(2000, 2050))
    for key, layer in [('0.5b', 14), ('3b', 18)]:
        assert source['models'][key] == p['models'][key] and p['models'][key]['layer'] == layer
        assert source['precision'][key] == 'float16'
    source_files = [train_path, validation_path, ed_path, *additional, ART / 'stage_e/protocol.json', ART / 'stage_e/data/manifest.json', ART / 'stage_h/protocol.json', ART / 'stage_h/samples.json', ART / 'stage_h/scores.sqlite3', ART / 'stage_h_distortion_matched/protocol.json', ART / 'stage_h_distortion_matched/performance_paired_ci.csv', ART / 'stage_c_empathetic_dialogues/selected_intervention_layers.json']
    code_names = ['stage_h.py', 'stage_h_distortion_matched.py', 'stage_e_common.py', 'stage_e_pipeline.py', 'stage_g_confirmation.py', 'stage_f.py', 'stage_d.py', 'prompt_templates.py']
    source_files += [ROOT / name for name in code_names]
    for key in KEYS:
        source_files += [ART / 'stage_h' / key / 'sst2' / 'bases.npz', ART / 'stage_h' / key / 'sst2' / 'bases_metadata.json', ART / 'stage_b_raw_vectors' / key / 'emotion_directions_raw.float32.npy', ART / 'stage_b_final_vectors' / key / 'emotion_directions_final.float32.npy']
    for rel in scanned:
        path = ROOT / rel
        if path.suffix in ('.sqlite3', '.jsonl') and ('sst2' in rel or path.name.endswith('scores.sqlite3')):
            source_files.append(path)
    fingerprints = {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(source_files))}
    save('input_hashes.json', fingerprints)
    save('prior_used_sst2_train_ids.json', sorted(used))
    save('dataset_audit.json', dict(source='stanfordnlp/sst2', official_split='train', official_train_n=len(train), prior_used_ids=len(used), available_unique_texts={str(k): len(v) for k, v in pools.items()}, n=600, negative=300, positive=300, selection_seed=SEED, scanned_files=scanned, source_id_overlap=0, normalized_text_overlap=0, official_validation_used=0, audit_scope='saved repository outcome/training IDs in SQLite, JSONL, CSV, sample JSON and frozen 12k manifest; cross-corpus construction/validation texts excluded', status='new row/text-disjoint internal holdout; exploratory extension, not official validation/test', limitations='Original review grouping unavailable; phrase/review-level independence and undocumented external inspection cannot be established.', outcome_access_before_freeze=False, batch_rule='75 frozen batches, four negative and four positive each'))
    with (OUT / 'dataset_ids.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['example_index', 'id', 'source_split', 'source_row_index', 'label', 'normalized_text_sha256'])
        writer.writeheader()
        for j, row in enumerate(rows):
            writer.writerow(dict(example_index=j, **{k: row[k] for k in ['id', 'source_split', 'source_row_index', 'label']}, normalized_text_sha256=hashlib.sha256(normalized(row['text']).encode()).hexdigest()))
    config = dict(schema=1, dataset='sst2', n=600, models=list(KEYS), batch_size=8, layers={k: p['models'][k]['layer'] for k in KEYS}, random_seeds=source['random_seeds'], precision={k: 'float16' for k in KEYS}, bootstrap_draws=2000, bootstrap_seed=45, seed_bootstrap_seed=46, match_relative_tolerance=0.01, match_absolute_tolerance=1e-07, matching='original matching_change unmodified; one beta per example; 16 bisection steps; actual rounded token-relative energy', statistics='paired class-stratified example and paired seed joint bootstrap; percentile 95%; nominal, no multiplicity correction', source_sha256=fingerprints, sample_sha256=None, exploratory=True, no_outcome_tuning=True, settings_origin='historical TweetEval matched ablation; frozen original SST-2 P1 and candidate-label scoring', reference_reuse='none: baseline, full emotion removals and all 50 random controls are evaluated afresh')
    save('samples.json', {'sst2': rows})
    config['sample_sha256'] = sha(OUT / 'samples.json')
    save('protocol.json', config)
    (OUT / 'source_snapshot').mkdir(exist_ok=True)
    for name in code_names:
        shutil.copyfile(ROOT / name, OUT / 'source_snapshot' / name)
    save('frozen_before_inference.json', dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), protocol_sha256=sha(OUT / 'protocol.json'), sample_sha256=config['sample_sha256'], runner_sha256=sha(__file__)))
    print('Frozen 600 new row/text-disjoint examples; prior used IDs:', len(used), flush=True)
    return (p, config, {'sst2': rows})

def verify_inputs():
    for rel, expected in read(OUT / 'input_hashes.json').items():
        if sha(ROOT / rel) != expected:
            raise ValueError('Historical input changed: ' + rel)
    if (OUT / 'protocol.json').exists():
        assert sha(OUT / 'samples.json') == read(OUT / 'protocol.json')['sample_sha256']
    if (OUT / 'frozen_before_inference.json').exists():
        assert sha(__file__) == read(OUT / 'frozen_before_inference.json')['runner_sha256']

def runtime_preflight(p):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from stage_e_common import MODELS
    evidence = dict(python=sys.version, executable=sys.executable, gpu=torch.cuda.get_device_name(0), models={})
    for package in ('torch', 'transformers', 'numpy', 'scikit-learn', 'nnsight'):
        evidence[package] = importlib.metadata.version(package)
    for key in KEYS:
        info = p['models'][key]
        cfg = AutoConfig.from_pretrained(MODELS[key], local_files_only=True, revision=model_revision(MODELS[key]))
        tok = AutoTokenizer.from_pretrained(MODELS[key], use_fast=True, local_files_only=True, revision=model_revision(MODELS[key]))
        assert cfg._commit_hash == info['checkpoint_commit']
        assert hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest() == info['tokenizer_sha256']
        assert cfg.hidden_size == info['width'] and cfg.num_hidden_layers == info['total_layers']
        evidence['models'][key] = dict(model_id=MODELS[key], checkpoint_commit=cfg._commit_hash, prompt=info['prompts']['sst2'], layer=info['layer'], dtype='float16')
    save('runtime_preflight.json', evidence)

def empty_reference(folder, config, n):
    M.H.database(folder).close()

def summarize(config, samples, args, smoke=False):
    M.report(config, samples, args, smoke)
    folder = OUT / 'smoke' if smoke else OUT
    summary = folder / 'results_summary.txt'
    summary.write_text(summary.read_text(encoding='utf-8').replace('frozen TweetEval examples', 'frozen SST-2 internal-holdout examples') + '\nE2 is an exploratory extension. New source IDs and normalized texts; review-level independence unverified. All forward results computed anew.\n', encoding='utf-8')
    rows = list(csv.DictReader((folder / 'damage_vs_distortion.csv').open(encoding='utf-8')))
    result = []
    for key in KEYS:
        for condition in ('random_full', 'emotion_raw_matched', 'emotion_denoised_matched'):
            selected = [r for r in rows if r['model'] == key and r['condition'] == condition]
            for metric in ('accuracy_damage_pp', 'macro_f1_damage_pp', 'auroc_damage_pp'):
                values = np.array([float(r[metric]) for r in selected])
                result.append(dict(model=key, condition=condition, metric=metric, seeds=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else None, minimum=float(values.min()), maximum=float(values.max())))
    M.write_csv(folder / 'between_seed_variation.csv', result)
    if not smoke:
        hist = list(csv.DictReader((ART / 'stage_h_distortion_matched/performance_paired_ci.csv').open()))
        current = list(csv.DictReader((folder / 'performance_paired_ci.csv').open()))
        comparison = []
        for key in KEYS:
            old = next((r for r in hist if r['model'] == key and r['condition'] == 'emotion_raw_matched' and (r['metric'] == 'accuracy')))
            new = next((r for r in current if r['model'] == key and r['condition'] == 'emotion_raw_matched' and (r['metric'] == 'accuracy')))
            valid = new['all_energy_matching_valid'] == 'True'
            assessment = 'INCONCLUSIVE' if not valid or float(new['excess_ci_low']) <= 0 <= float(new['excess_ci_high']) else 'SUPPORTS' if float(new['excess_ci_low']) > 0 else 'WEAKENS'
            comparison.append(dict(model=key, tweeteval_excess_pp=100 * float(old['excess_vs_random']), sst2_excess_pp=100 * float(new['excess_vs_random']), sst2_ci_low_pp=100 * float(new['excess_ci_low']), sst2_ci_high_pp=100 * float(new['excess_ci_high']), assessment=assessment, scope='directional exploratory generalization, not independent replication on TweetEval'))
        M.write_csv(folder / 'historical_comparison.csv', comparison)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'run', 'report'], required=True)
    args = parser.parse_args()
    args.output, args.batch_size, args.bootstrap, args.match_tolerance = (OUT, 8, 2000, 0.01)
    command = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), executable=sys.executable, argv=sys.argv)
    with (OUT / 'commands.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(command) + '\n')
    p, config, samples = audit_and_freeze()
    if args.phase == 'prepare':
        runtime_preflight(p)
    elif args.phase in ('smoke', 'run'):
        runtime_preflight(p)
        M.copy_reference = empty_reference
        M.execute(p, config, samples, args, args.phase == 'smoke')
        summarize(config, samples, args, args.phase == 'smoke')
    else:
        summarize(config, samples, args)
    verify_inputs()
    save('preservation_' + args.phase + '.json', dict(historical_inputs_unchanged=True, phase=args.phase, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('E2 phase complete:', args.phase, '; historical hashes unchanged.', flush=True)
if __name__ == '__main__':
    main()
