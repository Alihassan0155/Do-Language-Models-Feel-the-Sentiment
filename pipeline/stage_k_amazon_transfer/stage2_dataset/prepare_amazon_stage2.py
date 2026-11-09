from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import os, sys, json, csv, hashlib, unicodedata, time, subprocess
from pathlib import Path
sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
R = WORKSPACE
K = R / 'artifacts/stage_k_amazon_transfer'
O = K / 'stage2_dataset'
DATASET = 'mteb/amazon_polarity'
SEED = 42
N = 10000

def log(s):
    print(time.strftime('%H:%M:%S') + ' ' + s, flush=True)

def js(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    p = source_path(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def digest(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()

def norm(t):
    return ' '.join(unicodedata.normalize('NFKC', t).casefold().split())

def save(name, data):
    (O / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def md(name, text):
    (O / name).write_text(text, encoding='utf-8')

def source_sets():
    paths = []
    for folder in [R / 'artifacts/stage_i_sst2/splits', R / 'artifacts/stage_i_sentiment_correction/reused_inputs', R / 'artifacts/stage_e/data/sst2', R / 'artifacts/stage_e/data/tweet_eval_sentiment']:
        for split in ('train', 'validation', 'test'):
            paths.append(folder / (split + '.jsonl'))
    paths.append(R / 'artifacts/stage_d/examples.jsonl')
    exact = set()
    normalized = set()
    records = []
    for p in paths:
        if not p.exists():
            records.append(dict(path=str(p), available=False))
            continue
        n = 0
        for line in p.open(encoding='utf-8-sig'):
            r = json.loads(line)
            t = r['text']
            exact.add(digest(t))
            normalized.add(digest(norm(t)))
            n += 1
        records.append(dict(path=str(p), available=True, rows=n, sha256=sha(p)))
    return (exact, normalized, records)

def run_sampling(path, source_exact, source_norm, progress=True):
    rng = np.random.default_rng(SEED)
    reservoir = {0: [], 1: []}
    eligible = [0, 0]
    original = [0, 0]
    seen = set()
    excluded = dict(source_exact=0, source_normalized_only=0, amazon_normalized_duplicate=0, empty_normalized_text=0)
    idx = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=4096, columns=['text', 'label', 'label_text']):
        for r in batch.to_pylist():
            t = r['text']
            label = r['label']
            assert isinstance(t, str) and type(label) == int and (label in (0, 1))
            assert r['label_text'] == ('negative' if label == 0 else 'positive'), f'Label mapping mismatch row{idx}'
            original[label] += 1
            th = digest(t)
            nt = norm(t)
            nh = digest(nt)
            reason = None
            if th in source_exact:
                reason = 'source_exact'
            elif nh in source_norm:
                reason = 'source_normalized_only'
            elif not nt:
                reason = 'empty_normalized_text'
            elif nh in seen:
                reason = 'amazon_normalized_duplicate'
            if reason:
                excluded[reason] += 1
            else:
                seen.add(nh)
                eligible[label] += 1
                entry = dict(evaluation_id=f'amazon_polarity:{REV}:test:{idx}', dataset_row_index=idx, text=t, label=label, label_name=r['label_text'], character_count=len(t), text_sha256=th)
                if eligible[label] <= N:
                    reservoir[label].append(entry)
                else:
                    slot = int(rng.integers(eligible[label]))
                    if slot < N:
                        reservoir[label][slot] = entry
            idx += 1
        if progress and idx % 40960 == 0:
            log(f'Scan {idx:,}/{pf.metadata.num_rows:,}; eligible {eligible}')
    assert idx == pf.metadata.num_rows and idx == 400000, f'Unexpected test count{idx}'
    assert min(eligible) >= N, f'Insufficient eligible examples: {eligible}'
    out = sorted(reservoir[0] + reservoir[1], key=lambda r: r['dataset_row_index'])
    assert len(out) == 20000 and [sum((r['label'] == i for r in out)) for i in (0, 1)] == [N, N]
    assert len({r['evaluation_id'] for r in out}) == len(out) and len({r['dataset_row_index'] for r in out}) == len(out)
    assert len({digest(norm(r['text'])) for r in out}) == len(out)
    assert all((r['text_sha256'] not in source_exact and digest(norm(r['text'])) not in source_norm for r in out))
    return (out, dict(original_counts=original, eligible_counts=eligible, exclusions=excluded, total=idx))

def manifest_bytes(rows):
    return ''.join((json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows)).encode('utf-8')
O.mkdir(parents=True, exist_ok=True)
assert not (O / 'amazon_test_20000.jsonl').exists(), 'Refusing to overwrite frozen sample'
revision_path = O / 'dataset_revision.json'
if revision_path.exists():
    REV = js(revision_path)['revision']
else:
    REV = HfApi().dataset_info(DATASET).sha
    save('dataset_revision.json', dict(dataset=DATASET, revision=REV))
assert len(REV) == 40
log('Pinned dataset ' + REV + '; downloading TEST file only')
parquet = Path(hf_hub_download(DATASET, 'data/test-00000-of-00001.parquet', repo_type='dataset', revision=REV, cache_dir=str(O / 'download_cache')))
pf = pq.ParquetFile(parquet)
schema = {f.name: str(f.type) for f in pf.schema_arrow}
assert set(schema) == {'text', 'label', 'label_text'}, f'Schema mismatch{schema}'
assert schema['text'] in ['string', 'large_string'] and schema['label'] == 'int64' and (schema['label_text'] in ['string', 'large_string'])
assert pf.metadata.num_rows == 400000
log('Verified test schema ' + str(schema))
source_exact, source_norm, sources = source_sets()
save('source_overlap_inventory.json', sources)
log(f'Source inventory: {len(source_exact):,} exact / {len(source_norm):,} normalized unique texts')
rows, audit = run_sampling(parquet, source_exact, source_norm)
data = manifest_bytes(rows)
log('Repeating complete seeded sampling for reproducibility')
repeat, repeat_audit = run_sampling(parquet, source_exact, source_norm)
assert manifest_bytes(repeat) == data and repeat_audit == audit
selected = {r['dataset_row_index']: r for r in rows}
idx = 0
checked = 0
for batch in pf.iter_batches(batch_size=8192):
    for original in batch.to_pylist():
        if idx in selected:
            r = selected[idx]
            assert original['text'] == r['text'] and original['label'] == r['label'] and (original['label_text'] == r['label_name'])
            assert digest(original['text']) == r['text_sha256']
            checked += 1
        idx += 1
assert checked == 20000
(O / 'amazon_test_20000.jsonl').write_bytes(data)
manifest_sha = sha(O / 'amazon_test_20000.jsonl')
assert manifest_sha == hashlib.sha256(data).hexdigest()
log('Sample integrity passed; checking token lengths (no model loaded)')
from transformers import AutoTokenizer
settings = js(K / 'transfer_compatibility.json')['source_settings']
lengths = {}
stats = {}
tokenizer_checks = {}
mask_counts = {}
for s in settings:
    cfg = s['identity']
    name = s['source'] + '_' + s['model']
    snap = Path(s['snapshot'])
    tok = AutoTokenizer.from_pretrained(str(snap), use_fast=True, local_files_only=True, revision=model_revision(str(snap)))
    assert digest(tok.backend_tokenizer.to_str()) == cfg['tokenizer_sha256']
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    assert tok.encode('positive', add_special_tokens=False) == [30487] and tok.encode('negative', add_special_tokens=False) == [42224]
    skeleton = tok.apply_chat_template([{'role': 'user', 'content': s['prompt'].format(text='__STAGE_D_TEXT_SPAN_7ef8492c__')}], tokenize=False, add_generation_prompt=True)
    prefix, suffix = skeleton.split('__STAGE_D_TEXT_SPAN_7ef8492c__')
    values = []
    counts = []
    for start in range(0, len(rows), 256):
        subset = rows[start:start + 256]
        rendered = [prefix + r['text'] + suffix for r in subset]
        batch = tok(rendered, add_special_tokens=False, truncation=False, padding=False, return_offsets_mapping=True)
        for r, ids, offsets in zip(subset, batch['input_ids'], batch['offset_mapping']):
            a = len(prefix)
            b = a + len(r['text'])
            count = sum((x < b and y > a and (y > x) and (tid not in tok.all_special_ids) for tid, (x, y) in zip(ids, offsets)))
            assert count > 0
            counts.append(count)
            values.append(len(ids))
        if (start + 256) % 2048 == 0:
            log(f'Token lengths {name}: {min(start + 256, len(rows)):,}/20,000')
    arr = np.array(values, dtype=np.int64)
    lengths[name] = values
    mask_counts[name] = counts
    stats[name] = dict(min=int(arr.min()), mean=float(arr.mean()), median=float(np.median(arr)), p90=float(np.percentile(arr, 90)), p95=float(np.percentile(arr, 95)), p99=float(np.percentile(arr, 99)), max=int(arr.max()), total_input_tokens=int(arr.sum()), source_limit=int(s['source_effective_max_length']), exceed_source_limit=int((arr > s['source_effective_max_length']).sum()), model_context_limit=int(s['model_max_position_embeddings']), exceed_model_context=int((arr > s['model_max_position_embeddings']).sum()))
    tokenizer_checks[name] = dict(revision=cfg['checkpoint_commit'], tokenizer_sha256=cfg['tokenizer_sha256'], prompt=s['prompt'], chat_skeleton=skeleton, snapshot=str(snap), padding_side='left', truncation=False, add_special_tokens=False)
    log(name + ' ' + str(stats[name]))
with (O / 'token_length_report.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['evaluation_id', 'dataset_row_index', *lengths])
    writer.writeheader()
    for i, r in enumerate(rows):
        writer.writerow(dict(evaluation_id=r['evaluation_id'], dataset_row_index=r['dataset_row_index'], **{k: v[i] for k, v in lengths.items()}))
with (O / 'token_length_summary.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['configuration', *next(iter(stats.values()))])
    writer.writeheader()
    writer.writerows((dict(configuration=k, **v) for k, v in stats.items()))
affected = sorted((r['dataset_row_index'] for i, r in enumerate(rows) if any((v[i] > 1024 for v in lengths.values()))))
ready = not affected
try:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv'], text=True).strip()
except Exception as e:
    gpu = 'GPU query unavailable: ' + str(e)
metadata = dict(dataset_identifier=DATASET, dataset_revision=REV, split='test', schema=schema, full_test_size=400000, label_mapping={'0': 'negative', '1': 'positive'}, sampling_seed=SEED, sampling_algorithm='Algorithm R stratified reservoir over ascending official test rows; one numpy default_rng42; fill first10000 eligible/class, then uniform slot0..eligible_count-1 for that class; final output sorted row index', original_per_class_counts=dict(zip(['negative', 'positive'], audit['original_counts'])), eligible_per_class_counts=dict(zip(['negative', 'positive'], audit['eligible_counts'])), final_per_class_counts={'negative': 10000, 'positive': 10000}, exclusions=audit['exclusions'], exclusion_rule='Exact source match first, normalized-only source match second, empty normalized text third, duplicate eligible Amazon normalized text fourth; earliest eligible original row retained globally; duplicates removed before sampling, never based on lengths', normalization='NFKC then casefold then whitespace split/join; detection only', source_manifests=sources, manifest_sha256=manifest_sha, parquet_sha256=sha(parquet), parquet_path=str(parquet), reproducibility_checks={'independent_same_seed_full_rescan_byte_identical': True, 'selected_rows_reverified_against_original_test': 20000, 'unique_ids_rows_normalized_text': True, 'zero_selected_source_overlaps': True}, token_statistics=stats, tokenizer_checks=tokenizer_checks, unique_reviews_exceeding_source_limit=len(affected), overlength_original_row_indices=affected, sample_frozen=True, inference_ready=ready, no_inference=True, no_classifier_training=True, no_target_performance_metrics=True, protocol_amendment='User Stage2 request supersedes Stage1 2000 sample:20000, sorted output, duplicate exclusions before reservoir sampling; source rendering/model/feature definitions unchanged')
save('dataset_metadata.json', metadata)
md('sampling_audit.md', '# Amazon test sampling audit\n\nPinned ' + REV + '. Official test400000; schema ' + json.dumps(schema) + '. Test Parquet only downloaded; no training split.\n\n' + metadata['sampling_algorithm'] + '. Seed42,10000/class, sorted original row index. ' + metadata['exclusion_rule'] + '. Normalization ' + metadata['normalization'] + '.\n\nCounts/exclusions: ' + json.dumps(audit) + '. Selected20000,10000/class. Selected overlap zero; exact text and official labels checked independently against pinned file. Full seed42 rescan produces byte-identical manifest. SHA256 ' + manifest_sha + '.\n\nSource dataset manifests cover Stage I SST train/validation/test, all available Stage I-A Tweet train/validation/test, prior Stage E SST/Tweet splits, and Stage D prior-used sentiment examples. Full manifest inventory/availability/hashes saved in source_overlap_inventory.json. Repeated source copies deduplicated in lookup sets. Amazon duplicate counts are among otherwise eligible rows, with earliest normalized text occurrence retained; source exclusion categories mutually exclusive. No difficulty/length/confidence filtering. Text preserved verbatim; no source predictions inspected. Exact/normalized matching cannot establish absence from Qwen pretraining.\n\nThis sample is frozen for integrity/reproducibility; inference readiness is ' + str(ready) + '. Length violations do not cause replacement.\n')
summary_table = '|Configuration|Min|Mean|Median|P90|P95|P99|Max|>1024|>Context|Tokens|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(('|' + k + '|' + '|'.join((str(v[q]) for q in ('min', 'mean', 'median', 'p90', 'p95', 'p99', 'max', 'exceed_source_limit', 'exceed_model_context', 'total_input_tokens'))) + '|' for k, v in stats.items()))
handling = 'No sampled review exceeds the source limit. Stage3 length preflight passed; inference requires a separate Stage3 instruction.' if ready else f'{len(affected)} distinct sampled reviews exceed original1024-token source limits. Stage3 BLOCKED until a handling policy is approved before outcomes. Options: retain full text and raise the input limit equally across all four settings after context/memory checks, explicitly documenting departure from source protocol; or apply a predeclared common text truncation policy to every method and document changed target input. Do not replace reviews or silently alter settings. Original manifest remains unchanged. Neither option implemented.'
md('inference_workload_estimate.md', '# Stage3 workload estimate\n\n' + summary_table + '\n\nTotal ' + str(sum((v['total_input_tokens'] for v in stats.values()))) + ' unpadded input tokens;80000 model-input evaluations (20000 x2 models x2 source prompts). Same forward yields margin at final attended assistant-prefix token plus selected block14/18 pooled text states. No run per classifier: cache outputs once/configuration and reuse216 heads.\n\n' + handling + '\n\nHardware snapshot:\n```text\n' + gpu + '\n```\n\nFP16 frozen models, no quantization, selected-layer hook, no all-layer hidden output, inference_mode/use_cacheFalse. Start batch1, length bucket without altering IDs, increase only after a memory pilot. Initial practical candidate batch range1-4 for3B and1-8 for0.5B is a proposal, not verified capacity. Full vocabulary151936 FP16 logits at1024 tokens ~0.29GiB/example; float32 log_softmax only at final attended position. Weight storage alone3B ~5.75GiB, additional activations/workspace required. Pooled20k float32 cache~68.4MiB for0.5B and156.25MiB for3B per source prompt; all four~449.2MiB plus margins/manifests. Use GPU model one at a time, bounded batches. Stage I-SST has SQLite per-row scoring completion and memmap pooled cache/OOM retry; Amazon-specific cache/resume adapter still needs implementation, cannot reuse source checkpoints as target data. No performance benchmark run; Stage3 requires a small throughput/memory pilot, no reliable runtime estimate yet.\n')
md('stage2_report.md', '# Stage K Stage2 report\n\nRevision ' + REV + '; official test400000; schema ' + json.dumps(schema) + '. Selected10000 negative+10000 positive=20000. Exclusions ' + json.dumps(audit['exclusions']) + '; eligible ' + str(audit['eligible_counts']) + '.\n\n' + summary_table + '\n\n' + handling + '\n\nManifest SHA256 `' + manifest_sha + '`. All integrity/reproducibility assertions passed. Files: amazon_test_20000.jsonl,dataset_metadata.json,sampling_audit.md,token_length_report.csv,token_length_summary.csv,inference_workload_estimate.md,stage2_report.md,prepare_amazon_stage2.py,source_overlap_inventory.json,dataset_revision.json. No Qwen inference/predictions, PCA/scaler fitting, classifier retraining or target performance evaluation. All old B-I artifacts preserved.\n')
assert sha(O / 'amazon_test_20000.jsonl') == manifest_sha
for s in sources:
    if s['available']:
        assert sha(Path(s['path'])) == s['sha256'], 'Source file changed'
log('COMPLETE ' + json.dumps(dict(revision=REV, manifest_sha256=manifest_sha, counts=audit, stats=stats, inference_ready=ready)))
