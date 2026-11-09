from release_paths import WORKSPACE, CODE_ROOT, source_path
'Stage K3B full frozen inference wrapper; reuses unmodified validated pilot engine.\nNo classifier predictions, outcomes, fitting or source-artifact changes.\n'
import os, sys, json, csv, time, traceback, datetime, re, msvcrt
from pathlib import Path
sys.dont_write_bytecode = True
R = WORKSPACE
K = R / 'artifacts/stage_k_amazon_transfer'
P = K / 'stage3a_pilot'
O = K / 'stage3b_full_inference'
sys.path.insert(0, str(source_path(P / 'stage_k_inference.py').parent))
import stage_k_inference as g
import numpy as np
now = lambda: datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=5))).isoformat()
LOG = None
SESSION = ''
CURRENT = None
RUNSTATE = {}
LAST_BATCH = []

def event(kind, **data):
    row = dict(timestamp=now(), session=SESSION, event=kind, **data)
    if LOG is not None:
        LOG.write(json.dumps(row, allow_nan=False) + '\n')
    if kind != 'batch_complete' and LOG is not None:
        LOG.flush()
    return row

def log(message):
    print(now() + ' ' + message, flush=True)
    kind = 'engine_progress'
    if message.startswith('Resume verified'):
        kind = 'resume_chunk'
    elif 'CUDA OOM' in message:
        kind = 'oom_retry'
    elif ': saved ' in message:
        kind = 'chunk_complete'
    event(kind, configuration=CURRENT, message=message)
    match = re.search(': saved (\\d+)/(\\d+); batch(\\d+)', message)
    if match:
        RUNSTATE.setdefault('progress', {})[CURRENT] = dict(completed=int(match[1]), total=int(match[2]), batch_size=int(match[3]))
        g.atomic_json(O / 'run_state.json', RUNSTATE)
        LOG.flush()
        os.fsync(LOG.fileno())

def preflight():
    state = g.js(P / 'state.json')
    assert state['complete'] and state['stage3b_ready'] and state['all_source_checks_passed'] and state['resume_verification_passed']
    for n, h in state['verified_code_sha256'].items():
        assert g.sha(P / n) == h, 'Pilot code changed: ' + n
    for n, h in state['verified_input_sha256'].items():
        assert g.sha(K / n) == h, 'Frozen input changed: ' + n
    assert g.js(P / 'resume_verification.json')['passed']
    for s in g.settings():
        name = s['source'] + '_' + s['model']
        v = g.js(P / 'source_checks' / (name + '.json'))
        assert v['passed'] and v['hook_equals_hidden_states'] and v['hook_leaves_logits_unchanged']
        for c in v['comparisons']:
            assert c['passed'] and c['violating_elements'] == 0 and (c['atol'] == g.TOL[c['feature']]['atol']) and (c['rtol'] == g.TOL[c['feature']]['rtol'])
        assert all((c['passed'] and c['max_abs_difference'] <= 1e-10 for c in v['cached_transform_reproduction']))
    old = g.js(K / 'transfer_compatibility.json')['verified_source_sha256']
    tracked = dict(old)
    for folder in [P]:
        for p in folder.rglob('*'):
            if p.is_file():
                tracked[str(p)] = g.sha(p)
    for name in ('source_model_inventory.csv', 'transfer_compatibility.json'):
        tracked[str(K / name)] = g.sha(K / name)
    for p, h in tracked.items():
        assert g.sha(p) == h, 'Frozen/source artifact changed: ' + p
    rows = g.verify_dataset()
    lengths = list(csv.DictReader((g.D / 'token_length_report.csv').open(encoding='utf-8')))
    assert [v['evaluation_id'] for v in lengths] == [r['evaluation_id'] for r in rows]
    assert all(('label' not in r and 'label_name' not in r for r in rows))
    assert all((max((int(v[n]) for n in ('sst2_0.5b', 'sst2_3b', 'tweeteval_0.5b', 'tweeteval_3b'))) <= 1024 for v in lengths))
    for s in g.settings():
        g.Transforms(s)
    import shutil
    free = shutil.disk_usage(O).free
    assert free > 2 * 1024 ** 3, 'Need at least2GiB free for caches and safety margin'
    g.atomic_json(O / 'preflight.json', dict(passed=True, manifest_sha256=g.dataset_sha(), dataset_revision=g.js(g.D / 'dataset_metadata.json')['dataset_revision'], source_checks_preserved=True, engine_code_sha256=state['verified_code_sha256'], asset_hashes=tracked, free_disk_bytes=free, original_source_artifacts_unchanged=True, execution_script_sha256=g.sha(Path(__file__))))
    log(f'Preflight passed:20,000 label-free inference rows; {len(tracked)} preserved source/pilot assets verified; all frozen transforms available')
    return (rows, lengths, tracked)

def validate_config(folder, rows, expected_lengths):
    name = folder.name
    s = next((s for s in g.settings() if s['source'] + '_' + s['model'] == name))
    width = s['identity']['width']
    protocol = g.js(folder / 'protocol.json')
    done = g.js(folder / 'completion.json')
    seen = []
    chunks = []
    token_sum = 0
    expected_shapes = {'margin': (), 'log_probabilities': (2,), 'pooled': (width,), 'text_token_counts': (), 'input_lengths': (), 'sentiment_only': (1,), 'raw': (9,), 'denoised': (9,), 'pca_rank7': (8,), 'random_rank7': (50, 8)}
    for offset in range(0, len(rows), 64):
        file = folder / 'chunks' / f'{offset:06d}.npz'
        marker = file.with_suffix('.json')
        md = g.js(marker)
        hashval = g.sha(file)
        block = rows[offset:offset + 64]
        ids = [r['evaluation_id'] for r in block]
        assert md['sha256'] == hashval and md['ids'] == ids and (md['offset'] == offset) and (md['n'] == len(ids)) and (md['signature_sha256'] == done['signature_sha256'])
        with np.load(file, allow_pickle=False) as z:
            assert set(z.files) == set(expected_shapes) | {'ids'} and z['ids'].tolist() == ids
            for key, shape in expected_shapes.items():
                assert z[key].shape == (len(ids), *shape) and np.isfinite(z[key]).all(), name + '/' + key
            assert z['pooled'].dtype == np.float32 and z['log_probabilities'].dtype == np.float32
            for key in ('margin', 'sentiment_only', 'raw', 'denoised', 'pca_rank7', 'random_rank7'):
                assert z[key].dtype == np.float64
            m = z['margin']
            assert np.array_equal(m, (z['log_probabilities'][:, 0] - z['log_probabilities'][:, 1]).astype(np.float64))
            for key in ('sentiment_only', 'raw', 'denoised', 'pca_rank7'):
                assert np.array_equal(z[key][:, 0], m)
            assert np.array_equal(z['random_rank7'][:, :, 0], np.broadcast_to(m[:, None], (len(ids), 50)))
            assert np.all(z['text_token_counts'] > 0) and np.all(z['input_lengths'] <= 1024)
            assert z['input_lengths'].tolist() == expected_lengths[offset:offset + len(ids)], 'Token lengths drifted'
            token_sum += int(z['input_lengths'].sum())
        seen += ids
        chunks.append(dict(offset=offset, n=len(ids), path=str(file.relative_to(O)), sha256=hashval, marker_sha256=g.sha(marker)))
    assert len(seen) == len(set(seen)) == 20000 and seen == [r['evaluation_id'] for r in rows] and (done['n'] == 20000)
    assert done['ids_sha256'] == g.digest('\n'.join(seen)) and done['signature_sha256'] == g.digest(json.dumps(protocol, sort_keys=True))
    assert protocol == g.js(P / 'pilot_features' / name / 'protocol.json'), 'Pilot/full extraction configuration drift'
    return dict(configuration=name, passed=True, n=20000, unique_ids=20000, missing_ids=0, duplicate_ids=0, chunks_verified=len(chunks), input_tokens=token_sum, pooled_shape=[20000, width], raw_emotion_projection_shape=[20000, 8], denoised_emotion_projection_shape=[20000, 8], pca_projection_shape=[20000, 7], random_projection_shape=[20000, 50, 7], stored_classifier_matrix_shapes={'sentiment_only': [20000, 1], 'raw': [20000, 9], 'denoised': [20000, 9], 'pca_rank7': [20000, 8], 'random_rank7': [20000, 50, 8]}, emotion_order=g.EMOTIONS, random_seeds=list(range(2000, 2050)), all_finite=True, labels_in_cache=False, margin_identity_verified=True, token_lengths_verified=True, signature_sha256=done['signature_sha256'], protocol_path=str((folder / 'protocol.json').relative_to(O)), completion_path=str((folder / 'completion.json').relative_to(O)), chunks=chunks)

def report(rows, runtimes, checks, wall, tracked, start_timestamp):
    assert len(checks) == 4 and all((v['passed'] for v in checks)) and (sum((c['n'] for c in checks)) == 80000)
    for p, h in tracked.items():
        assert g.sha(p) == h, 'Preserved source or pilot changed: ' + p
    dims = {k: v for k, v in checks[0].items() if k.endswith('projection_shape')}
    manifest = dict(dataset_revision=g.js(g.D / 'dataset_metadata.json')['dataset_revision'], manifest_sha256=g.dataset_sha(), evaluation_ids=[r['evaluation_id'] for r in rows], evaluation_ids_sha256=g.digest('\n'.join((r['evaluation_id'] for r in rows))), ids_per_configuration=20000, total_evaluations=80000, configurations=checks, emotion_order=g.EMOTIONS, feature_definition='Classifier matrices contain margin first; raw/denoised projection columns1:9, PCA/random columns1:8; pooled h float32, projections float64; labels excluded.', no_predictions_or_outcome_metrics=True)
    g.atomic_json(O / 'feature_manifest.json', manifest)
    integrity = dict(passed=True, configurations=[{k: v for k, v in c.items() if k != 'chunks'} for c in checks], total_valid_evaluations=80000, all_chunk_hashes_verified=True, source_and_pilot_hashes_unchanged=True, code_and_transforms_frozen=True, classifier_pipelines_compatible=216, ground_truth_not_used_for_inference=True, no_target_sentiment_metrics=True)
    g.atomic_json(O / 'integrity_checks.json', integrity)
    fields = [k for k in runtimes[0] if k != 'classifier_check']
    with (O / 'runtime_summary.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(({k: r[k] for k in fields} for r in runtimes))
    g.atomic_json(O / 'total_runtime.json', dict(start_timestamp=start_timestamp, finished_timestamp=now(), session_wall_seconds=wall, total_forward_and_tokenization_seconds=sum((b['forward_and_tokenization_seconds'] for b in runtimes)), total_configuration_wall_seconds=sum((b['configuration_wall_including_validation_seconds'] for b in runtimes)), unique_model_load_seconds=sum((next((b['model_load_seconds'] for b in runtimes if b['configuration'].endswith(k))) for k in ('0.5b', '3b'))), valid_completed_evaluations=80000, pilot_token_based_estimate_seconds=g.js(P / 'runtime_estimate.json')['total_token_based_seconds'], no_performance_outcomes=True))
    (O / 'classifier_compatibility_report.md').write_text('# Classifier input readiness\n\nAll216 frozen source-trained pipelines/scalers verified,54 per configuration. Their full20,000-row input matrices were checked for correct feature dimensions, finite values, train-fitted scaler state, classifier/hash identity and C. Scaler.transform was called only to verify finite compatible inputs; no fit,predict,decision_function,accuracy or label comparisons. Margin first; raw/denoised8 named emotion coordinates, PCA7 or random7 coordinates. Source prompts/bases/scalers remain separate; no substitutions. Complete raw matrices shape20,000x9; rank7 inputs20,000x8; random tensor20,000x50x8 indexed by seeds2000..2049.\n', encoding='utf-8')
    lines = ['# Stage K3B full inference complete', '', f'20,000 reviews per each4 configurations;80,000 valid unique configuration/example outputs. All source/pilot assets unchanged; chunk hashes and model/transform/code signatures verified. No predictions, sentiment outcomes, fitting or tuning. Session wall-clock {wall / 60:.2f} minutes.', '', '|Configuration|Completed|Forward seconds|Config wall seconds|Input tok/sec|Peak allocated GPU GiB|Chunks|', '|---|---:|---:|---:|---:|---:|---:|']
    for b in runtimes:
        lines.append(f'|{b['configuration']}|{b['n']}|{b['forward_and_tokenization_seconds']:.2f}|{b['configuration_wall_including_validation_seconds']:.2f}|{b['input_tokens_per_second']:.2f}|{b['peak_gpu_allocated_bytes'] / 2 ** 30:.3f}|{b['verified_chunks']}|')
    lines += ['', 'Runtime scope must be determined from the current run logs; resumed checks are not full inference timings.', 'Source benchmark load time recorded per model and shared across source prompts; do not sum duplicate per-source model load entries. Resume/failure events recorded in execution_log.jsonl; existing completed chunks are reused only after exact hash/ID/config verification. Pilot32 chunks are not reused: full caches have deterministic manifest ordering and independent64-row chunk layout.', 'Stage4 evaluation is technically ready; NOT executed. No classifier predictions/outcome metrics computed.', 'Artifacts: features/,feature_manifest.json,execution_log.jsonl,runtime_summary.csv,total_runtime.json,integrity_checks.json,classifier_compatibility_report.md,stage3b_report.md,run_stage3b_full.py,verify_stage3b_full.py.']
    (O / 'stage3b_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    g.atomic_json(O / 'run_state.json', dict(complete=True, total_valid_evaluations=80000, completed_configurations=[b['configuration'] for b in runtimes], stage4_ready=True, no_outcome_evaluation=True))

def main():
    global LOG, SESSION, CURRENT, RUNSTATE, LAST_BATCH
    O.mkdir(parents=True, exist_ok=True)
    lock = (O / 'execution.lock').open('a+b')
    lock.seek(0)
    if lock.read(1) == b'':
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    SESSION = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(os.getpid())
    LOG = (O / 'execution_log.jsonl').open('a', encoding='utf-8', buffering=16384)
    g.log = log
    start = time.perf_counter()
    start_timestamp = now()
    RUNSTATE = dict(complete=False, session=SESSION, progress={}, started=start_timestamp)
    runtimes = []
    checks = []
    try:
        event('session_start', pid=os.getpid(), batch_size=4, chunk_size=64, stage='3B', no_sentiment_metrics=True)
        rows, lengths, tracked = preflight()
        g.atomic_json(O / 'run_state.json', RUNSTATE)
        for key in ('0.5b', '3b'):
            e = g.Extractor(key)
            event('model_loaded', model=key, seconds=e.load_seconds, device=g.torch.cuda.get_device_name(0), dtype='float16', quantization=False)
            original_batch = e.batch

            def measured_batch(batch, prompt, verify=False):
                global LAST_BATCH
                assert all(('label' not in r and 'label_name' not in r for r in batch))
                LAST_BATCH = [r['evaluation_id'] for r in batch]
                t = time.perf_counter()
                try:
                    result = original_batch(batch, prompt, verify=verify)
                except Exception as exc:
                    event('batch_failure', configuration=CURRENT, ids=LAST_BATCH, error=repr(exc), traceback=traceback.format_exc(), recoverable_oom=isinstance(exc, g.torch.cuda.OutOfMemoryError))
                    raise
                event('batch_complete', configuration=CURRENT, ids=LAST_BATCH, n=len(batch), input_tokens=int(result['input_lengths'].sum()), seconds=time.perf_counter() - t)
                return result
            e.batch = measured_batch
            try:
                for s in g.settings():
                    if s['model'] != key:
                        continue
                    CURRENT = s['source'] + '_' + key
                    folder = O / 'features' / CURRENT
                    existing = len(list((folder / 'chunks').glob('*.json'))) if (folder / 'chunks').exists() else 0
                    event('configuration_start', configuration=CURRENT, already_completed_chunks=existing)
                    config_started = time.perf_counter()
                    arrays, b = g.run_chunks(e, s, rows, folder, batch_size=4, chunk_size=64)
                    compatible = g.verify_classifiers(s, {n: arrays[n] for n in ('sentiment_only', 'raw', 'denoised', 'pca_rank7', 'random_rank7')})
                    assert compatible['count'] == 54 and compatible['passed']
                    check = validate_config(folder, rows, [int(v[CURRENT]) for v in lengths])
                    checks.append(check)
                    b.update(configuration_wall_including_validation_seconds=time.perf_counter() - config_started, verified_chunks=check['chunks_verified'], resumed_chunks_at_start=existing, unique_valid_examples=20000, unresolved_failed_examples=0, reviews_per_second=b['newly_processed'] / b['forward_and_tokenization_seconds'] if b['forward_and_tokenization_seconds'] else None, all_rows_scaler_compatibility_passed=True)
                    runtimes.append(b)
                    event('configuration_complete', configuration=CURRENT, n=20000, verified_chunks=check['chunks_verified'], benchmark={k: v for k, v in b.items() if k != 'classifier_check'})
                    del arrays
                    g.gc.collect()
            finally:
                e.close()
                del original_batch, e
                g.gc.collect()
        wall = time.perf_counter() - start
        report(rows, runtimes, checks, wall, tracked, start_timestamp)
        event('session_complete', wall_seconds=wall, valid_evaluations=80000, stage4_ready=True, no_sentiment_metrics=True)
        LOG.flush()
        os.fsync(LOG.fileno())
        log(f'FULL INFERENCE COMPLETE:80,000 outputs; wall{wall / 60:.2f}min; all integrity/216 classifier checks passed; Stage4 not run')
    except BaseException as exc:
        event('session_failure', configuration=CURRENT, ids=LAST_BATCH, error=repr(exc), traceback=traceback.format_exc())
        RUNSTATE.update(complete=False, error=repr(exc), last_configuration=CURRENT, last_batch_ids=LAST_BATCH)
        g.atomic_json(O / 'run_state.json', RUNSTATE)
        LOG.flush()
        os.fsync(LOG.fileno())
        raise
    finally:
        LOG.close()
        LOG = None
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
if __name__ == '__main__':
    main()
