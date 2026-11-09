"""Stage3A:32-review inference pilot only; never evaluate sentiment outcomes."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import sys, json, csv, time
from pathlib import Path
sys.dont_write_bytecode = True
from stage_k_inference import *

def source_reference(s, n=8):
    source = s['source']
    key = s['model']
    A = R / 'artifacts'
    if source == 'sst2':
        folder = A / 'stage_i_sst2' / key
        rs = []
        with (A / 'stage_i_sst2/splits/train.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                rs.append({'evaluation_id': r['id'], 'text': r['text'], 'text_sha256': r['text_sha256']})
                if len(rs) == n:
                    break
        h = np.array(np.load(folder / 'pooled.float32.npy', mmap_mode='r')[:n])
        path = folder / 'scores.sqlite3'
        wal = Path(str(path) + '-wal')
        assert not wal.exists() or wal.stat().st_size == 0, 'Source WAL uncheckpointed'
        db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
        values = db.execute('select row_index,id,margin from scores where row_index < ? order by row_index', (n,)).fetchall()
        db.close()
        assert [v[1] for v in values] == [r['evaluation_id'] for r in rs]
        m = np.array([v[2] for v in values], dtype=np.float64)
        cached = {name: np.array(np.load(folder / 'features' / (name + '.npy'), mmap_mode='r')[:n]) for name in ('raw', 'denoised', 'pca_rank7')}
    else:
        IB = A / 'stage_i_emotion_guided_correction'
        folder = A / 'stage_i_tweeteval_matched_controls' / key
        chosen = list(csv.DictReader((IB / 'train_ids.csv').open(encoding='utf-8')))[:n]
        indices = [int(r['source_index']) for r in chosen]
        allrows = []
        with (A / 'stage_i_sentiment_correction/reused_inputs/train.jsonl').open(encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in indices:
                    r = json.loads(line)
                    allrows.append((i, {'evaluation_id': r['id'], 'text': r['text'], 'text_sha256': r['text_sha256']}))
                if i >= max(indices):
                    break
        rs = [dict(allrows)[i] for i in indices]
        assert [r['evaluation_id'] for r in rs] == [r['id'] for r in chosen]
        h = np.array(np.load(A / 'stage_e/hidden' / key / 'tweet_eval_sentiment/train/pooled.float32.npy', mmap_mode='r')[indices])
        m = np.array(np.load(IB / key / 'train/margin.float64.npy', mmap_mode='r')[:n])
        cached = {name: np.column_stack([m, np.load(IB / key / 'train' / (name + '_emotion_scores.float64.npy'), mmap_mode='r')[:n]]) for name in ('raw', 'denoised')}
        cached['pca_rank7'] = np.array(np.load(folder / 'features/pca_rank7.npy', mmap_mode='r')[:n])
    cached['random_rank7'] = np.stack([np.array(np.load(folder / 'features' / f'random_rank7_seed_{seed}.npy', mmap_mode='r')[:n]) for seed in range(2000, 2050)], axis=1)
    return (rs, m, h, cached)

def diff_report(name, fresh, old):
    assert fresh.shape == old.shape and np.isfinite(fresh).all() and np.isfinite(old).all()
    delta = np.abs(fresh.astype(np.float64) - old.astype(np.float64))
    tol = TOL[name]
    allowed = tol['atol'] + tol['rtol'] * np.abs(old)
    return dict(feature=name, shape=list(old.shape), max_abs_difference=float(delta.max()), mean_abs_difference=float(delta.mean()), atol=tol['atol'], rtol=tol['rtol'], violating_elements=int((delta > allowed).sum()), passed=bool((delta <= allowed).all()), relative_l2_difference=float(np.linalg.norm(delta) / (np.linalg.norm(old) + 1e-30)))

def compatibility(e, s):
    rows, m, h, cached = source_reference(s)
    tf = Transforms(s)
    from_cached = tf.apply(m, h)
    exact = []
    for name in cached:
        delta = np.abs(from_cached[name] - cached[name])
        assert np.allclose(from_cached[name], cached[name], atol=1e-10, rtol=0), 'Cached transform changed ' + name
        exact.append(dict(feature=name, max_abs_difference=float(delta.max()), mean_abs_difference=float(delta.mean()), passed=True, atol=1e-10))
    log('Source extraction verification ' + s['source'] + '/' + s['model'] + ' on8 cached training rows')
    fresh = e.batch(rows, s['prompt'], verify=True)
    projected = tf.apply(fresh['margin'], fresh['pooled'])
    reports = [diff_report('margin', fresh['margin'], m), diff_report('pooled', fresh['pooled'], h)]
    for name in cached:
        reports.append(diff_report(name, projected[name], cached[name]))
    result = dict(source=s['source'], model=s['model'], n=8, ids=[r['evaluation_id'] for r in rows], comparisons=reports, cached_transform_reproduction=exact, hook_equals_hidden_states=True, hook_leaves_logits_unchanged=True, passed=all((v['passed'] for v in reports)))
    atomic_json(P / 'source_checks' / (s['source'] + '_' + s['model'] + '.json'), result)
    for item in reports:
        log(f'{s['source']}/{s['model']} {item['feature']}: maxdiff{item['max_abs_difference']:.6g} meandiff{item['mean_abs_difference']:.6g} pass{item['passed']}')
    assert result['passed'], 'Source extraction mismatch; investigate before Amazon inference'
    return result

def select_pilot(rows):
    lens = list(csv.DictReader((D / 'token_length_report.csv').open(encoding='utf-8')))
    assert [r['evaluation_id'] for r in rows] == [r['evaluation_id'] for r in lens]
    order = sorted(range(len(rows)), key=lambda i: (int(lens[i]['sst2_0.5b']), rows[i]['evaluation_id']))
    rng = np.random.default_rng(42)
    selected = []
    strata = {}
    for name, ids, n in [('short', order[:4000], 10), ('medium', order[4000:16000], 12), ('long', order[16000:], 10)]:
        chosen = rng.choice(ids, size=n, replace=False).tolist()
        selected += chosen
        for i in chosen:
            strata[rows[i]['evaluation_id']] = name
    selected.sort(key=lambda i: rows[i]['dataset_row_index'])
    out = [rows[i] for i in selected]
    assert len(out) == 32 and len({r['evaluation_id'] for r in out}) == 32
    spec = dict(seed=42, selection='Sort by SST token length then ID; lower20%=short10, middle60%=medium12, upper20%=long10; seeded uniform no-replacement within strata; final original row order', manifest_sha256=dataset_sha(), ids=[r['evaluation_id'] for r in out], strata=strata, ground_truth_used_for_selection=False)
    if (P / 'pilot_ids.json').exists():
        assert js(P / 'pilot_ids.json') == spec
    else:
        atomic_json(P / 'pilot_ids.json', spec)
    return out

def main():
    P.mkdir(parents=True, exist_ok=True)
    atomic_json(P / 'extraction_tolerances.json', dict(tolerances=TOL, declared_before_forward_passes=True, rationale='FP16 CUDA SDPA batch/padding changes may alter final logits by several half-precision ULPs; verify per-element atol+rtol, report actual differences. Cached float64 projection reproduction must match atol1e-10. Reject out-of-tolerance rather than change thresholds.'))
    rows = verify_dataset()
    pilot = select_pilot(rows)
    bench = []
    checks = []
    try:
        for key in ('0.5b', '3b'):
            e = Extractor(key)
            try:
                for s in settings():
                    if s['model'] != key:
                        continue
                    checks.append(compatibility(e, s))
                    features, b = run_chunks(e, s, pilot, P / 'pilot_features' / (s['source'] + '_' + key), batch_size=4, chunk_size=8)
                    bench.append(b)
                    atomic_json(P / 'state.json', dict(complete=False, completed_configurations=[v['configuration'] for v in bench], source_checks_passed=True, no_outcome_evaluation=True))
            finally:
                e.close()
                del e
        assert len(bench) == 4 and all((b['n'] == 32 and b['complete'] and b['classifier_check']['passed'] for b in bench))
        fields = [k for k in bench[0] if k != 'classifier_check']
        with (P / 'benchmark_results.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(({k: b[k] for k in fields} for b in bench))
        estimates = []
        tokenstats = js(D / 'dataset_metadata.json')['token_statistics']
        for b in bench:
            speed = b['input_tokens_per_second']
            assert speed and speed > 0
            estimates.append(dict(configuration=b['configuration'], full_tokens=tokenstats[b['configuration']]['total_input_tokens'], token_based_seconds=tokenstats[b['configuration']]['total_input_tokens'] / speed, example_based_seconds=b['seconds_per_example'] * 20000))
        loading = sum((next((b['model_load_seconds'] for b in bench if b['configuration'].endswith(key))) for key in ('0.5b', '3b')))
        estimate = dict(per_configuration=estimates, unique_model_load_seconds=loading, total_token_based_seconds=sum((v['token_based_seconds'] for v in estimates)) + loading, total_example_based_seconds=sum((v['example_based_seconds'] for v in estimates)) + loading, limitation='Pilot32 length-stratified reviews, batch4; estimates exclude full-run classifier/hash/cache processing overhead; thermal/storage/batch-length variation may change throughput. Not a confidence interval or guaranteed runtime.')
        atomic_json(P / 'runtime_estimate.json', estimate)
        compatibility_lines = ['# Extraction compatibility', '', 'Frozen source training examples only;8 per configuration. No Amazon labels, predictions or outcome metrics. All features compared to original cached arrays. Exact Stage D helper reused via AST without executing old experiment imports/main. FP16 SDPA CUDA preserved. Margin subtracts float32 log probabilities before casting to float64. Pooled hidden states float32; projections float64 using saved unit vectors/PCA/random bases.', '', '|Source/model|Feature|Max abs diff|Mean abs diff|Passed|', '|---|---|---:|---:|---|']
        for c in checks:
            for v in c['comparisons']:
                compatibility_lines.append(f'|{c['source']}/{c['model']}|{v['feature']}|{v['max_abs_difference']:.9g}|{v['mean_abs_difference']:.9g}|{v['passed']}|')
        compatibility_lines += ['', 'Predeclared tolerance: ' + json.dumps(TOL) + '. Per-element allowable difference atol+rtol*abs(reference); stored raw statistics include relative L2 and violation count. Source cache projections from cached h must match1e-10 and did; freshly forwarded differences reflect FP16 batching. Selected-block hook equals matching output_hidden_states; removing hook leaves candidate logits exactly unchanged. Detailed per-example source IDs and diagnostics in source_checks/.', 'All216 classifier pipelines/scalers verified against finite pilot features and frozen hashes;54 per configuration. Existing source fitted mean/scale used, no fitting APIs or classifier predict/decision_function called. Historical Stage B checkpoint commit remains unrecorded; downstream vector/weight/config/tokenizer hashes verified.']
        (P / 'extraction_compatibility_report.md').write_text('\n'.join(compatibility_lines) + '\n', encoding='utf-8')
        memory = ['# Memory report', '', 'One FP16 CUDA model resident at a time; no quantization; selected-layer hook; no KV cache. Source verification uses all-hidden output only for8 training examples to validate block indexing; pilot forward extracts only selected block. Bench GPU allocator peaks reset per configuration and include resident weights. CPU peak is process lifetime high-water mark, not independent per-configuration peak.', '']
        for b in bench:
            memory.append(f'{b['configuration']}: GPU peak allocated{b['peak_gpu_allocated_bytes'] / 2 ** 30:.3f}GiB, reserved{b['peak_gpu_reserved_bytes'] / 2 ** 30:.3f}GiB; process CPU peak{b['process_lifetime_peak_cpu_ram_bytes'] / 2 ** 30:.3f}GiB; OOM{b['oom_events']}; batch{b['final_batch_size']}.')
        (P / 'memory_report.md').write_text('\n'.join(memory) + '\n', encoding='utf-8')
        plan = '# Full inference plan (prepared; not executed)\n\nScript run_stage3b.py requires --execute-full, successful pilot state and exact source/dataset hashes. Full run output isolated under stage3b_inference. Deterministic frozen row order; chunk64, batch4 by default, separate source/model directories. One forward supplies margin plus pooled h and all216 compatible feature sets, no per-classifier forward. Existing tokenizer/mask/scoring and FP16 untouched. Each chunk npz is fsynced and atomically replaced, then JSON completion marker committed; marker records IDs, offset, config signature and file hash. Resume verifies stored IDs/hash/config, finite values, dimensions and no duplicates; incomplete/orphan chunk regenerated without claiming completion. Original ground-truth keys stripped in isolated verifier before selection/inference; inference cache contains IDs/features only. Model/source settings never selected using outcomes. Saved scalers compatibility checked, no predictions. OOM halves only unsaved batch, cannot change precision/examples. Console progress and per-config completion.json saved. Runtime estimates are pilot extrapolations; full run not performed. Explicit user authorization required.\n'
        (P / 'inference_plan.md').write_text(plan + '\n' + json.dumps(estimate, indent=2), encoding='utf-8')
        report = ['# Stage3A complete', '', '32 identical pilot reviews x4 configurations=128 model-input evaluations, plus8 source-reference examples per setting and hook/no-hook validation passes. Pilot selection labels excluded; frozen dataset unchanged. All source checks and216 pipeline compatibility checks passed. No Amazon sentiment metrics/predictions, no fitting, no full run.', '', '|Configuration|Forward+tokenization sec|Sec/review|Input tokens/sec|Peak GPU GiB|', '|---|---:|---:|---:|---:|']
        for b in bench:
            report.append(f'|{b['configuration']}|{b['forward_and_tokenization_seconds']:.3f}|{b['seconds_per_example']:.4f}|{b['input_tokens_per_second']:.2f}|{b['peak_gpu_allocated_bytes'] / 2 ** 30:.3f}|')
        report += ['', f'Token-based full inference estimate {estimate['total_token_based_seconds'] / 3600:.3f} hours including one load/model. Example-based estimate {estimate['total_example_based_seconds'] / 3600:.3f} hours. {estimate['limitation']}', 'Stage3B technically ready; not started. Full-run script and reusable engine saved here.', 'Re-running completed pilot resumes features; benchmark fresh-forward measurements remain recorded in benchmark_results.csv from original run. Any numerical mismatches must stop, never loosen thresholds after outcomes.']
        (P / 'stage3a_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        atomic_json(P / 'state.json', dict(complete=True, completed_configurations=[v['configuration'] for v in bench], successful_reviews_per_configuration=32, all_source_checks_passed=True, classifier_pipelines_verified=216, stage3b_ready=True, no_outcome_evaluation=True))
        state = js(P / 'state.json')
        state['verified_code_sha256'] = {name: sha(P / name) for name in ('stage_k_inference.py', 'run_stage3a.py', 'run_stage3b.py')}
        state['verified_input_sha256'] = {name: sha(K / name) for name in ('transfer_compatibility.json', 'source_model_inventory.csv', 'stage2_dataset/amazon_test_20000.jsonl', 'stage2_dataset/dataset_metadata.json')}
        atomic_json(P / 'state.json', state)
        assert sha(D / 'amazon_test_20000.jsonl') == dataset_sha()
        log('PILOT COMPLETE ' + json.dumps(estimate))
    except Exception as exc:
        atomic_json(P / 'failure.json', dict(error=repr(exc), completed_configurations=[v['configuration'] for v in bench], stage3b_ready=False))
        raise
if __name__ == '__main__':
    main()
