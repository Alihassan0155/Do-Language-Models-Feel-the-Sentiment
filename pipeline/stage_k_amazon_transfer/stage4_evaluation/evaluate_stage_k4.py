"""Stage K4 frozen evaluation; CPU-only, no fitting or Qwen inference."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os, sys, json, csv, time, datetime, hashlib, importlib.util
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
R = WORKSPACE
K = R / 'artifacts/stage_k_amazon_transfer'
B = K / 'stage3b_full_inference'
O = K / 'stage4_evaluation'
P = K / 'stage3a_pilot'
sys.path.insert(0, str(source_path(P / 'stage_k_inference.py').parent))
sys.path.insert(0, str(source_path(B / 'run_stage3b_full.py').parent))
import numpy as np, pandas as pd, joblib, pyarrow as pa, pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import stage_k_inference as g
from run_stage3b_full import validate_config
METHODS = ['original', 'sentiment_only', 'raw', 'denoised', 'pca_rank7'] + [f'random_rank7_seed_{s}' for s in range(2000, 2050)]
METRICS = ['accuracy', 'macro_f1', 'balanced_accuracy', 'auroc']
PRIMARY = [('raw', 'original'), ('raw', 'sentiment_only'), ('raw', 'pca_rank7'), ('raw', 'random_rank7_mean')]
SECONDARY = [('denoised', 'raw'), ('denoised', 'pca_rank7'), ('pca_rank7', 'sentiment_only')]

def log(s):
    print(time.strftime('%H:%M:%S') + ' ' + s, flush=True)

def save(n, v):
    g.atomic_json(O / n, v)

def sha(p):
    p = source_path(p)
    return g.sha(p)

def write_csv(n, rows):
    pd.DataFrame(rows).to_csv(O / n, index=False)

def readonly_rows():
    assert sha(g.D / 'amazon_test_20000.jsonl') == g.dataset_sha()
    out = []
    for line in (g.D / 'amazon_test_20000.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        out.append({k: r[k] for k in ('evaluation_id', 'dataset_row_index', 'text', 'text_sha256')})
    assert len(out) == 20000 and len({r['evaluation_id'] for r in out}) == 20000
    return out

def timing_audit():
    import collections
    logs = [json.loads(line) for line in (B / 'execution_log.jsonl').open(encoding='utf-8')]
    sessions = {}
    timing = {}
    counts = {}
    unique = {}
    for r in logs:
        se = r['session']
        sessions.setdefault(se, {'first': r['timestamp'], 'last': r['timestamp']})
        sessions[se]['last'] = r['timestamp']
        if r['event'] == 'batch_complete':
            c = r['configuration']
            timing[c] = timing.get(c, 0) + r['seconds']
            counts[c] = counts.get(c, 0) + r['n']
            unique.setdefault(c, set()).update(r['ids'])
    assert len(counts) == 4 and all((counts[c] == len(unique[c]) == 20000 for c in counts))
    rows = []
    for c in counts:
        rows.append(dict(configuration=c, completed_batch_examples=counts[c], unique_ids=len(unique[c]), logged_batch_body_seconds=timing[c]))
    sessions_seconds = {se: (datetime.datetime.fromisoformat(v['last']) - datetime.datetime.fromisoformat(v['first'])).total_seconds() for se, v in sessions.items()}
    text = ['# Stage K3B timing interpretation audit', '', 'Runtime scope must be determined from the current run logs; resumed checks are not full inference timings.', 'Runtime scope must be determined from the current run logs; resumed checks are not full inference timings.', 'All raw batch_complete events across both sessions show exactly20k unique input IDs per configuration,80k total, no duplicate forward IDs or batch failures. Original chunk/log/report files are unchanged.', '', pd.DataFrame(rows).to_markdown(index=False) if importlib.util.find_spec('tabulate') else pd.DataFrame(rows).to_string(index=False), '', f'Observed session spans (first to last recorded event):{sessions_seconds}; summed {sum(sessions_seconds.values()) / 60:.3f}min. The first session has no shutdown marker; its span is a lower bound on that interrupted session, not exact process lifetime. Between-session idle time is excluded. Logged batch bodies include tokenization/forward/pooling transfers but omit outer synchronization, feature projection, chunk writes and verification.', 'The originally reported per-configuration scopes are retained in raw artifacts; this new audit supplies corrected interpretation only. No missing/misaligned feature evidence was found.']
    (O / 'runtime_reporting_audit.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    write_csv('runtime_scope_reconstruction.csv', rows)
    return dict(batch_counts=counts, batch_seconds=timing, session_spans_seconds=sessions_seconds, integrity_issue=False)

def preflight():
    assert g.js(B / 'run_state.json')['complete'] and g.js(B / 'integrity_checks.json')['passed']
    state = g.js(P / 'state.json')
    for n, h in state['verified_code_sha256'].items():
        assert sha(P / n) == h
    for n, h in state['verified_input_sha256'].items():
        assert sha(K / n) == h
    tracked = g.js(B / 'preflight.json')['asset_hashes']
    for p, h in tracked.items():
        assert sha(p) == h, 'Preserved artifact hash mismatch' + p
    rows = readonly_rows()
    lengths = list(csv.DictReader((g.D / 'token_length_report.csv').open(encoding='utf-8')))
    checks = []
    for s in g.settings():
        name = s['source'] + '_' + s['model']
        g.Transforms(s)
        log('Integrity checking ' + name)
        checks.append(validate_config(B / 'features' / name, rows, [int(v[name]) for v in lengths]))
    assert all((v['passed'] for v in checks)) and sum((v['n'] for v in checks)) == 80000
    audit = timing_audit()
    inventory = list(csv.DictReader((K / 'source_model_inventory.csv').open(encoding='utf-8')))
    assert len(inventory) == 216
    protocol = dict(stage='K4', frozen_before_predictions_and_target_label_join=True, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), dataset_revision=g.js(g.D / 'dataset_metadata.json')['dataset_revision'], dataset_manifest_sha256=g.dataset_sha(), n=20000, source_model_inventory_sha256=sha(K / 'source_model_inventory.csv'), feature_manifest_sha256=sha(B / 'feature_manifest.json'), evaluation_script_sha256=sha(Path(__file__)), methods=METHODS, original_threshold=0.0, saved_classifier_threshold='positive iff decision_function>0; ties negative; classes[0,1] verified', auc_score='positive-oriented continuous decision_function; original cached margin; random50 average metrics, not predictions or scores', primary_outcomes=METRICS, secondary_outcomes=['positive_recall', 'negative_recall', 'confusion_matrix', 'errors_fixed', 'errors_introduced', 'net_corrections', 'fraction_predictions_changed'], primary_comparisons=PRIMARY, secondary_comparisons=SECONDARY, bootstrap=dict(draws=2000, seed=47, stratification='10000 negative and10000 positive sampled with replacement each draw; same ID multiplicities for all methods/configurations', interval='paired95% percentile,nominal no multiplicity adjustment', random_uncertainty='condition on fixed50 bases; mean metric per resample, seed SD separately, no extra independent examples'), label_join='generate and hash all80k prediction rows first, then load Amazon labels and match IDs', training=False, selection=False, precision_and_features='unchanged source-trained inputs and preprocessing; no inference', interpretation='cross-source prompt and training-domain confounded; predictive utility is not causality or unique emotion semantics', random_seed_sd='sample standard deviation ddof1')
    assert not (O / 'stage4_evaluation_protocol.json').exists(), 'Protocol already exists: do not overwrite a frozen evaluation'
    save('stage4_evaluation_protocol.json', protocol)
    log('Evaluation protocol FROZEN; target-label performance join not yet performed')
    return (rows, inventory, checks, tracked, audit)

def predict_all(rows, inventory):
    target = O / 'amazon_predictions.parquet'
    assert not target.exists()
    tmp = target.with_suffix('.parquet.tmp')
    writer = None
    try:
        for s in g.settings():
            name = s['source'] + '_' + s['model']
            folder = B / 'features' / name
            parts = {n: [] for n in ('margin', 'sentiment_only', 'raw', 'denoised', 'pca_rank7', 'random_rank7')}
            ids = []
            for p in sorted((folder / 'chunks').glob('*.npz')):
                with np.load(p, allow_pickle=False) as z:
                    ids += z['ids'].tolist()
                    for n in parts:
                        parts[n].append(z[n])
            assert ids == [r['evaluation_id'] for r in rows]
            features = {n: np.concatenate(v) for n, v in parts.items()}
            scores = {'original': features['margin']}
            pred = {'original': (scores['original'] > 0).astype(np.uint8)}
            selected = [r for r in inventory if r['source'] == s['source'] and r['model'] == s['model']]
            assert len(selected) == 54
            for r in selected:
                assert sha(r['classifier_path']) == r['sha256']
                pipeline = joblib.load(r['classifier_path'])
                clf = pipeline.named_steps['classifier']
                assert np.array_equal(clf.classes_, [0, 1])
                method = r['method']
                x = features['random_rank7'][:, int(method.rsplit('_', 1)[1]) - 2000, :] if method.startswith('random_rank7_seed_') else features[method]
                assert x.shape == (20000, int(r['feature_count'])) and np.isfinite(x).all()
                assert int(pipeline.named_steps['scaler'].n_samples_seen_) == 14186 and clf.C == float(r['C'])
                scores[method] = pipeline.decision_function(x)
                pred[method] = pipeline.predict(x).astype(np.uint8)
                assert np.array_equal(pred[method], scores[method] > 0) and np.isfinite(scores[method]).all()
            assert set(scores) == set(METHODS)
            columns = {'evaluation_id': ids, 'configuration': [name] * 20000, 'source': [s['source']] * 20000, 'model': [s['model']] * 20000}
            for method in METHODS:
                columns[method + '_prediction'] = pred[method]
                columns[method + '_score'] = scores[method].astype(np.float64)
            table = pa.table(columns)
            assert 'label' not in table.column_names
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression='zstd')
            writer.write_table(table, row_group_size=20000)
            log('Frozen predictions generated: ' + name + ';55 methods; no target labels joined')
        writer.close()
        writer = None
        os.replace(tmp, target)
    finally:
        if writer:
            writer.close()
    assert pq.ParquetFile(target).metadata.num_rows == 80000
    freeze = dict(predictions_sha256=sha(target), rows=80000, protocol_sha256=sha(O / 'stage4_evaluation_protocol.json'), classifier_inventory_sha256=sha(K / 'source_model_inventory.csv'), freeze_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), label_join_not_yet_performed=True)
    save('prediction_freeze.json', freeze)
    log('All predictions saved and hashed BEFORE ground-truth label join')
    return freeze

def point_metrics(y, p, score):
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(accuracy=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average='macro', zero_division=0)), balanced_accuracy=float(balanced_accuracy_score(y, p)), auroc=float(roc_auc_score(y, score)), negative_recall=float(tn / (tn + fp)), positive_recall=float(tp / (tp + fn)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def ranked_auc(weights, order, starts, y):
    ordered = np.ascontiguousarray(weights[:, order])
    ys = y[order]
    neg = np.add.reduceat(ordered * (ys == 0), starts, axis=1, dtype=np.uint32)
    pos = np.add.reduceat(ordered * (ys == 1), starts, axis=1, dtype=np.uint32)
    cumulative = np.cumsum(neg, axis=1, dtype=np.uint32)
    return np.sum(pos.astype(np.float64) * (cumulative - 0.5 * neg), axis=1) / 100000000.0

def bootstrap(y, pred, scores, weights):
    results = np.empty((2000, len(METHODS), 4), dtype=np.float64)
    masks = np.column_stack([(pred[:, j] == 0) & (y == 0) for j in range(len(METHODS))] + [(pred[:, j] == 1) & (y == 1) for j in range(len(METHODS))]).astype(np.float64)
    totals = weights.astype(np.float64) @ masks
    tn = totals[:, :len(METHODS)]
    tp = totals[:, len(METHODS):]
    fp = 10000 - tn
    fn = 10000 - tp
    results[:, :, 0] = (tn + tp) / 20000
    results[:, :, 2] = 0.5 * (tn / 10000 + tp / 10000)
    results[:, :, 1] = 0.5 * (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn))
    for j, m in enumerate(METHODS):
        order = np.argsort(scores[:, j], kind='stable')
        values = scores[order, j]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        for start in range(0, 2000, 32):
            results[start:start + 32, j, 3] = ranked_auc(weights[start:start + 32], order, starts, y)
        if j % 10 == 0:
            log(f'Bootstrap AUROC:method{j + 1}/55;2000 paired resamples')
    return results

def figures(summary, comparisons, intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    folder = O / 'figures'
    folder.mkdir(exist_ok=True)
    names = ['original', 'sentiment_only', 'raw', 'denoised', 'pca_rank7', 'random_rank7_mean']
    labels = ['Original', 'Margin LR', 'Raw emotion', 'Denoised', 'PCA-7', 'Random-7 mean']
    configs = ['sst2_0.5b', 'sst2_3b', 'tweeteval_0.5b', 'tweeteval_3b']
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharey=True, layout='constrained')
    for ax, c in zip(axs.flat, configs):
        vals = [summary[c][m]['accuracy'] * 100 for m in names]
        bounds = [intervals[c, m, 'accuracy'] for m in names]
        errors = np.array([[v - lo * 100, hi * 100 - v] for v, (lo, hi) in zip(vals, bounds)]).T
        ax.bar(np.arange(6), vals, color=['#777777', '#4477AA', '#228833', '#66CCEE', '#AA3377', '#CCBB44'], yerr=errors, capsize=2)
        ax.set_xticks(range(6), labels, rotation=30, ha='right')
        ax.set_title(c.replace('_', ' / '))
        ax.set_ylim(0, 100)
        ax.set_ylabel('Amazon accuracy (%)')
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(folder / ('amazon_accuracy.' + ext), dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6), layout='constrained')
    for i, c in enumerate(configs):
        r = next((r for r in comparisons if r['configuration'] == c and r['method'] == 'raw' and (r['reference'] == 'pca_rank7') and (r['metric'] == 'accuracy')))
        value = r['difference_pp']
        ax.errorbar(value, i, xerr=[[value - r['ci_lower_pp']], [r['ci_upper_pp'] - value]], fmt='o', color='#228833', capsize=4)
    ax.axvline(0, color='#666666', linestyle='--', linewidth=1)
    ax.set_yticks(range(4), [c.replace('_', ' / ') for c in configs])
    ax.invert_yaxis()
    ax.set_xlabel('Raw emotion minus PCA-7 accuracy (percentage points)')
    ax.grid(axis='x', alpha=0.2)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(folder / ('raw_vs_pca_accuracy_difference.' + ext), dpi=300)
    plt.close(fig)
    (folder / 'captions.md').write_text('Accuracy: frozen source-trained classifiers evaluated on the same 20,000 balanced Amazon examples. Error bars are nominal class-stratified paired-bootstrap 95% percentile intervals (2,000 draws). Random-7 denotes the mean metric of 50 fixed seed classifiers, not an ensemble; its interval conditions on these seeds. All correction methods include the sentiment margin. Source prompts differ.\n\nRaw versus PCA: accuracy differences on the same target examples, with nominal paired-bootstrap 95% intervals. No target-domain fitting or model selection.\n', encoding='utf-8')

def historical():
    output = []
    for source, path in [('sst2', R / 'artifacts/stage_i_sst2/metrics_all_methods.csv'), ('tweeteval', R / 'artifacts/stage_i_tweeteval_matched_controls/metrics_all_methods.csv')]:
        if path.exists():
            frame = pd.read_csv(path, float_precision='round_trip')
            log('Historical source metrics columns: ' + str(list(frame.columns)))
            output.append({'source': source, 'path': str(path), 'sha256': sha(path), 'columns': list(frame.columns)})
            frame.to_csv(O / (source + '_historical_saved_metrics_copy.csv'), index=False)
    return output

def main():
    O.mkdir(parents=True, exist_ok=True)
    rows, inventory, checks, tracked, audit = preflight()
    freeze = predict_all(rows, inventory)
    assert sha(O / 'amazon_predictions.parquet') == freeze['predictions_sha256']
    assert sha(O / 'stage4_evaluation_protocol.json') == freeze['protocol_sha256']
    target = [json.loads(line) for line in (g.D / 'amazon_test_20000.jsonl').open(encoding='utf-8')]
    assert [r['evaluation_id'] for r in target] == [r['evaluation_id'] for r in rows]
    y = np.array([r['label'] for r in target], dtype=np.uint8)
    assert np.bincount(y).tolist() == [10000, 10000]
    assert all((r['label_name'] == ('positive' if r['label'] == 1 else 'negative') for r in target))
    save('label_join_audit.json', dict(join_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), prediction_freeze_sha256=sha(O / 'prediction_freeze.json'), dataset_sha256=sha(g.D / 'amazon_test_20000.jsonl'), n=20000, negative=10000, positive=10000))
    log('Frozen predictions verified; target labels now joined for evaluation')
    rng = np.random.default_rng(47)
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    weights = np.empty((2000, 20000), dtype=np.uint16)
    for b in range(2000):
        draw = np.concatenate([rng.choice(ix, size=10000, replace=True) for ix in classes])
        weights[b] = np.bincount(draw, minlength=20000)
    assert all((np.all(weights[:, ix].sum(axis=1) == 10000) for ix in classes))
    metrics = []
    random_rows = []
    comparisons = []
    corrections = []
    agreements = []
    ci_rows = []
    summary = {}
    intervals = {}
    parity = []
    for s in g.settings():
        c = s['source'] + '_' + s['model']
        log('Evaluating ' + c)
        table = pq.read_table(O / 'amazon_predictions.parquet', filters=[('configuration', '=', c)]).to_pandas()
        assert table.evaluation_id.tolist() == [r['evaluation_id'] for r in rows]
        pred = np.column_stack([table[m + '_prediction'].to_numpy() for m in METHODS])
        scores = np.column_stack([table[m + '_score'].to_numpy() for m in METHODS])
        points = {m: point_metrics(y, pred[:, j], scores[:, j]) for j, m in enumerate(METHODS)}
        for j, m in enumerate(METHODS):
            r = dict(configuration=c, source=s['source'], model=s['model'], method=m, n=20000, **points[m])
            metrics.append(r)
            if j >= 5:
                random_rows.append(dict(**r, seed=2000 + j - 5))
            old_correct = pred[:, 0] == y
            new_correct = pred[:, j] == y
            fixed = int(np.sum(~old_correct & new_correct))
            broken = int(np.sum(old_correct & ~new_correct))
            corrections.append(dict(configuration=c, method=m, errors_fixed=fixed, correct_made_incorrect=broken, net_corrections=fixed - broken, predictions_changed=int(np.sum(pred[:, j] != pred[:, 0])), fraction_changed=float(np.mean(pred[:, j] != pred[:, 0]))))
        mean = {k: float(np.mean([points[m][k] for m in METHODS[5:]])) for k in points['original']}
        points['random_rank7_mean'] = mean
        metrics.append(dict(configuration=c, source=s['source'], model=s['model'], method='random_rank7_mean', n=20000, **mean, **{k + '_seed_sd': float(np.std([points[m][k] for m in METHODS[5:]], ddof=1)) for k in METRICS}))
        raw_ok = pred[:, 2] == y
        pca_ok = pred[:, 4] == y
        agreements.append(dict(configuration=c, both_correct=int(np.sum(raw_ok & pca_ok)), both_wrong=int(np.sum(~raw_ok & ~pca_ok)), only_raw_correct=int(np.sum(raw_ok & ~pca_ok)), only_pca_correct=int(np.sum(~raw_ok & pca_ok))))
        boot = bootstrap(y, pred, scores, weights)
        for b in (0, 1, 2):
            ix = np.repeat(np.arange(20000), weights[b].astype(np.int64))
            for j in (0, 2, 4, 5, 54):
                reference = point_metrics(y[ix], pred[ix, j], scores[ix, j])
                np.testing.assert_allclose(boot[b, j], [reference[m] for m in METRICS], atol=1e-12, rtol=0)
        parity.append(dict(configuration=c, draws_verified=3, methods_verified=5, tolerance=1e-12, passed=True))
        np.save(O / (c + '_bootstrap_metrics.npy'), boot, allow_pickle=False)
        distribution = {m: boot[:, j, :] for j, m in enumerate(METHODS)}
        distribution['random_rank7_mean'] = boot[:, 5:, :].mean(axis=1)
        for m in distribution:
            for k, metric in enumerate(METRICS):
                lo, hi = np.quantile(distribution[m][:, k], [0.025, 0.975])
                intervals[c, m, metric] = (lo, hi)
                ci_rows.append(dict(configuration=c, kind='method_metric', method=m, reference='', metric=metric, estimate=points[m][metric], ci_lower=lo, ci_upper=hi))
        for kind, pairs in [('primary', PRIMARY), ('secondary', SECONDARY)]:
            for m, reference in pairs:
                for k, metric in enumerate(METRICS):
                    diff = distribution[m][:, k] - distribution[reference][:, k]
                    lo, hi = np.quantile(diff, [0.025, 0.975])
                    effect = points[m][metric] - points[reference][metric]
                    r = dict(configuration=c, comparison_type=kind, method=m, reference=reference, metric=metric, difference=effect, ci_lower=lo, ci_upper=hi, difference_pp=100 * effect, ci_lower_pp=100 * lo, ci_upper_pp=100 * hi, draws=2000, seed=47)
                    comparisons.append(r)
                    ci_rows.append(dict(configuration=c, kind=kind + '_paired_difference', method=m, reference=reference, metric=metric, estimate=effect, ci_lower=lo, ci_upper=hi))
        summary[c] = points
        log(c + ' complete; raw accuracy ' + format(points['raw']['accuracy'], '.4f'))
    write_csv('all_method_metrics.csv', metrics)
    write_csv('random_seed_metrics.csv', random_rows)
    write_csv('primary_comparisons.csv', [r for r in comparisons if r['comparison_type'] == 'primary'])
    write_csv('secondary_comparisons.csv', [r for r in comparisons if r['comparison_type'] == 'secondary'])
    write_csv('bootstrap_confidence_intervals.csv', ci_rows)
    write_csv('correction_analysis.csv', corrections)
    write_csv('emotion_vs_pca_agreement.csv', agreements)
    figures(summary, comparisons, intervals)
    history = historical()
    names = ['original', 'sentiment_only', 'raw', 'denoised', 'pca_rank7', 'random_rank7_mean']
    lines = ['# Stage K4 frozen Amazon cross-domain evaluation', '', '20,000 unchanged balanced official Amazon test examples; four frozen source/model configurations; no fitting, tuning, target selection or Qwen forward passes. All 80,000 prediction rows were saved and hashed before labels were joined. All learned correction methods include the sentiment margin; emotion/PCA/random coordinates are supplementary features.', '']
    lines += ['| Configuration | Original | Margin LR | Raw emotion | Denoised | PCA-7 | Random-7 mean |', '|---|---:|---:|---:|---:|---:|---:|']
    for c, v in summary.items():
        lines.append('| ' + c + ' | ' + ' | '.join((f'{v[m]['accuracy'] * 100:.2f}' for m in names)) + ' |')
    lines += ['', 'Accuracy values above are percentages. Random means average 50 separate classifier metrics, not ensemble predictions.', '', '| Configuration | Method | Accuracy | Macro-F1 | Balanced accuracy | AUROC |', '|---|---|---:|---:|---:|---:|']
    for c, v in summary.items():
        for m in names:
            lines.append('| ' + c + ' | ' + m + ' | ' + ' | '.join((f'{v[m][k]:.6f}' for k in METRICS)) + ' |')
    lines += ['', '| Configuration | Comparison | Metric | Difference (pp) | Nominal paired 95% CI (pp) |', '|---|---|---|---:|---|']
    for r in comparisons:
        lines.append(f'| {r['configuration']} | {r['method']} minus {r['reference']} | {r['metric']} | {r['difference_pp']:+.3f} | [{r['ci_lower_pp']:+.3f}, {r['ci_upper_pp']:+.3f}] |')
    lines += ['', '| Configuration | Method | Errors fixed | Correct predictions broken | Net |', '|---|---|---:|---:|---:|']
    for r in corrections:
        if r['method'] in names:
            lines.append(f'| {r['configuration']} | {r['method']} | {r['errors_fixed']} | {r['correct_made_incorrect']} | {r['net_corrections']} |')
    lines += ['', '## Findings and limits', '']
    for c, v in summary.items():
        for reference in ('original', 'sentiment_only', 'pca_rank7', 'random_rank7_mean'):
            r = next((r for r in comparisons if r['configuration'] == c and r['method'] == 'raw' and (r['reference'] == reference) and (r['metric'] == 'accuracy')))
            conclusion = 'positive interval' if r['ci_lower'] > 0 else 'negative interval' if r['ci_upper'] < 0 else 'interval includes zero'
            lines.append(f'- {c}: raw versus {reference}: {r['difference_pp']:+.3f} pp [{r['ci_lower_pp']:+.3f}, {r['ci_upper_pp']:+.3f}]; {conclusion}.')
    lines += ['', 'Intervals are 2,000 class-stratified paired percentile bootstrap draws, seed 47, with identical resampled IDs across all methods and configurations. Nominal intervals have no multiplicity correction. Random-control bootstrap uncertainty conditions on the 50 fixed bases; sample seed SD (ddof=1) is reported separately. This evaluates predictive transfer, not causal necessity or uniquely emotional semantics. Source-trained heads and their prompts differ across SST-2 and TweetEval, so cross-source differences are confounded. Existing source test results were previously inspected; copies are provided only for historical context, not independent confirmation. Original Stage B model-commit provenance is limited by its saved metadata. No source or target hyperparameter selection was performed here.', '', 'Timing scope is derived from the current run logs; cache verification is not inference timing.']
    (O / 'stage4_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log('Rechecking preserved source assets and feature chunks')
    for path, h in tracked.items():
        assert sha(path) == h
    lengths = list(csv.DictReader((g.D / 'token_length_report.csv').open(encoding='utf-8')))
    for s in g.settings():
        c = s['source'] + '_' + s['model']
        assert validate_config(B / 'features' / c, rows, [int(v[c]) for v in lengths])['passed']
    assert sha(O / 'amazon_predictions.parquet') == freeze['predictions_sha256'] and sha(O / 'stage4_evaluation_protocol.json') == freeze['protocol_sha256']
    save('stage4_integrity_report.json', dict(passed=True, complete=True, no_training=True, no_qwen_forward=True, no_target_tuning=True, source_assets_preserved=True, all_feature_caches_revalidated=True, predictions_frozen_before_label_join=True, predictions_sha256=freeze['predictions_sha256'], configuration_count=4, examples_per_configuration=20000, methods_per_configuration=55, frozen_classifier_count=216, bootstrap_parity_checks=parity, runtime_audit=audit, historical_source_files=history, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    log('STAGE K4 COMPLETE: ' + str(O))
    print('\n'.join(lines[:10]), flush=True)
if __name__ == '__main__':
    main()
