"""CPU reporting of STE117 new forwards; no inference/training imports or calls."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import csv, json, sqlite3, hashlib, importlib.util, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
OUT = WORKSPACE / 'artifacts/asd_ste117_binary_prompt'
ROOT = WORKSPACE
ART = ROOT / 'artifacts'
DS = 'tweet_eval_sentiment'
KEYS = ('0.5b', '3b')
spec = importlib.util.spec_from_file_location('metric116', source_path(ART / 'binary_reanalysis_20261019/reanalyze.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

def js(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    p = source_path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, v):
    (OUT / name).write_text(json.dumps(v, indent=2) + '\n', encoding='utf-8')

def readcsv(p):
    return list(csv.DictReader(p.open(encoding='utf-8')))

def writecsv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def ro(path):
    c = sqlite3.connect('file:' + path.as_posix() + '?mode=ro&immutable=1', uri=True)
    c.row_factory = sqlite3.Row
    return c

def main(baseline_only=False):
    protocol = js(OUT / 'protocol.json')
    samples = js(OUT / 'samples.json')[DS]
    allrows = js(OUT / 'baseline_samples.json')
    n = len(samples)
    y = np.array([r['label'] for r in samples])
    seeds = protocol['random_seeds']
    B = protocol['bootstrap']['draws']
    assert n == 600 and np.bincount(y).tolist() == [300, 300] and (len({r['id'] for r in samples}) == 600)
    assert js(OUT / 'smoke_validation.json')['failures'] == 0
    for rel, digest in protocol['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    lock = js(OUT / 'frozen_runner.json')
    for name, field in [('run117.py', 'run117_sha256'), ('binary_engine.py', 'engine_sha256'), ('protocol.json', 'protocol_sha256')]:
        assert sha(OUT / name) == lock[field]
    baseline = ro(OUT / 'baseline/scores.sqlite3')
    basepred = []
    basemetrics = []
    oldbaseline = readcsv(ART / 'stage_i_sentiment_correction/summary_table.csv')
    baselinecomparison = []
    for model in KEYS:
        rr = [dict(x) for x in baseline.execute('select * from scores where model=? and dataset=? and method=? order by example_index', (model, DS, 'baseline'))]
        assert [x['example_index'] for x in rr] == list(range(6347))
        gold = np.array([r['label'] for r in allrows])
        p = np.array([r['predicted'] for r in rr])
        m = np.array([r['margin'] for r in rr])
        assert np.array_equal(p, m > 0)
        cm = confusion_matrix(gold, p, labels=[0, 1])
        basemetrics.append(dict(model=model, n=len(gold), accuracy=float(accuracy_score(gold, p)), macro_f1=float(f1_score(gold, p, average='macro')), balanced_accuracy=float(balanced_accuracy_score(gold, p)), auroc=float(roc_auc_score(gold, m)), true_negative=int(cm[0, 0]), false_positive=int(cm[0, 1]), false_negative=int(cm[1, 0]), true_positive=int(cm[1, 1]), predicted_negative=int((p == 0).sum()), predicted_positive=int((p == 1).sum())))
        for row, item in zip(rr, allrows):
            assert np.isfinite([row['logp_positive'], row['logp_negative'], row['margin']]).all() and row['logp_neutral'] is None
            assert np.float32(row['logp_positive']) - np.float32(row['logp_negative']) == row['margin']
            basepred.append(dict(model=model, id=item['id'], source_split=item['source_split'], source_row_index=item['source_row_index'], gold_label=item['label'], logp_positive=row['logp_positive'], logp_negative=row['logp_negative'], margin=row['margin'], binary_prediction=row['predicted']))
        old = next((r for r in oldbaseline if r['model'] == model and r['split'] == 'test' and (r['method'] == 'original_margin')))
        for metric in ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc'):
            baselinecomparison.append(dict(model=model, n=6347, metric=metric, old_three_label_prompt_binary_readout=float(old[metric]), new_binary_prompt=basemetrics[-1][metric], change=basemetrics[-1][metric] - float(old[metric])))
    baseline.close()
    pq.write_table(pa.Table.from_pylist(basepred), OUT / 'baseline_predictions.parquet')
    writecsv('baseline_metrics.csv', basemetrics)
    writecsv('baseline_historical_comparison.csv', baselinecomparison)
    if baseline_only:
        print('Both 6347-example baseline results saved; no inference or setting changes.', flush=True)
        return
    full = ro(OUT / 'full/scores.sqlite3')
    matched = ro(OUT / 'matched/scores.sqlite3')
    matching = ro(OUT / 'matched/matching.sqlite3')
    methods = ['baseline', 'emotion_raw', 'emotion_denoised'] + [f'random_rank7_seed_{s}' for s in seeds]
    allmethods = methods + ['baseline_parity'] + [f'matched_{kind}_seed_{s}' for s in seeds for kind in ('raw', 'denoised')]
    cached = {}
    matchingrows = []
    predrows = []
    max_point_gap = 0.0
    interval_check_gap = 0.0
    checked = 0
    for stage, db, names in [('full', full, methods), ('matched', matched, allmethods)]:
        for model in KEYS:
            for name in names:
                rr = [dict(x) for x in db.execute('select * from scores where model=? and dataset=? and method=? order by example_index', (model, DS, name))]
                assert [r['example_index'] for r in rr] == list(range(n))
                margin = np.array([r['margin'] for r in rr])
                p = np.array([r['predicted'] for r in rr])
                assert np.isfinite(margin).all() and np.array_equal(p, margin > 0)
                lp = np.array([[r['logp_positive'], r['logp_negative']] for r in rr], dtype=np.float32)
                assert np.isfinite(lp).all() and np.array_equal(lp[:, 0] - lp[:, 1], margin)
                assert all((r['logp_neutral'] is None for r in rr))
                cached[stage, model, name] = (rr, p, margin)
                if stage == 'matched' and name in methods:
                    assert rr == cached['full', model, name][0]
                for i, r in enumerate(rr):
                    predrows.append(dict(**r, stage=stage, id=samples[i]['id'], gold_label=int(y[i])))
            if stage == 'matched':
                assert np.array_equal(cached[stage, model, 'baseline'][2], cached[stage, model, 'baseline_parity'][2])
                for seed in seeds:
                    random = cached[stage, model, f'random_rank7_seed_{seed}'][0]
                    for kind in ('raw', 'denoised'):
                        name = f'matched_{kind}_seed_{seed}'
                        aa = [dict(x) for x in matching.execute('select * from matching where model=? and method=? order by example_index', (model, name))]
                        assert [r['example_index'] for r in aa] == list(range(n))
                        target = np.array([r['target_energy'] for r in aa])
                        actual = np.array([r['actual_energy'] for r in aa])
                        assert np.isfinite(np.c_[target, actual, [r['beta'] for r in aa]]).all()
                        assert np.allclose(target, [r['token_removed_energy_fraction'] for r in random], atol=1e-07, rtol=1e-05)
                        assert np.allclose(actual, [r['token_removed_energy_fraction'] for r in cached[stage, model, name][0]], atol=1e-07, rtol=1e-05)
                        for i, r in enumerate(aa):
                            assert 0 <= r['beta'] <= 1
                            gap = abs(r['actual_energy'] - r['target_energy'])
                            ok = bool(r['feasible']) and gap <= 0.01 * r['target_energy'] + 1e-07
                            matchingrows.append(dict(**r, id=samples[i]['id'], gold_label=int(y[i]), seed=seed, kind=kind, absolute_energy_gap=gap, relative_energy_gap=gap / max(r['target_energy'], 1e-12), matching_ok=ok))
    assert len(matchingrows) == 120000
    writecsv('matching_diagnostics.csv', matchingrows)
    writecsv('unmatched_energy_cases.csv', [r for r in matchingrows if not r['matching_ok']] or [dict(status='none')])
    pq.write_table(pa.Table.from_pylist(predrows), OUT / 'ablation_predictions.parquet')
    results = []
    cis = []
    variation = []
    energy = []
    metric_names = ('accuracy', 'macro_f1', 'auroc')
    point = np.full((1, n), 1 / n)
    for stage in ('full', 'matched'):
        w = helper.weights(y, B, 44 if stage == 'full' else 45)
        sw = np.random.default_rng(46).multinomial(50, np.full(50, 1 / 50), size=B) / 50 if stage == 'matched' else np.full((B, 50), 1 / 50)
        for model in KEYS:

            def get(name):
                nonlocal max_point_gap
                rr, p, m = cached[stage, model, name]
                v = helper.metrics(y, p, m, point)[0]
                b = helper.metrics(y, p, m, w)
                expected = [accuracy_score(y, p), f1_score(y, p, average='macro'), roc_auc_score(y, m)]
                gap = float(np.max(np.abs(v - expected)))
                assert gap < 1e-12
                max_point_gap = max(max_point_gap, gap)
                return (v, b)
            bp, bb = get('baseline')
            rp = []
            rb = []
            for seed in seeds:
                v, b = get(f'random_rank7_seed_{seed}')
                rp.append(v)
                rb.append(b)
            rp = np.array(rp)
            rb = np.stack(rb, axis=1)
            rmean = rp.mean(axis=0)
            rboot = np.einsum('bs,bsm->bm', sw, rb)
            conditions = {'baseline': (bp, bb, None), 'random_rank7_mean': (rmean, rboot, rp)}
            for kind in ('raw', 'denoised'):
                if stage == 'full':
                    v, b = get('emotion_' + kind)
                    ps = np.tile(v, (50, 1))
                    conditions['emotion_' + kind] = (v, b, None)
                else:
                    vals = [get(f'matched_{kind}_seed_{s}') for s in seeds]
                    ps = np.stack([x[0] for x in vals])
                    conditions['emotion_' + kind] = (ps.mean(axis=0), np.einsum('bs,bsm->bm', sw, np.stack([x[1] for x in vals], axis=1)), ps)
                for i, seed in enumerate(seeds):
                    for j, metric in enumerate(metric_names):
                        variation.append(dict(stage=stage, model=model, condition='emotion_' + kind, seed=seed, metric=metric, performance=float(ps[i, j]), damage=float(bp[j] - ps[i, j]), excess_vs_random=float(rp[i, j] - ps[i, j])))
            for i, seed in enumerate(seeds):
                for j, metric in enumerate(metric_names):
                    variation.append(dict(stage=stage, model=model, condition='random_rank7', seed=seed, metric=metric, performance=float(rp[i, j]), damage=float(bp[j] - rp[i, j]), excess_vs_random=''))
            for condition, (v, b, _) in conditions.items():
                rec = dict(stage=stage, model=model, condition=condition, n=n, random_controls=50)
                for j, metric in enumerate(metric_names):
                    damage = float(bp[j] - v[j])
                    excess = float(rmean[j] - v[j]) if condition.startswith('emotion') else ''
                    rec[metric] = float(v[j])
                    rec[metric + '_damage'] = damage
                    rec[metric + '_excess_vs_random'] = excess
                    for stat, value, draw in [('performance', v[j], b[:, j]), ('damage', damage, bb[:, j] - b[:, j])] + ([('excess_vs_random', excess, rboot[:, j] - b[:, j])] if condition.startswith('emotion') else []):
                        lo, hi = helper.interval(draw)
                        cis.append(dict(stage=stage, model=model, condition=condition, metric=metric, statistic=stat, value=float(value), ci_low=lo, ci_high=hi, bootstrap_draws=B))
                results.append(rec)
            print('Reported', stage, model, flush=True)
    for c in (full, matched, matching):
        c.close()
    writecsv('full_ablation_results.csv', [r for r in results if r['stage'] == 'full'])
    writecsv('matched_ablation_results.csv', [r for r in results if r['stage'] == 'matched'])
    writecsv('full_confidence_intervals.csv', [r for r in cis if r['stage'] == 'full'])
    writecsv('matched_confidence_intervals.csv', [r for r in cis if r['stage'] == 'matched'])
    writecsv('random_seed_variation.csv', variation)
    summary = []
    for stage in ('full', 'matched'):
        for model in KEYS:
            for condition in ('random_rank7', 'emotion_raw', 'emotion_denoised'):
                for metric in metric_names:
                    rr = [r for r in variation if (r['stage'], r['model'], r['condition'], r['metric']) == (stage, model, condition, metric)]
                    summary.append(dict(stage=stage, model=model, condition=condition, metric=metric, performance_sd=float(np.std([r['performance'] for r in rr], ddof=1)), excess_sd=float(np.std([r['excess_vs_random'] for r in rr], ddof=1)) if condition.startswith('emotion') else ''))
    writecsv('seed_variation_summary.csv', summary)
    oldci = readcsv(ART / 'binary_reanalysis_20261019/binary_confidence_intervals.csv')
    comparison = []
    for r in cis:
        old = next((x for x in oldci if all((x[k] == r[k] for k in ('stage', 'model', 'condition', 'metric', 'statistic')))))
        comparison.append(dict(**r, old_value=float(old['value']), old_ci_low=float(old['ci_low']), old_ci_high=float(old['ci_high']), change=r['value'] - float(old['value'])))
    writecsv('historical_comparison.csv', comparison)
    matching_summary = []
    for model in KEYS:
        for kind in ('raw', 'denoised'):
            rr = [r for r in matchingrows if r['model'] == model and r['kind'] == kind]
            matching_summary.append(dict(model=model, kind=kind, comparisons=len(rr), infeasible=sum((not r['feasible'] for r in rr)), failed=sum((not r['matching_ok'] for r in rr)), mean_beta=float(np.mean([r['beta'] for r in rr])), mean_target_energy=float(np.mean([r['target_energy'] for r in rr])), mean_actual_energy=float(np.mean([r['actual_energy'] for r in rr])), max_absolute_energy_gap=max((r['absolute_energy_gap'] for r in rr)), max_relative_energy_gap=max((r['relative_energy_gap'] for r in rr))))
    writecsv('matching_summary.csv', matching_summary)
    lines = ['# ASD-STE117 — Binary Prompt Standardization', '', '## A. Exact prompt verification', '', 'Exact SST-2 P1, unchanged whitespace; stage_d.token_batch chat rendering, left padding, no additional special tokens, first assistant-position full-vocabulary log probabilities. Only negative and positive are scored; positive iff float32 M>0, ties negative.', '', '```text', protocol['prompt'], '```', '', '## B. Dataset membership verification', '', 'Original 6,347 non-neutral official TweetEval test IDs for baseline; original 600 balanced intervention IDs/order/batches, 300/class. Same frozen source labels/texts, checkpoints, selected layers14/18, emotion/raw denoising parameters, historical rank7 bases and seeds2000–2049. Retrospective sample, not a new untouched test. All output margins, interventions on binary-prompt activations, betas and energies computed anew.', '', '## C. New binary Qwen baseline results', '', '|Model|N|Accuracy %|Macro-F1 %|Balanced accuracy %|AUROC|TN / FP / FN / TP|Predicted negative / positive|', '|---|---:|---:|---:|---:|---:|---|---|']
    for r in basemetrics:
        lines.append(f'|{r['model']}|{r['n']}|{100 * r['accuracy']:.4f}|{100 * r['macro_f1']:.4f}|{100 * r['balanced_accuracy']:.4f}|{r['auroc']:.6f}|{r['true_negative']} / {r['false_positive']} / {r['false_negative']} / {r['true_positive']}|{r['predicted_negative']} / {r['predicted_positive']}|')
    for stage, title in [('full', 'D. New binary full-ablation results'), ('matched', 'E. New binary matched-ablation results')]:
        lines += ['', '## ' + title, '', '|Model|Condition|Accuracy %|Macro-F1 %|AUROC|Accuracy damage pp|Excess loss pp [95% CI]|', '|---|---|---:|---:|---:|---:|---|']
        for r in results:
            if r['stage'] != stage:
                continue
            ex = '—'
            if r['condition'].startswith('emotion'):
                c = next((x for x in cis if (x['stage'], x['model'], x['condition'], x['metric'], x['statistic']) == (stage, r['model'], r['condition'], 'accuracy', 'excess_vs_random')))
                ex = f'{100 * c['value']:+.4f} [{100 * c['ci_low']:+.4f}, {100 * c['ci_high']:+.4f}]'
            lines.append(f'|{r['model']}|{r['condition']}|{100 * r['accuracy']:.4f}|{100 * r['macro_f1']:.4f}|{r['auroc']:.6f}|{100 * r['accuracy_damage']:+.4f}|{ex}|')
    valid = all((r['matching_ok'] for r in matchingrows))
    lines += ['', '## F. Confidence intervals and matching checks', '', f'All120000 original-design example/seed/emotion comparisons recorded; failures={sum((not r['matching_ok'] for r in matchingrows))}, infeasible={sum((not r['feasible'] for r in matchingrows))}. Maximum absolute normalized-energy difference={max((r['absolute_energy_gap'] for r in matchingrows)):.9g}. Failures are retained and flagged; no outcome-based exclusion.', '', 'Full: 2000 paired stratified example draws seed44, conditional on fixed bases. Matched: 2000 paired stratified example draws seed45 and joint paired-seed multinomial draws seed46. Means average per-seed metrics; repeated seed predictions are not independent examples. Pointwise percentile95% CIs, no multiplicity correction. Full removal is NOT energy matched. See matching_summary.csv and seed_variation_summary.csv.', '', '## G. New vs historical results', '', '|Model|Old raw matched excess pp [95% CI]|New raw matched excess pp [95% CI]|Assessment|', '|---|---|---|---|']
    for model in KEYS:
        r = next((x for x in comparison if (x['stage'], x['model'], x['condition'], x['metric'], x['statistic']) == ('matched', model, 'emotion_raw', 'accuracy', 'excess_vs_random')))
        raw_valid = all((x['matching_ok'] for x in matchingrows if x['model'] == model and x['kind'] == 'raw'))
        assessment = 'INCONCLUSIVE' if not raw_valid or r['ci_low'] <= 0 <= r['ci_high'] else 'PRESERVED: positive excess' if r['ci_low'] > 0 else 'REVERSED: negative excess'
        if raw_valid and r['ci_low'] <= 0 <= r['ci_high'] and (r['old_ci_low'] > 0):
            assessment = 'WEAKENED: CI crosses zero'
        lines.append(f'|{model}|{100 * r['old_value']:+.4f} [{100 * r['old_ci_low']:+.4f}, {100 * r['old_ci_high']:+.4f}]|{100 * r['value']:+.4f} [{100 * r['ci_low']:+.4f}, {100 * r['ci_high']:+.4f}]|{assessment}|')
    lines += ['', 'historical_comparison.csv includes baseline, raw, denoised, random, damage, excess and CI changes for all three metrics under both ablation types. baseline_historical_comparison.csv separately compares the6347-example original-margin baseline. Do not substitute that baseline for the600-example intervention baseline.', '', '## H. Downstream experiments requiring reruns', '', 'See downstream_dependency_audit.md for E/F/G/I/K dependencies and exact implementation locations. These stages were not rerun. Old prompted states/margins/classifiers remain historical outputs. New source correction pipelines require source-only inference/refitting and new target inference for standardized Amazon transfer; no Amazon-label tuning.', '', '## I. Exact manuscript sections affected', '', 'Current manuscript source is unavailable: exact manuscript line numbers, section names and replacement quotations NOT VERIFIED. Update TweetEval binary baseline and ablation methods/results/captions and any associated abstract/conclusion claims. Keep generated-label Table1 as the historical three-class analysis. Do not relabel old E/F/G/I/K results as new binary-prompt results. Manuscript unchanged.', '', '## J. Final scientific assessment', '', 'Old three-label prompting allowed neutral in its instructions and candidate argmax. STE116 restricted the readout to two candidates under that old prompt. STE117 changes the actual prompt to the exact SST-2 binary P1 and recomputes all forwards and matched energies/betas. This tests prompt sensitivity of the intervention finding on previously inspected examples; it is not independent replication. Matching perturbation energy does not equate intervention direction or prove uniquely emotional necessity.', '', '## Reproducibility', '', '```powershell', 'python -B artifacts/asd_ste117_binary_prompt/run117.py --phase prepare', 'python -B artifacts/asd_ste117_binary_prompt/run117.py --phase all', 'python -B artifacts/asd_ste117_binary_prompt/report117.py', '```', '', 'All phases use resumable new SQLite checkpoints. all runs smoke, then both-model baseline, full ablation, matched ablation in order. Protocol and runner hashes frozen before new inference. Historical files preserved. The inherited matching-engine console message about historical controls is superseded: only STE117 freshly computed full-stage controls were copied within this new experiment.']
    (OUT / 'final_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for rel, digest in protocol['source_sha256'].items():
        assert sha(ROOT / rel) == digest
    save('verification.json', dict(complete=True, baseline_predictions=12694, ablation_score_groups=len(cached), ablation_predictions=len(predrows), missing_scores=0, missing_seeds=0, matching_comparisons=120000, matching_failures=sum((not r['matching_ok'] for r in matchingrows)), historical_inputs_unchanged=True, point_metric_sklearn_max_gap=max_point_gap, output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json'}))
    print('STE117 report complete; historical inputs unchanged.', flush=True)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-only', action='store_true')
    args = ap.parse_args()
    main(args.baseline_only)
