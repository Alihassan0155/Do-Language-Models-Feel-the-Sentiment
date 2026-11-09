"""Independently check saved E2 metrics, export predictions, and write the report."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
OUT = WORKSPACE / 'artifacts/e2_sst2_distortion_matched_20261018'
ROOT = WORKSPACE
sys.path.insert(0, str(OUT))
import run_e2

def write(name, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with (OUT / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def verify_intervals(frame, saved, protocol, sample):
    """Replay requested intervals from prediction indicators, without H.metrics."""
    y = np.array([r['label'] for r in sample])
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    w = np.zeros((2000, 600), dtype=np.float64)
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        w[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=2000)
    w /= 600
    sw = np.random.default_rng(protocol['seed_bootstrap_seed']).multinomial(50, np.full(50, 1 / 50), size=2000) / 50
    checks = []
    for key in ('0.5b', '3b'):
        groups = {name: g.sort_values('example_index').predicted.to_numpy() for name, g in frame[frame.model == key].groupby('method')}
        columns = {'baseline': ['baseline'], 'random_full': [f'random_rank7_seed_{s}' for s in protocol['random_seeds']], 'emotion_raw_full_reference': ['emotion_raw'], 'emotion_denoised_full_reference': ['emotion_denoised']}
        for kind in ('raw', 'denoised'):
            columns[f'emotion_{kind}_matched'] = [f'matched_{kind}_seed_{s}' for s in protocol['random_seeds']]
        distributions = {}
        for condition, names in columns.items():
            pred = np.column_stack([groups[n] for n in names])
            acc = w @ (pred == y[:, None])
            pp = w @ (pred == 1)
            tp = w @ ((pred == 1) & (y[:, None] == 1))
            tn = w @ ((pred == 0) & (y[:, None] == 0))
            f1 = (2 * tp / (0.5 + pp) + 2 * tn / (0.5 + 1 - pp)) / 2
            distributions[condition] = {metric: (values * sw).sum(axis=1) if len(names) == 50 else values[:, 0] for metric, values in [('accuracy', acc), ('macro_f1', f1)]}
        for _, row in saved[(saved.model == key) & saved.metric.isin(['accuracy', 'macro_f1'])].iterrows():
            values = distributions[row.condition][row.metric]
            quantities = {'performance': (values, row.performance_ci_low, row.performance_ci_high), 'damage': (distributions['baseline'][row.metric] - values, row.damage_ci_low, row.damage_ci_high)}
            if row.condition.endswith('_matched'):
                quantities['excess'] = (distributions['random_full'][row.metric] - values, row.excess_ci_low, row.excess_ci_high)
            for quantity, (distribution, lo, hi) in quantities.items():
                expected = np.array([lo, hi])
                actual = np.quantile(distribution, [0.025, 0.975])
                if not np.allclose(actual, expected, atol=1e-11, rtol=0):
                    raise ValueError(f'Independent interval mismatch: {key}/{row.condition}/{row.metric}/{quantity}')
                checks.append(dict(model=key, condition=row.condition, metric=row.metric, quantity=quantity, independent_low=float(actual[0]), independent_high=float(actual[1]), saved_low=float(lo), saved_high=float(hi), max_absolute_gap=float(np.abs(actual - expected).max())))
    write('independent_interval_checks.csv', checks)
    return checks

def main():
    run_e2.verify_inputs()
    protocol = run_e2.read(OUT / 'protocol.json')
    sample = run_e2.read(OUT / 'samples.json')['sst2']
    state = run_e2.read(OUT / 'state.json')
    assert state['complete'] and len(sample) == 600
    db = sqlite3.connect('file:' + (OUT / 'scores.sqlite3').as_posix() + '?mode=ro&immutable=1', uri=True)
    frame = pd.read_sql_query('SELECT * FROM scores ORDER BY model,method,example_index', db)
    db.close()
    assert len(frame) == 184800 and set(frame.dataset) == {'sst2'}
    assert set(frame.predicted) <= {0, 1}
    frame['id'] = [sample[int(i)]['id'] for i in frame.example_index]
    frame['gold_label'] = [sample[int(i)]['label'] for i in frame.example_index]
    calculated = {}
    for (model, method), g in frame.groupby(['model', 'method'], sort=False):
        assert sorted(g.example_index) == list(range(600))
        calculated[model, method] = dict(accuracy=accuracy_score(g.gold_label, g.predicted), macro_f1=f1_score(g.gold_label, g.predicted, labels=[0, 1], average='macro', zero_division=0), auroc=roc_auc_score(g.gold_label, g.margin))
    frame.to_csv(OUT / 'predictions.csv', index=False)
    saved = pd.read_csv(OUT / 'performance_paired_ci.csv')
    comparisons = []
    for _, row in saved.iterrows():
        key, condition, metric = (row.model, row.condition, row.metric)
        if condition == 'baseline':
            names = ['baseline']
        elif condition == 'random_full':
            names = [f'random_rank7_seed_{s}' for s in protocol['random_seeds']]
        elif condition.endswith('_matched'):
            kind = condition.split('_')[1]
            names = [f'matched_{kind}_seed_{s}' for s in protocol['random_seeds']]
        else:
            names = [{'emotion_raw_full_reference': 'emotion_raw', 'emotion_denoised_full_reference': 'emotion_denoised'}[condition]]
        actual = float(np.mean([calculated[key, name][metric] for name in names]))
        if not np.isclose(actual, row.performance, atol=1e-12, rtol=0):
            raise ValueError(f'Independent metric mismatch: {key}/{condition}/{metric}')
        comparisons.append(dict(model=key, condition=condition, metric=metric, independent=actual, saved=float(row.performance), absolute_gap=abs(actual - row.performance)))
    write('independent_metric_checks.csv', comparisons)
    interval_checks = verify_intervals(frame, saved, protocol, sample)
    audit = sqlite3.connect('file:' + (OUT / 'matching.sqlite3').as_posix() + '?mode=ro&immutable=1', uri=True)
    match = pd.read_sql_query('SELECT * FROM matching ORDER BY model,method,example_index', audit)
    audit.close()
    assert len(match) == 120000
    match['id'] = [sample[int(i)]['id'] for i in match.example_index]
    match['absolute_energy_gap'] = abs(match.actual_energy - match.target_energy)
    match['relative_energy_gap'] = match.absolute_energy_gap / np.maximum(match.target_energy, 1e-12)
    match['matching_ok'] = match.feasible.astype(bool) & (match.absolute_energy_gap <= protocol['match_relative_tolerance'] * match.target_energy + protocol['match_absolute_tolerance'])
    match.to_csv(OUT / 'energy_matching_per_example.csv', index=False)
    match[~match.matching_ok].to_csv(OUT / 'unmatched_energy_cases.csv', index=False)
    matching_summary = []
    for key in ('0.5b', '3b'):
        for kind in ('raw', 'denoised'):
            g = match[(match.model == key) & match.method.str.startswith('matched_' + kind + '_')]
            matching_summary.append(dict(model=key, space=kind, comparisons=len(g), passed=int(g.matching_ok.sum()), infeasible=int((g.feasible == 0).sum()), failed=int((~g.matching_ok).sum()), mean_beta=float(g.beta.mean()), max_absolute_energy_gap=float(g.absolute_energy_gap.max()), max_relative_energy_gap=float(g.relative_energy_gap.max()), mean_target_energy_percent=100 * float(g.target_energy.mean()), mean_actual_energy_percent=100 * float(g.actual_energy.mean())))
    write('matching_summary.csv', matching_summary)
    pairs = pd.read_csv(OUT / 'seed_pair_comparisons.csv')
    spread = []
    for (key, kind, metric), g in pairs.groupby(['model', 'kind', 'metric']):
        assert len(g) == 50
        spread.append(dict(model=key, space=kind, metric=metric, seeds=50, mean_excess_pp=100 * float(g.excess_damage.mean()), sd_excess_pp=100 * float(g.excess_damage.std(ddof=1)), min_excess_pp=100 * float(g.excess_damage.min()), max_excess_pp=100 * float(g.excess_damage.max())))
    write('paired_excess_seed_variation.csv', spread)
    seed_variation = pd.read_csv(OUT / 'between_seed_variation.csv')
    table = []
    for key in ('0.5b', '3b'):
        for condition in ('baseline', 'random_full', 'emotion_raw_matched', 'emotion_denoised_matched', 'emotion_raw_full_reference', 'emotion_denoised_full_reference'):
            acc = saved[(saved.model == key) & (saved.condition == condition) & (saved.metric == 'accuracy')].iloc[0]
            f1 = saved[(saved.model == key) & (saved.condition == condition) & (saved.metric == 'macro_f1')].iloc[0]
            seed = seed_variation[(seed_variation.model == key) & (seed_variation.condition == condition) & (seed_variation.metric == 'accuracy_damage_pp')]
            table.append(dict(model=key, condition=condition, n=600, accuracy_percent=100 * float(acc.performance), accuracy_ci_low_percent=100 * float(acc.performance_ci_low), accuracy_ci_high_percent=100 * float(acc.performance_ci_high), macro_f1_percent=100 * float(f1.performance), macro_f1_ci_low_percent=100 * float(f1.performance_ci_low), macro_f1_ci_high_percent=100 * float(f1.performance_ci_high), accuracy_loss_pp=100 * float(acc.damage), excess_loss_pp=100 * float(acc.excess_vs_random) if pd.notna(acc.excess_vs_random) else None, excess_ci_low_pp=100 * float(acc.excess_ci_low) if pd.notna(acc.excess_ci_low) else None, excess_ci_high_pp=100 * float(acc.excess_ci_high) if pd.notna(acc.excess_ci_high) else None, accuracy_between_seed_sd_pp=float(seed.iloc[0]['sd']) if len(seed) else None))
    write('results_table.csv', table)
    verdicts = pd.read_csv(OUT / 'historical_comparison.csv')
    lines = ['# E2 — SST-2 distortion-matched ablation', '', 'Exploratory internal-holdout extension. 600 new official-training source rows (300/class); zero audited source-ID or normalized-text overlap with prior saved uses. Review-level independence and undocumented prior external inspection remain unverified.', '', 'Frozen models: Qwen2.5-0.5B-Instruct at layer 14 and Qwen2.5-3B-Instruct at layer 18. Original SST-2 P1; negative/positive candidate-label argmax. No training, tuning, or outcome-based sample/seed selection.', '', '50 paired random rank-seven controls, seeds 2000–2049. Original rounded per-example energy matching, beta capped at one, 16 bisection steps, tolerance 1% relative + 1e-7 absolute. 2,000 paired class-stratified/example-and-seed bootstrap draws (example seed 45, seed-resampling seed 46), percentile 95% CIs; no multiplicity correction.', '', '|Model|Condition|Accuracy (%)|Macro-F1 (%)|Excess loss vs random (pp), 95% CI|Accuracy seed SD (pp)|', '|---|---|---:|---:|---|---:|']
    for row in table:
        excess = '—' if row['excess_loss_pp'] is None else f'{row['excess_loss_pp']:+.4f} [{row['excess_ci_low_pp']:+.4f}, {row['excess_ci_high_pp']:+.4f}]'
        sd = '—' if row['accuracy_between_seed_sd_pp'] is None else f'{row['accuracy_between_seed_sd_pp']:.4f}'
        lines.append(f'|{row['model']}|{row['condition']}|{row['accuracy_percent']:.4f}|{row['macro_f1_percent']:.4f}|{excess}|{sd}|')
    lines += ['', '## Matching checks', '', json.dumps(matching_summary, indent=2), '', '## Historical comparison', '']
    for _, row in verdicts.iterrows():
        lines.append(f'- {row.model}: {row.assessment}. SST-2 raw excess {row.sst2_excess_pp:+.4f} pp [{row.sst2_ci_low_pp:+.4f}, {row.sst2_ci_high_pp:+.4f}], versus historical TweetEval {row.tweeteval_excess_pp:+.4f} pp.')
    lines += ['', '## Implementation differences', '', '- Dataset: new balanced 600-row SST-2 internal holdout, not the historical 600-row TweetEval sample; same methods and 50 bases.', '- SST-2 P1 and candidate readout are binary (negative/positive); historical TweetEval retains a three-label prompt and neutral-inclusive candidate readout on binary gold examples. Cross-dataset differences are not a controlled domain-only comparison.', '- Existing TweetEval matching/report code imported without editing; dataset constant redirected to SST-2 in the new adapter.', '- Historical reference-copy function disabled: all baseline, full raw/denoised, random and matched intervention predictions computed anew.', '- The inherited execution status line says historical controls were reused; this legacy message does not apply to E2. The frozen protocol and empty_reference adapter document no prediction reuse.', '- Same FP16, layers, bases, batch size 8, matching iterations/tolerance and bootstrap settings. New batches are frozen four-negative/four-positive batches.', '- Added independent sklearn metric checks, per-example matching exports and separate across-seed excess SDs. No parameter changes after outcomes.', '', 'All historical input hashes were verified unchanged after inference and reporting. The archived source code and input hashes permit reproduction without editing historical artifacts.', '', 'Commands (working directory WORK_DIR):', '```powershell', 'python -B artifacts/e2_sst2_distortion_matched_20261018/run_e2.py --phase prepare', 'python -B artifacts/e2_sst2_distortion_matched_20261018/run_e2.py --phase smoke', 'python -B artifacts/e2_sst2_distortion_matched_20261018/run_e2.py --phase run', 'python -B artifacts/e2_sst2_distortion_matched_20261018/report_e2.py', '```']
    (OUT / 'E2_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    run_e2.verify_inputs()
    exports = ['scores.sqlite3', 'matching.sqlite3', 'predictions.csv', 'samples.json', 'dataset_ids.csv', 'protocol.json', 'performance_paired_ci.csv', 'results_table.csv', 'matching_summary.csv', 'energy_matching_per_example.csv', 'unmatched_energy_cases.csv', 'paired_excess_seed_variation.csv', 'independent_metric_checks.csv', 'independent_interval_checks.csv', 'historical_comparison.csv', 'E2_REPORT.md', 'run_e2.py', 'report_e2.py']
    run_e2.save('verification.json', dict(complete=True, independently_verified_prediction_sets=len(calculated), metric_checks=len(comparisons), metric_max_abs_gap=max((r['absolute_gap'] for r in comparisons)), interval_checks=len(interval_checks), interval_max_abs_gap=max((r['max_absolute_gap'] for r in interval_checks)), matching_comparisons=len(match), matching_failures=int((~match.matching_ok).sum()), historical_inputs_unchanged=True, output_sha256={name: run_e2.sha(OUT / name) for name in exports}))
    print('E2 independently verified. See E2_REPORT.md and results_table.csv.', flush=True)
if __name__ == '__main__':
    main()
