"""ASD-STE116: CPU-only binary readout of immutable historical scores."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import csv, hashlib, json, sqlite3, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
OUT = WORKSPACE / 'artifacts/binary_reanalysis_20261019'
ROOT = WORKSPACE
ART = ROOT / 'artifacts'
DS = 'tweet_eval_sentiment'
MODELS = ('0.5b', '3b')

def js(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    p = source_path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, v):
    (OUT / name).write_text(json.dumps(v, indent=2) + '\n', encoding='utf-8')

def csvout(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def ro(p):
    c = sqlite3.connect('file:' + p.as_posix() + '?mode=ro&immutable=1', uri=True)
    c.row_factory = sqlite3.Row
    return c

def weights(y, n, seed):
    rng = np.random.default_rng(seed)
    w = np.zeros((n, len(y)))
    for label in (0, 1):
        ix = np.flatnonzero(y == label)
        w[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=n)
    return w / len(y)

def metrics(y, p, m, w):
    acc = w @ (p == y)
    fs = []
    for lab in (0, 1):
        tp = w @ ((p == lab) & (y == lab))
        den = w @ (p == lab) + w @ (y == lab)
        fs.append(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0))
    order = np.argsort(m, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(m[order]) != 0) + 1]
    pos = np.add.reduceat(w[:, order] * (y[order] == 1), starts, axis=1)
    neg = np.add.reduceat(w[:, order] * (y[order] == 0), starts, axis=1)
    auc = (pos * (neg.cumsum(axis=1) - neg + 0.5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))
    return np.stack([acc, (fs[0] + fs[1]) / 2, auc], axis=1)

def interval(a):
    return [float(x) for x in np.quantile(a, [0.025, 0.975])]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'binary_reanalysis_protocol.json').exists(), 'Use a new output directory; do not overwrite a completed reanalysis.'
    paths = [ART / x for x in ('stage_h', 'stage_h_distortion_matched')]
    configs = [js(p / 'protocol.json') for p in paths]
    samples = [js(p / 'samples.json')[DS] for p in paths]
    assert samples[0] == samples[1]
    rows = samples[0]
    y = np.array([r['label'] for r in rows])
    n = len(y)
    assert n == 600 and len({r['id'] for r in rows}) == 600 and (np.bincount(y).tolist() == [300, 300])
    official = {r['id']: r for r in map(json.loads, (ART / 'stage_e/data/tweet_eval_sentiment/test.jsonl').read_text(encoding='utf-8').splitlines())}
    assert all((all((r[k] == official[r['id']][k] for k in ('text', 'label', 'source_split', 'source_row_index'))) for r in rows))
    seeds = configs[0]['random_seeds']
    assert seeds == configs[1]['random_seeds'] == list(range(2000, 2050))
    assert all((c['bootstrap_draws'] == 2000 for c in configs))
    assert configs[0]['bootstrap_seed'] == 44 and configs[1]['bootstrap_seed'] == 45
    for rel, digest in configs[1]['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    prompt = configs[0]['models']['0.5b']['prompts'][DS]
    assert 'neutral' in prompt
    assert all((configs[0]['models'][m]['prompts'][DS] == prompt for m in MODELS))
    protected = []
    for p in paths:
        protected += [p / x for x in ('protocol.json', 'samples.json', 'scores.sqlite3')]
        protected += list(p.glob('*.csv')) + list(p.glob('*.txt')) + list(p.glob('state.json'))
    protected += [paths[1] / 'matching.sqlite3', ROOT / 'stage_h.py', ROOT / 'stage_h_distortion_matched.py', ROOT / 'stage_g_confirmation.py']
    for m in MODELS:
        protected += list((paths[0] / m / DS).glob('bases*'))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    caches = [ro(p / 'scores.sqlite3') for p in paths]
    audit = ro(paths[1] / 'matching.sqlite3')
    cache = {}
    auditrows = []
    availability = []
    max_metric_gap = 0.0
    changed = []
    predictions = []
    names0 = ['baseline', 'emotion_raw', 'emotion_denoised'] + [f'random_rank7_seed_{s}' for s in seeds]
    names1 = names0 + ['baseline_parity'] + [f'matched_{k}_seed_{s}' for s in seeds for k in ('raw', 'denoised')]
    for stage, names in enumerate((names0, names1)):
        for model in MODELS:
            for name in names:
                rr = [dict(x) for x in caches[stage].execute('SELECT * FROM scores WHERE model=? AND dataset=? AND method=? ORDER BY example_index', (model, DS, name))]
                assert [x['example_index'] for x in rr] == list(range(n)), (stage, model, name)
                a = np.array([[x['logp_positive'], x['logp_negative'], x['margin']] for x in rr])
                assert np.isfinite(a).all()
                margin = (a[:, 0].astype(np.float32) - a[:, 1].astype(np.float32)).astype(float)
                assert np.array_equal(margin, a[:, 2]), (model, name, 'saved margin convention mismatch')
                p = (margin > 0).astype(int)
                old = np.array([x['predicted'] for x in rr])
                assert np.array_equal(p, np.argmax(a[:, [1, 0]], axis=1))
                cache[stage, model, name] = (rr, p, margin, old)
                availability.append(dict(stage=('full', 'matched')[stage], model=model, method=name, n=n, finite_scores=True, ties_negative=int((margin == 0).sum())))
                changed.append(dict(stage=('full', 'matched')[stage], model=model, method=name, old_neutral_predictions=int((old == 2).sum()), changed_predictions=int((old != p).sum())))
                for i, x in enumerate(rr):
                    predictions.append(dict(stage=('full', 'matched')[stage], model=model, method=name, example_index=i, id=rows[i]['id'], gold=int(y[i]), logp_positive=x['logp_positive'], logp_negative=x['logp_negative'], margin=float(margin[i]), binary_prediction=int(p[i]), historical_prediction=int(old[i])))
                if stage == 1 and name in names0:
                    assert rr == cache[0, model, name][0], (model, name, 'copied references differ')
            if stage == 1:
                assert np.array_equal(cache[1, model, 'baseline'][2], cache[1, model, 'baseline_parity'][2])
                for seed in seeds:
                    random = cache[1, model, f'random_rank7_seed_{seed}'][0]
                    for kind in ('raw', 'denoised'):
                        name = f'matched_{kind}_seed_{seed}'
                        aa = [dict(x) for x in audit.execute('SELECT * FROM matching WHERE model=? AND method=? ORDER BY example_index', (model, name))]
                        assert [x['example_index'] for x in aa] == list(range(n))
                        target = np.array([x['target_energy'] for x in aa])
                        actual = np.array([x['actual_energy'] for x in aa])
                        beta = np.array([x['beta'] for x in aa])
                        assert np.isfinite(np.c_[target, actual, beta]).all() and ((beta >= 0) & (beta <= 1)).all()
                        assert np.allclose(target, [x['token_removed_energy_fraction'] for x in random], atol=1e-07, rtol=1e-05)
                        assert np.allclose(actual, [x['token_removed_energy_fraction'] for x in cache[1, model, name][0]], atol=1e-07, rtol=1e-05)
                        for i, x in enumerate(aa):
                            gap = abs(x['actual_energy'] - x['target_energy'])
                            ok = bool(x['feasible']) and gap <= configs[1]['match_relative_tolerance'] * x['target_energy'] + configs[1]['match_absolute_tolerance']
                            auditrows.append(dict(**x, id=rows[i]['id'], seed=seed, kind=kind, absolute_energy_gap=gap, matching_ok=ok))
    assert len(auditrows) == 120000
    print('Complete cached-score/provenance audit passed; computing CPU bootstrap only.', flush=True)
    protocol = dict(task='ASD-STE116', dataset=DS, n=n, label_counts={'negative': 300, 'positive': 300}, prompt=prompt, prompt_type='original three-label P1; binary scoring only', models={m: dict(layer=configs[0]['models'][m]['layer'], model_id=configs[0]['models'][m]['model_id']) for m in MODELS}, prediction='M=float32(logPpositive)-float32(logPnegative); positive iff M>0, otherwise negative', random_seeds=seeds, full_bootstrap=dict(draws=2000, example_seed=44, seed_resampling=False, conditional_on_50_fixed_bases=True), matched_bootstrap=dict(draws=2000, example_seed=45, paired_seed_resampling_seed=46, joint_example_seed_resampling=True), bootstrap_rule='original class-stratified multinomial example weights; pointwise percentile 95% CI; no multiplicity correction', aggregation='mean of per-seed metrics, never pooled repeated predictions or ensemble labels', interventions='unchanged cached interventions; no forward passes; original betas and energies unchanged', leakage='historically inspected evaluation set; labels used only for metrics/class stratification and original balanced sampling, not intervention construction or energy matching', source_sha256=before, command='python -B artifacts/binary_reanalysis_20261019/reanalyze.py', versions=dict(python=sys.version, numpy=np.__version__))
    save('binary_reanalysis_protocol.json', protocol)
    csvout('dataset_ids.csv', [dict(example_index=i, **r) for i, r in enumerate(rows)])
    csvout('saved_score_availability.csv', availability)
    csvout('binary_predictions.csv', predictions)
    csvout('matching_diagnostics_unchanged.csv', auditrows)
    csvout('prediction_conversion_diagnostics.csv', changed)
    tables = [[], []]
    cis = []
    variation = []
    diagnostic = []
    names = ('accuracy', 'macro_f1', 'auroc')
    point = np.full((1, n), 1 / n)
    for stage in (0, 1):
        w = weights(y, 2000, 44 + stage)
        sw = np.random.default_rng(46).multinomial(50, np.full(50, 1 / 50), size=2000) / 50 if stage else np.full((2000, 50), 1 / 50)
        for model in MODELS:

            def get(name):
                nonlocal max_metric_gap
                rr, p, m, old = cache[stage, model, name]
                v = metrics(y, p, m, point)[0]
                b = metrics(y, p, m, w)
                expected = np.array([accuracy_score(y, p), f1_score(y, p, average='macro'), roc_auc_score(y, m)])
                gap = float(np.max(np.abs(v - expected)))
                max_metric_gap = max(max_metric_gap, gap)
                assert gap < 1e-12
                oldv = metrics(y, old, m, point)[0]
                diagnostic.append(dict(stage=('full', 'matched')[stage], model=model, method=name, binary_accuracy=float(v[0]), three_candidate_accuracy=float(oldv[0]), binary_macro_f1=float(v[1]), three_candidate_macro_f1=float(oldv[1]), binary_auroc=float(v[2]), historical_auroc=float(oldv[2])))
                return (v, b)
            bp, bb = get('baseline')
            rp = []
            rb = []
            for seed in seeds:
                v, b = get(f'random_rank7_seed_{seed}')
                rp.append(v)
                rb.append(b)
                for j, metric in enumerate(names):
                    variation.append(dict(stage=('full', 'matched')[stage], model=model, condition='random_rank7', seed=seed, metric=metric, performance=float(v[j]), damage=float(bp[j] - v[j]), excess_vs_random=''))
            rp = np.array(rp)
            rb = np.stack(rb, axis=1)
            rmean = rp.mean(axis=0)
            rboot = np.einsum('bs,bsm->bm', sw, rb)
            conditions = {'baseline': (bp, bb, None), 'random_rank7_mean': (rmean, rboot, rp)}
            for kind in ('raw', 'denoised'):
                if stage:
                    ps = []
                    bs = []
                    for i, seed in enumerate(seeds):
                        v, b = get(f'matched_{kind}_seed_{seed}')
                        ps.append(v)
                        bs.append(b)
                        for j, metric in enumerate(names):
                            variation.append(dict(stage='matched', model=model, condition=f'emotion_{kind}', seed=seed, metric=metric, performance=float(v[j]), damage=float(bp[j] - v[j]), excess_vs_random=float(rp[i, j] - v[j])))
                    ps = np.array(ps)
                    conditions[f'emotion_{kind}'] = (ps.mean(axis=0), np.einsum('bs,bsm->bm', sw, np.stack(bs, axis=1)), ps)
                else:
                    v, b = get(f'emotion_{kind}')
                    conditions[f'emotion_{kind}'] = (v, b, None)
                    for i, seed in enumerate(seeds):
                        for j, metric in enumerate(names):
                            variation.append(dict(stage='full', model=model, condition=f'emotion_{kind}', seed=seed, metric=metric, performance=float(v[j]), damage=float(bp[j] - v[j]), excess_vs_random=float(rp[i, j] - v[j])))
            for condition, (v, b, ps) in conditions.items():
                rec = dict(model=model, condition=condition, n=n, random_controls=50)
                for j, metric in enumerate(names):
                    damage = float(bp[j] - v[j])
                    excess = float(rmean[j] - v[j]) if condition.startswith('emotion') else ''
                    rec[metric] = float(v[j])
                    rec[metric + '_damage'] = damage
                    rec[metric + '_excess_vs_random'] = excess
                    for stat, value, draw in [('performance', v[j], b[:, j]), ('damage', damage, bb[:, j] - b[:, j])] + ([('excess_vs_random', excess, rboot[:, j] - b[:, j])] if condition.startswith('emotion') else []):
                        lo, hi = interval(draw)
                        cis.append(dict(stage=('full', 'matched')[stage], model=model, condition=condition, metric=metric, statistic=stat, value=float(value), ci_low=lo, ci_high=hi, bootstrap_draws=2000))
                tables[stage].append(rec)
            print(('full', 'matched')[stage], model, 'complete', flush=True)
    for c in caches:
        c.close()
    audit.close()
    csvout('binary_full_ablation_results.csv', tables[0])
    csvout('binary_matched_ablation_results.csv', tables[1])
    csvout('binary_confidence_intervals.csv', cis)
    csvout('binary_seed_variation.csv', variation)
    csvout('historical_three_candidate_diagnostic.csv', diagnostic)
    sdrows = []
    for stage in ('full', 'matched'):
        for model in MODELS:
            for condition in ('random_rank7', 'emotion_raw', 'emotion_denoised'):
                for metric in names:
                    rr = [x for x in variation if (x['stage'], x['model'], x['condition'], x['metric']) == (stage, model, condition, metric)]
                    sdrows.append(dict(stage=stage, model=model, condition=condition, metric=metric, performance_seed_sd=float(np.std([x['performance'] for x in rr], ddof=1)), excess_seed_sd=float(np.std([x['excess_vs_random'] for x in rr], ddof=1)) if condition.startswith('emotion') else ''))
    csvout('binary_seed_variation_summary.csv', sdrows)
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert before == after, 'Historical source changed'
    valid = sum((not x['matching_ok'] for x in auditrows))
    report = ['# ASD-STE116 — Binary-only scoring of historical TweetEval ablations', '', 'Cached-score reanalysis only; zero inference or intervention changes. Original P1 still requests negative, neutral, or positive. This is NOT genuinely binary prompting.', '', f'600 identical unique official test examples: 300 negative/300 positive; original seed pairs 2000–2049. Matching failures: {valid}/120000. Historical sample was previously inspected.', '', f'Prompt: `{prompt!r}`', '', 'Full ablation: 2,000 paired class-stratified example draws, seed44; random bases fixed. Matched ablation: 2,000 paired stratified example draws seed45 and joint paired-seed draws seed46. Means average seed metrics, not repeated-example pooling. Nominal percentile 95% CIs, no multiplicity correction.', '']
    for stage, label in enumerate(('Full ablation', 'Energy-matched ablation')):
        report += ['## ' + label, '', '|Model|Condition|Accuracy %|Macro-F1 %|AUROC|Accuracy damage pp|Excess accuracy loss pp [95% CI]|', '|---|---|---:|---:|---:|---:|---|']
        for r in tables[stage]:
            excess = '—'
            if r['condition'].startswith('emotion'):
                ci = next((x for x in cis if (x['stage'], x['model'], x['condition'], x['metric'], x['statistic']) == (('full', 'matched')[stage], r['model'], r['condition'], 'accuracy', 'excess_vs_random')))
                excess = f'{100 * ci['value']:+.4f} [{100 * ci['ci_low']:+.4f}, {100 * ci['ci_high']:+.4f}]'
            report.append(f'|{r['model']}|{r['condition']}|{100 * r['accuracy']:.4f}|{100 * r['macro_f1']:.4f}|{r['auroc']:.6f}|{100 * r['accuracy_damage']:+.4f}|{excess}|')
    report += ['', '## Interpretation and manuscript changes', '', 'Use the new binary results for binary ablation claims; retain historical three-candidate results only as a separately labeled diagnostic. Do not substitute the separate 6,347-example baseline. Reassess each emotion/model conclusion using its new excess-loss CI. A positive lower bound supports excess loss relative to these controls; a crossing-zero CI is inconclusive. Energy matching does not prove direction-specific necessity.', '', 'All labels are held fixed and used for evaluation and class-stratified resampling only in this reanalysis. Original matching code chooses beta without labels; historical outcome inspection cannot be undone. Prompts, layers, bases, strengths, betas, target/actual energies and seed pairing were not changed.', '', 'Historical comparison uses cached three-candidate predictions in historical_three_candidate_diagnostic.csv. Previously reported +2.39/+1.37 pp must not be labeled binary results.', '', '## Reproduction', '', '```powershell', protocol['command'], '```', '', 'All inputs and SHA256 hashes: binary_reanalysis_protocol.json. Membership: dataset_ids.csv. Exact converted predictions: binary_predictions.csv. Original beta/energy records: matching_diagnostics_unchanged.csv. Full confidence intervals and sample SDs are in the named CSV outputs.']
    (OUT / 'binary_reanalysis_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    save('validation.json', dict(complete=True, n=600, unique_ids=600, negative=300, positive=300, score_groups=len(availability), finite_scores=True, missing_seeds=0, source_labels_texts_ids_verified=True, reference_scores_identical=True, matching_comparisons=len(auditrows), matching_failures=valid, infeasible=sum((not x['feasible'] for x in auditrows)), max_absolute_energy_gap=max((x['absolute_energy_gap'] for x in auditrows)), point_metric_sklearn_max_gap=max_metric_gap, historical_hashes_unchanged=before == after, output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()}))
    print('Binary reanalysis complete. Historical hashes unchanged.', flush=True)
if __name__ == '__main__':
    main()
