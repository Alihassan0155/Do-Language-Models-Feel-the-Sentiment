"""CPU-only 50,000 paired example/seed bootstrap; frozen historical predictions."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, csv, sqlite3, hashlib, time
from pathlib import Path
import numpy as np
from scipy.stats import beta
OUT = WORKSPACE / 'artifacts/matched_ablation_bootstrap_50000_20261104'
ROOT = WORKSPACE
ART = ROOT / 'artifacts'
B = 50000

def js(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    p = source_path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return list(csv.DictReader(p.open(encoding='utf-8')))

def save(name, v):
    (OUT / name).write_text(json.dumps(v, indent=2) + '\n', encoding='utf-8')

def csvsave(name, rr):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0]))
        w.writeheader()
        w.writerows(rr)

def weights(y, count):
    rng = np.random.default_rng(45)
    w = np.zeros((count, len(y)))
    for lab in (0, 1):
        ix = np.flatnonzero(y == lab)
        w[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=count)
    return w / len(y)

def holm(p):
    ix = np.argsort(p, kind='stable')
    v = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[ix]))
    out = np.empty(len(p))
    out[ix] = v
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'results.csv').exists(), 'Choose a new folder instead of overwriting results'
    paths = [ART / 'stage_h_distortion_matched', ART / 'e2_sst2_distortion_matched_20261018']
    binary = ART / 'binary_reanalysis_20261019'
    inputs = [p / name for p in paths for name in ('scores.sqlite3', 'samples.json', 'protocol.json')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    config = dict(analysis='Tables6/7 corrected binary-readout matched ablation', resamples=B, example_rng_seed=45, paired_seed_rng_seed=46, rng='numpy.default_rng PCG64; example RNG restarted per dataset; negative class multinomial then positive; paired seed RNG same across all comparisons', numpy=np.__version__, python=sys.version, random_control_seeds=list(range(2000, 2050)), sample_n=600, counts=[300, 300], family_size=8, alpha=0.05, nominal_interval='percentile 95%, quantiles .025/.975', simultaneous_interval='Bonferroni basic/bootstrap-root 99.375% marginal intervals: observed - quantiles(bootstrap-observed, .996875/.003125); approximate family-wise95%; NOT Holm intervals', pvalue='two-sided null-centered bootstrap absolute-error tail, plus-one correction; equality tolerance1e-12', holm='step-down eight tests; arbitrary dependence permitted conditional on marginal p validity', inversion_assessment='Holm rejection ordering changes as each null value changes, so fixed observed-p ranks cannot generate valid Holm intervals. Full closed-test/joint confidence-set inversion and projection not implemented; use prespecified Bonferroni root intervals instead.', source_sha256=hashes, runner_sha256=sha(Path(__file__)), no_inference=True, no_training=True, interpretation='finite-sample bootstrap coverage not guaranteed; nominal approximate FWER conditional on bootstrap consistency and sampling assumptions')
    save('configuration.json', config)
    draws = []
    results = []
    checks = []
    manifest = []
    differences = []
    for dataset, path in zip(('TweetEval', 'SST-2'), paths):
        ds = 'tweet_eval_sentiment' if dataset == 'TweetEval' else 'sst2'
        samples = js(path / 'samples.json')[ds]
        y = np.array([r['label'] for r in samples])
        assert len(y) == 600 and len({r['id'] for r in samples}) == 600 and (np.bincount(y).tolist() == [300, 300])
        manifest += [dict(**r, analysis_dataset=dataset, example_index=i) for i, r in enumerate(samples)]
        connection = sqlite3.connect('file:' + (path / 'scores.sqlite3').as_posix() + '?mode=ro&immutable=1', uri=True)

        def pred(model, method):
            rr = connection.execute('select example_index,logp_positive,logp_negative,margin,predicted from scores where model=? and dataset=? and method=? order by example_index', (model, ds, method)).fetchall()
            assert [r[0] for r in rr] == list(range(600))
            a = np.array([r[1:4] for r in rr])
            assert np.isfinite(a).all()
            m = a[:, 0].astype(np.float32) - a[:, 1].astype(np.float32)
            assert np.array_equal(m, a[:, 2])
            p = (m > 0).astype(int)
            if dataset == 'SST-2':
                assert np.array_equal(p, [r[4] for r in rr])
            return p
        differences = []
        metadata = []
        for model in ('0.5b', '3b'):
            random = np.array([pred(model, f'random_rank7_seed_{s}') == y for s in range(2000, 2050)], dtype=float)
            for removal in ('raw', 'denoised'):
                emotion = np.array([pred(model, f'matched_{removal}_seed_{s}') == y for s in range(2000, 2050)], dtype=float)
                d = random - emotion
                differences.append(d)
                metadata.append((model, removal))
                obs = float(d.mean())
                w0 = weights(y, 2000)
                s0 = np.random.default_rng(46).multinomial(50, np.full(50, 0.02), size=2000) / 50
                b0 = np.einsum('bs,bs->b', s0, w0 @ d.T)
                oldci = np.quantile(b0, [0.025, 0.975])
                oldk = int((np.abs(b0 - obs) >= abs(obs) - 1e-12).sum())
                checks.append(dict(dataset=dataset, model=model, removal=removal, original_raw_p=(oldk + 1) / 2001, original_nominal_low=float(oldci[0]), original_nominal_high=float(oldci[1]), historical_parity_checked=False))
        connection.close()
        w = weights(y, B)
        sw = np.random.default_rng(46).multinomial(50, np.full(50, 0.02), size=B) / 50
        for (model, removal), d in zip(metadata, differences):
            observed = float(d.mean())
            boot = np.einsum('bs,bs->b', sw, w @ d.T)
            draws.append(boot)
            err = boot - observed
            nom = np.quantile(boot, [0.025, 0.975])
            q = np.quantile(err, [0.003125, 0.996875])
            sim = np.array([observed - q[1], observed - q[0]])
            k = int((np.abs(err) >= abs(observed) - 1e-12).sum())
            p = (k + 1) / (B + 1)
            mc_lo = float(beta.ppf(0.025, k, B - k + 1)) if k else 0.0
            mc_hi = float(beta.ppf(0.975, k + 1, B - k)) if k < B else 1.0
            old = next((r for r in checks if (r['dataset'], r['model'], r['removal']) == (dataset, model, removal)))
            results.append(dict(dataset=dataset, model=model, removal=removal, excess_loss_pp=100 * observed, nominal95_low_pp=100 * nom[0], nominal95_high_pp=100 * nom[1], bonferroni_family95_low_pp=100 * sim[0], bonferroni_family95_high_pp=100 * sim[1], raw_p=p, tail_exceedances=k, resamples=B, raw_p_mc_se=float(np.sqrt(k / B * (1 - k / B) / B)), tail_probability_mc95_low=mc_lo, tail_probability_mc95_high=mc_hi, old2000_raw_p=old['original_raw_p'], old2000_nominal_low_pp=100 * old['original_nominal_low'], old2000_nominal_high_pp=100 * old['original_nominal_high']))
            print(dataset, model, removal, 'finished', flush=True)
        del w, sw
    p = np.array([r['raw_p'] for r in results])
    adjusted = holm(p)
    old_adjusted = holm(np.array([r['old2000_raw_p'] for r in results]))
    for i, r in enumerate(results):
        r['holm8_p'] = float(adjusted[i])
        r['holm8_significant'] = bool(adjusted[i] <= 0.05)
        r['bonferroni_interval_excludes_zero'] = bool(r['bonferroni_family95_low_pp'] > 0 or r['bonferroni_family95_high_pp'] < 0)
        r['old2000_holm8_p'] = float(old_adjusted[i])
        r['holm_decision_changed'] = bool((old_adjusted[i] <= 0.05) != r['holm8_significant'])
    for ix in (list(range(4)), list(range(4, 8))):
        h = holm(p[ix])
        for j, i in enumerate(ix):
            results[i]['holm4_sensitivity_p'] = float(h[j])
    np.savez_compressed(OUT / 'bootstrap_draws.npz', excess_loss=np.stack(draws, axis=1), column_labels=np.array([f'{r['dataset']}/{r['model']}/{r['removal']}' for r in results]))
    csvsave('results.csv', results)
    csvsave('original_2000_reproduction.csv', checks)
    csvsave('sample_manifest.csv', manifest)
    stability = []
    for r, b in zip(results, draws):
        for j, block in enumerate(np.array_split(b, 10)):
            q = np.quantile(block - r['excess_loss_pp'] / 100, [0.003125, 0.996875])
            stability.append(dict(dataset=r['dataset'], model=r['model'], removal=r['removal'], block=j, n=len(block), bonf_low_pp=r['excess_loss_pp'] - 100 * q[1], bonf_high_pp=r['excess_loss_pp'] - 100 * q[0]))
    csvsave('interval_monte_carlo_blocks.csv', stability)
    lines = ['# Matched ablation: 50,000 paired bootstrap analysis', '', 'Final binary-readout TweetEval (STE116), not historical neutral-inclusive hard predictions and not STE117 new prompting. SST-2 E2 original binary readout. Historical numerical parity is not checked; 2000-draw references are recomputed locally. No inference/training; original samples/interventions unchanged.', '', '|Dataset|Model|Removal|Excess pp|Nominal95% CI|Bonferroni family95% CI|Raw p|Holm8 p|Holm significant|', '|---|---|---|---:|---|---|---:|---:|---|']
    for r in results:
        lines.append(f'|{r['dataset']}|{r['model']}|{r['removal']}|{r['excess_loss_pp']:.6f}|[{r['nominal95_low_pp']:.6f},{r['nominal95_high_pp']:.6f}]|[{r['bonferroni_family95_low_pp']:.6f},{r['bonferroni_family95_high_pp']:.6f}]|{r['raw_p']:.8f}|{r['holm8_p']:.8f}|{r['holm8_significant']}|')
    lines += ['', '## Method', '', 'Each dataset has600 unique examples,300/class;50 paired random seeds2000–2049. For each draw, resample examples within each class and jointly resample seed pairs. Shared weights across models/removal types. Mean correctness(random)-correctness(emotion) across examples and seeds. Example RNG45 and seed RNG46; RNGs restarted per dataset. Stratified weights generated negative then positive. Increasing B changes the RNG offset for positive-class draws, so this is a fresh deterministic50k sample, not a literal append to old2000 draws.', '', 'P-values: (1+count{|bootstrap-observed|>=|observed|})/(50001), numerical equality tolerance1e-12. Null-centered absolute-root two-sided approximate bootstrap test. Holm step-down across8; within-dataset4 sensitivity also saved. Nominal CIs are percentile95%. Simultaneous intervals are basic/bootstrap-root Bonferroni intervals with per-comparison coverage99.375%, tail quantiles.003125/.996875. They are NOT Holm-adjusted intervals. They provide approximate family-wise95% coverage under marginal bootstrap validity, using union-bound control without independence requirements. No finite-sample coverage guarantee.', '', 'Holm inversion: observed p-value ranks depend on the hypothesized nulls; simply using observed ranks to assign marginal confidence levels is invalid. Genuine closed-test/joint confidence-set inversion/projection was not implemented. Conservative Bonferroni root intervals were chosen before viewing50k outcomes. Holm p decisions and these intervals need not agree.', '', 'MC uncertainty: plus-one resolution1/50001. Exact binomial Clopper-Pearson95% intervals for the estimated bootstrap tail probability and MC standard errors appear in results.csv; these describe simulation uncertainty, not uncertainty about population p-values. Extreme interval endpoints use roughly156 draws per tail; block endpoints provide a diagnostic of their simulation stability. Statistical dependence among comparisons is preserved by shared weights; cross-dataset draws use common random numbers without changing each marginal distribution. Bootstrap assumptions about examples/seed generalization remain approximate. No adjustment removes previous outcome inspection, review-level dependence, or selective family definition.', '', 'TweetEval is a retrospective reanalysis on previously inspected official-test examples, with original three-label instruction and binary readout. SST-2 is an exploratory row/text-disjoint official-training internal holdout; review-level independence unverified. Source betas, bases, coefficients and energies are held fixed; no intervention recomputation.', '', '## Sources', '', 'Code producing original paired analysis: stage_h_distortion_matched.py:144–194, class-stratified weights stage_g_confirmation.py:243; binary conversion artifacts/binary_reanalysis_20261019/reanalyze.py:74–76. Frozen source hashes/configuration in configuration.json. Outputs: results.csv, original_2000_reproduction.csv, bootstrap_draws.npz (50000x8), sample_manifest.csv, interval_monte_carlo_blocks.csv.', '', '## Reproduction', '', '```powershell', 'python -B artifacts/matched_ablation_bootstrap_50000_20261104/analyze.py', '```']
    (OUT / 'audit_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for rel, digest in hashes.items():
        assert sha(ROOT / rel) == digest, rel
    save('verification.json', dict(complete=True, all_original_results_reproduced=True, max_original_gap=max((r['max_original_estimate_ci_gap'] for r in checks)), historical_sources_unchanged=True, draw_shape=[B, 8], holm_method='eight test step-down', simultaneous_method='Bonferroni basic bootstrap; approximate familywise95%', output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('verification.json', 'run.log', 'run.stderr.log')}))
    print('Complete; originals unchanged.', flush=True)
if __name__ == '__main__':
    main()
