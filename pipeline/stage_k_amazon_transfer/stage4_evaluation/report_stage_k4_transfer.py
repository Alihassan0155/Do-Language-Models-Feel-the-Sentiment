"""Read-only analysis of saved K4/source scores; writes new K4 reporting only."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os, sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import pandas as pd, numpy as np
R = WORKSPACE
O = R / 'artifacts/stage_k_amazon_transfer/stage4_evaluation'

def main():
    amazon = pd.read_csv(O / 'all_method_metrics.csv', float_precision='round_trip')
    primary = pd.read_csv(O / 'primary_comparisons.csv', float_precision='round_trip')
    secondary = pd.read_csv(O / 'secondary_comparisons.csv', float_precision='round_trip')
    correction = pd.read_csv(O / 'correction_analysis.csv', float_precision='round_trip')
    comparisons = []
    for source, folder in [('sst2', 'stage_i_sst2'), ('tweeteval', 'stage_i_tweeteval_matched_controls')]:
        saved = pd.read_csv(R / 'artifacts' / folder / 'metrics_all_methods.csv', float_precision='round_trip')
        saved = saved[saved['split'] == 'test']
        for model in ('0.5b', '3b'):
            original = saved[(saved['model'] == model) & (saved['method'] == 'original')].iloc[0]
            raw = saved[(saved['model'] == model) & (saved['method'] == 'raw')].iloc[0]
            config = source + '_' + model
            am = amazon[amazon.configuration == config].set_index('method')
            for baseline in ('original', 'sentiment_only', 'pca_rank7'):
                old = saved[(saved['model'] == model) & (saved['method'] == baseline)].iloc[0]
                comparisons.append(dict(configuration=config, reference=baseline, source_test_n=int(raw['n']), source_raw_accuracy=float(raw.accuracy), source_reference_accuracy=float(old.accuracy), source_accuracy_gain_pp=100 * (raw.accuracy - old.accuracy), amazon_accuracy_gain_pp=100 * (am.loc['raw', 'accuracy'] - am.loc[baseline, 'accuracy']), source_raw_auroc=float(raw.auroc), source_reference_auroc=float(old.auroc), source_auroc_gain_pp=100 * (raw.auroc - old.auroc), amazon_auroc_gain_pp=100 * (am.loc['raw', 'auroc'] - am.loc[baseline, 'auroc'])))
    pd.DataFrame(comparisons).to_csv(O / 'historical_source_vs_amazon_gains.csv', index=False)
    lines = ['', '## Historical source-domain gains versus Amazon transfer', '', "These are separate saved test evaluations, not a common sample, and source tests were examined previously. SST-2 uses the project's internal held-out test rather than hidden official SST-2 test labels. Source-trained prompts differ. No new inference or fitted models.", '', '| Configuration | Reference | Source test n | Source raw accuracy gain (pp) | Amazon raw accuracy gain (pp) |', '|---|---|---:|---:|---:|']
    for r in comparisons:
        lines.append(f'| {r['configuration']} | {r['reference']} | {r['source_test_n']} | {r['source_accuracy_gain_pp']:+.3f} | {r['amazon_accuracy_gain_pp']:+.3f} |')
    lines += ['', '## Denoising and control results', '', '| Configuration | Denoised minus raw accuracy (pp) | Paired 95% CI (pp) | Random mean accuracy (%) | Between-seed SD (pp) |', '|---|---:|---|---:|---:|']
    for config in amazon.configuration.unique():
        r = secondary[(secondary.configuration == config) & (secondary.method == 'denoised') & (secondary.reference == 'raw') & (secondary.metric == 'accuracy')].iloc[0]
        random = amazon[(amazon.configuration == config) & (amazon.method == 'random_rank7_mean')].iloc[0]
        lines.append(f'| {config} | {r.difference_pp:+.3f} | [{r.ci_lower_pp:+.3f}, {r.ci_upper_pp:+.3f}] | {random.accuracy * 100:.3f} | {random.accuracy_seed_sd * 100:.3f} |')
        seeds = correction[(correction.configuration == config) & correction.method.str.startswith('random_rank7_seed_')]
        assert len(seeds) == 50
        row = dict(configuration=config, method='random_rank7_mean', **{key: float(seeds[key].mean()) for key in ('errors_fixed', 'correct_made_incorrect', 'net_corrections', 'predictions_changed', 'fraction_changed')})
        correction = pd.concat([correction, pd.DataFrame([row])], ignore_index=True)
    correction.to_csv(O / 'correction_analysis.csv', index=False)
    lines += ['', 'Random-mean correction counts are averages over the same 50 fixed classifiers and need not be integers; they are not an ensemble confusion matrix. All individual counts remain saved.', '']
    (O / 'stage4_results.md').open('a', encoding='utf-8').write('\n'.join(lines))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print('\nRAW ACCURACY COMPARISONS')
    print(primary[primary.metric == 'accuracy'].to_string(index=False))
    print('\nAUROC')
    print(amazon[amazon.method.isin(['original', 'sentiment_only', 'raw', 'denoised', 'pca_rank7', 'random_rank7_mean'])][['configuration', 'method', 'auroc']].to_string(index=False))
    print('\nRAW CORRECTIONS')
    print(correction[correction.method == 'raw'].to_string(index=False))
if __name__ == '__main__':
    main()
