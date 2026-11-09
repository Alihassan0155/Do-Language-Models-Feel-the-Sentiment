"""Focused Stage H follow-up: actual-energy-matched partial emotion removal."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent) if (Path(__file__).parent / 'stage_h.py').exists() else str(CODE_ROOT))
import argparse, csv, json, sqlite3, hashlib
from types import SimpleNamespace
import numpy as np
import stage_h as H
from stage_e_common import ROOT, ART, E, MODELS, atomic_json, sha
from stage_e_pipeline import assert_protocol
from stage_g_confirmation import bootstrap_weights
SOURCE = ART / 'stage_h'
KEYS = ('0.5b', '3b')
DATASET = 'tweet_eval_sentiment'
ORIGINAL_REMOVE = H.remove
ACTIVE = {}

def write_csv(path, rows):
    H.write_csv(path, rows)

def prepare(args):
    source = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    if source['centering'] != 'none':
        raise ValueError('This follow-up requires original uncentered Stage H.')
    if args.batch_size != source['batch_size']:
        raise ValueError('Use the original frozen batch size for exact control reuse.')
    state = json.loads((SOURCE / 'state.json').read_text(encoding='utf-8'))
    if not all((state['model_datasets'][f'{key}/{DATASET}'] == 'complete' for key in KEYS)):
        raise ValueError('Source results incomplete.')
    p = json.loads((E / 'protocol.json').read_text(encoding='utf-8'))
    assert_protocol(p)
    if source['stage_e_protocol_sha256'] != sha(E / 'protocol.json'):
        raise ValueError('Original Stage H and current Stage E provenance differ.')
    if any((source['models'][key] != p['models'][key] or source['precision'][key] != H.PRECISION[key] for key in KEYS)):
        raise ValueError('Original model configuration/precision differs.')
    if len(source['random_seeds']) < 50:
        raise ValueError('Expected all 50+ frozen random controls.')
    samples = json.loads((SOURCE / 'samples.json').read_text(encoding='utf-8'))
    frozen = samples[DATASET]
    if len(frozen) != 600 or len({r['id'] for r in frozen}) != 600 or sum((r['label'] == 0 for r in frozen)) != 300 or (sum((r['label'] == 1 for r in frozen)) != 300):
        raise ValueError('Expected the same frozen, balanced 600 unique binary examples.')
    source_files = [SOURCE / 'scores.sqlite3', SOURCE / 'protocol.json', SOURCE / 'samples.json', ROOT / 'stage_h.py']
    for key in KEYS:
        source_files += [SOURCE / key / DATASET / 'bases.npz', SOURCE / key / DATASET / 'bases_metadata.json']
    signatures = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    config = dict(schema=1, source_sha256=signatures, stage_e_protocol_sha256=sha(E / 'protocol.json'), models=list(KEYS), dataset=DATASET, n=600, batch_size=args.batch_size, layers={key: p['models'][key]['layer'] for key in KEYS}, random_seeds=source['random_seeds'], precision={key: source['precision'][key] for key in KEYS}, bootstrap_draws=args.bootstrap, bootstrap_seed=45, matching='per-example mean text-token ||delta h||^2/||h||^2; target actual float16 full random rank-7 projection removal', coefficient='one beta per example, capped at 1; 16-step bisection on actual rounded emotion perturbation; no labels/outcomes used', match_relative_tolerance=args.match_tolerance, match_absolute_tolerance=1e-07, statistics='paired class-stratified examples and paired random seeds resampled jointly; fixed layers and frozen T1', caution='Matches perturbation energy, not total post-ablation norm loss or every token/directional distortion. Original results preserved.')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('protocol.json', config), ('samples.json', {DATASET: samples[DATASET]})]:
        path = args.output / name
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Follow-up settings/source changed; choose a separate --output folder.')
        if not path.exists():
            atomic_json(path, value)
    return (p, config, {DATASET: samples[DATASET]})

def bases(key, config, smoke=False):
    folder = SOURCE / key / DATASET
    meta = json.loads((folder / 'bases_metadata.json').read_text(encoding='utf-8'))
    if meta['bases_sha256'] != sha(folder / 'bases.npz'):
        raise ValueError('Source bases changed.')
    with np.load(folder / 'bases.npz') as saved:
        raw = saved['emotion_raw'].copy()
        clean = saved['emotion_denoised'].copy()
        anchor = saved['anchor'].copy()
        random = {seed: saved[f'random_rank7_seed_{seed}'].copy() for seed in config['random_seeds']}
    if np.any(anchor != 0) or raw.shape[0] != 7 or clean.shape[0] != 7:
        raise ValueError('Wrong anchor/rank.')
    values = {'emotion_raw': raw, 'emotion_denoised': clean, 'baseline_parity': np.empty((0, raw.shape[1]), dtype=np.float32)}
    for seed in config['random_seeds'][:1] if smoke else config['random_seeds']:
        r = random[seed]
        values[f'random_rank7_seed_{seed}'] = r
        for kind, e in [('raw', raw), ('denoised', clean)]:
            marker = np.zeros((1, e.shape[1]), dtype=np.float32)
            marker[0, 0] = seed
            marker[0, 1] = 0 if kind == 'raw' else 1
            values[f'matched_{kind}_seed_{seed}'] = np.concatenate([e, r, marker], axis=0)
    return (values, anchor, dict(meta, adapter='energy matched partial rank-7 ablation; payload rows encode two bases, not combined rank', methods=list(values)))

def matching_change(h, mask, e, r, iterations=16):
    import torch
    a = h.float()
    den = a.square().sum(dim=-1).clamp_min(1e-20)
    counts = mask.sum(dim=1).clamp_min(1)

    def energy(changed):
        return ((changed.float() - a).square().sum(dim=-1) / den * mask).sum(dim=1) / counts
    random_changed = ORIGINAL_REMOVE(h, mask, r, torch.zeros(h.shape[-1], device=h.device))
    target = energy(random_changed)
    projection = a @ e.T @ e

    def at(beta):
        return (a - projection * mask.unsqueeze(-1) * beta[:, None, None]).to(h.dtype)
    zero = torch.zeros(len(h), device=h.device)
    one = torch.ones_like(zero)
    capacity = energy(at(one))
    feasible = target <= capacity + 1e-07
    lo = zero
    hi = one
    for _ in range(iterations):
        mid = (lo + hi) / 2
        measured = energy(at(mid))
        below = measured < target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    low_energy = energy(at(lo))
    high_energy = energy(at(hi))
    beta = torch.where((low_energy - target).abs() <= (high_energy - target).abs(), lo, hi)
    beta = torch.where(target == 0, zero, beta)
    beta = torch.where(feasible, beta, one)
    changed = at(beta)
    actual = energy(changed)
    return (changed, target, actual, beta, feasible)

def matched_remove(output, mask, rows, anchor):
    if rows.shape[0] != 15:
        return ORIGINAL_REMOVE(output, mask, rows, anchor)
    h = output[0] if isinstance(output, tuple) else output
    changed, target, actual, beta, feasible = matching_change(h, mask, rows[:7], rows[7:14])
    seed = int(rows[14, 0].item())
    kind = 'raw' if int(rows[14, 1].item()) == 0 else 'denoised'
    method = f'matched_{kind}_seed_{seed}'
    done = {r[0] for r in ACTIVE['progress'].execute('SELECT example_index FROM scores WHERE model=? AND dataset=? AND method=?', (ACTIVE['key'], DATASET, method))}
    pending = [i for i in range(ACTIVE['n']) if i not in done]
    start = pending[0] // ACTIVE['batch_size'] * ACTIVE['batch_size']
    arrays = [x.detach().cpu().numpy() for x in (target, actual, beta, feasible)]
    records = [(ACTIVE['key'], method, start + j, float(arrays[0][j]), float(arrays[1][j]), float(arrays[2][j]), int(arrays[3][j])) for j in range(len(h))]
    with ACTIVE['audit']:
        ACTIVE['audit'].executemany('INSERT OR REPLACE INTO matching VALUES (?,?,?,?,?,?,?)', records)
    return (changed, *output[1:]) if isinstance(output, tuple) else changed

def copy_reference(folder, config, n):
    dest = H.database(folder)
    source = sqlite3.connect(f'file:{(SOURCE / 'scores.sqlite3').as_posix()}?mode=ro', uri=True)
    for key in KEYS:
        names = ['baseline', 'emotion_raw', 'emotion_denoised'] + [f'random_rank7_seed_{seed}' for seed in config['random_seeds']]
        for method in names:
            records = source.execute('SELECT * FROM scores WHERE model=? AND dataset=? AND method=? AND example_index<? ORDER BY example_index', (key, DATASET, method, n)).fetchall()
            if len(records) != n:
                raise ValueError('Incomplete source reference.')
            with dest:
                dest.executemany('INSERT OR IGNORE INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', records)
    source.close()
    dest.close()

def execute(p, config, samples, args, smoke=False):
    folder = args.output / 'smoke' if smoke else args.output
    folder.mkdir(parents=True, exist_ok=True)
    current = {DATASET: samples[DATASET][:8]} if smoke else samples
    n = len(current[DATASET])
    copy_reference(folder, config, n)
    audit = sqlite3.connect(folder / 'matching.sqlite3')
    audit.execute('CREATE TABLE IF NOT EXISTS matching(model TEXT,method TEXT,example_index INTEGER,target_energy REAL,actual_energy REAL,beta REAL,feasible INTEGER,PRIMARY KEY(model,method,example_index))')
    progress = sqlite3.connect(folder / 'scores.sqlite3')
    original = (H.remove, H.build_bases, H.DATASETS)
    H.remove = matched_remove
    H.DATASETS = (DATASET,)

    def adapted_bases(p, key, dataset, unused, folder):
        return bases(key, config, smoke)
    H.build_bases = adapted_bases
    try:
        for key in KEYS:
            ACTIVE.update(key=key, n=n, batch_size=args.batch_size, audit=audit, progress=progress)
            H.execute(p, dict(config, random_seeds=config['random_seeds']), current, SimpleNamespace(output=folder, batch_size=args.batch_size), [key], False)
            baseline = progress.execute('SELECT example_index,predicted,margin FROM scores WHERE model=? AND method="baseline" ORDER BY example_index', (key,)).fetchall()
            parity = progress.execute('SELECT example_index,predicted,margin FROM scores WHERE model=? AND method="baseline_parity" ORDER BY example_index', (key,)).fetchall()
            if baseline != parity:
                raise ValueError('New forward baseline differs from source; copied controls cannot be treated as matched.')
    finally:
        H.remove, H.build_bases, H.DATASETS = original
        audit.close()
        progress.close()
        ACTIVE.clear()
    print('Frozen-batch baseline parity passed; original full random controls reused read-only.', flush=True)

def report(config, samples, args, smoke=False):
    folder = args.output / 'smoke' if smoke else args.output
    n = 8 if smoke else 600
    rows = samples[DATASET][:n]
    y = np.array([r['label'] for r in rows])
    w = bootstrap_weights(y, args.bootstrap, 45)
    point = np.full((1, n), 1 / n)
    seed_count = 1 if smoke else len(config['random_seeds'])
    seeds = config['random_seeds'][:seed_count]
    rng = np.random.default_rng(46)
    seed_weights = rng.multinomial(seed_count, np.full(seed_count, 1 / seed_count), size=args.bootstrap) / seed_count
    db = sqlite3.connect(folder / 'scores.sqlite3')
    db.row_factory = sqlite3.Row
    audit = sqlite3.connect(folder / 'matching.sqlite3')
    audit.row_factory = sqlite3.Row

    def get(method, key):
        records = [dict(r) for r in db.execute('SELECT * FROM scores WHERE model=? AND method=? ORDER BY example_index', (key, method))]
        if [r['example_index'] for r in records] != list(range(n)):
            raise ValueError(f'Incomplete follow-up: {key}/{method}; resume run before reporting.')
        return records

    def compute(records):
        pred = np.array([r['predicted'] for r in records])
        m = np.array([r['margin'] for r in records])
        return (H.metrics(y, pred, m, point)[0], H.metrics(y, pred, m, w))
    summaries = []
    pairs = []
    distortion = []
    aggregates = {}
    checks = []
    metric_names = ('accuracy', 'macro_f1', 'auroc')
    for key in KEYS:
        baseline = get('baseline', key)
        bp, bb = compute(baseline)
        baseline_parity = get('baseline_parity', key)
        if any(((a['predicted'], a['margin']) != (b['predicted'], b['margin']) for a, b in zip(baseline, baseline_parity))):
            raise ValueError('Baseline parity failed.')
        random_points = []
        random_boot = []
        random_energy = []
        groups = {'raw': [], 'denoised': []}
        group_boot = {'raw': [], 'denoised': []}
        group_energy = {'raw': [], 'denoised': []}
        valid = True
        for seed in seeds:
            rrecords = get(f'random_rank7_seed_{seed}', key)
            rp, rb = compute(rrecords)
            random_points.append(rp)
            random_boot.append(rb)
            re = np.array([r['token_removed_energy_fraction'] for r in rrecords])
            random_energy.append(re.mean())
            distortion.append(dict(model=key, condition='random_full', seed=seed, n=n, energy_percent=100 * re.mean(), accuracy_damage_pp=100 * (bp[0] - rp[0]), macro_f1_damage_pp=100 * (bp[1] - rp[1]), auroc_damage_pp=100 * (bp[2] - rp[2])))
            for kind in ('raw', 'denoised'):
                name = f'matched_{kind}_seed_{seed}'
                records = get(name, key)
                mp, mb = compute(records)
                groups[kind].append(mp)
                group_boot[kind].append(mb)
                aa = [dict(r) for r in audit.execute('SELECT * FROM matching WHERE model=? AND method=? ORDER BY example_index', (key, name))]
                if [r['example_index'] for r in aa] != list(range(n)):
                    raise ValueError('Matching audit incomplete.')
                actual = np.array([r['actual_energy'] for r in aa])
                target = np.array([r['target_energy'] for r in aa])
                feasible = np.array([r['feasible'] for r in aa], dtype=bool)
                beta = np.array([r['beta'] for r in aa])
                saved = np.array([r['token_removed_energy_fraction'] for r in records])
                if not np.allclose(actual, saved, atol=1e-07, rtol=1e-05) or not np.allclose(target, re, atol=1e-07, rtol=1e-05):
                    raise ValueError('Matching target or applied energy differs from scored/source controls.')
                gap = np.abs(actual - target)
                ok = feasible & (gap <= args.match_tolerance * target + 1e-07)
                valid = valid and bool(ok.all())
                group_energy[kind].append(actual.mean())
                checks.append(dict(model=key, kind=kind, seed=seed, n=n, matching_ok=int(ok.sum()), infeasible=int((~feasible).sum()), mean_beta=float(beta.mean()), min_beta=float(beta.min()), max_beta=float(beta.max()), target_energy_percent=100 * target.mean(), actual_energy_percent=100 * actual.mean(), max_abs_energy_gap_pp=100 * gap.max(), max_relative_gap=float(np.max(gap / np.maximum(target, 1e-12)))))
                distortion.append(dict(model=key, condition=f'emotion_{kind}_matched', seed=seed, n=n, energy_percent=100 * actual.mean(), accuracy_damage_pp=100 * (bp[0] - mp[0]), macro_f1_damage_pp=100 * (bp[1] - mp[1]), auroc_damage_pp=100 * (bp[2] - mp[2])))
                for j, metric in enumerate(metric_names):
                    excess = rb[:, j] - mb[:, j]
                    ci = H.interval(excess)
                    pairs.append(dict(model=key, kind=kind, seed=seed, n=n, metric=metric, random=float(rp[j]), matched_emotion=float(mp[j]), excess_damage=float(rp[j] - mp[j]), excess_ci_low=ci[0], excess_ci_high=ci[1], energy_matching_ok=bool(ok.all())))
        random_points = np.asarray(random_points)
        random_boot = np.stack(random_boot, axis=1)
        random_mean_boot = np.einsum('bs,bsm->bm', seed_weights, random_boot)
        conditions = {'baseline': (bp, bb, 0.0), 'random_full': (random_points.mean(axis=0), random_mean_boot, float(np.mean(random_energy)))}
        for kind in ('raw', 'denoised'):
            boot = np.stack(group_boot[kind], axis=1)
            mean_boot = np.einsum('bs,bsm->bm', seed_weights, boot)
            conditions[f'emotion_{kind}_matched'] = (np.mean(groups[kind], axis=0), mean_boot, float(np.mean(group_energy[kind])))
        for kind in ('raw', 'denoised'):
            full = get(f'emotion_{kind}', key)
            fp, fb = compute(full)
            fe = float(np.mean([r['token_removed_energy_fraction'] for r in full]))
            conditions[f'emotion_{kind}_full_reference'] = (fp, fb, fe)
            distortion.append(dict(model=key, condition=f'emotion_{kind}_full_reference', seed=-1, n=n, energy_percent=100 * fe, accuracy_damage_pp=100 * (bp[0] - fp[0]), macro_f1_damage_pp=100 * (bp[1] - fp[1]), auroc_damage_pp=100 * (bp[2] - fp[2])))
        for name, (mp, mb, energy) in conditions.items():
            for j, metric in enumerate(metric_names):
                damage = bp[j] - mp[j]
                ci = H.interval(bb[:, j] - mb[:, j])
                pci = H.interval(mb[:, j])
                excess_ci = H.interval(random_mean_boot[:, j] - mb[:, j]) if name.endswith('_matched') else ['', '']
                summaries.append(dict(model=key, dataset=DATASET, condition=name, n=n, seed_pairs=seed_count, metric=metric, baseline=float(bp[j]), performance=float(mp[j]), performance_ci_low=pci[0], performance_ci_high=pci[1], damage=float(damage), damage_ci_low=ci[0], damage_ci_high=ci[1], energy_percent=100 * energy, excess_vs_random=float(random_points[:, j].mean() - mp[j]) if name.endswith('_matched') else '', excess_ci_low=excess_ci[0], excess_ci_high=excess_ci[1], all_energy_matching_valid=valid))
        aggregates[key] = dict(complete=True, energy_matching_valid=valid, n=n, seed_pairs=seed_count)
    db.close()
    audit.close()
    write_csv(folder / 'performance_paired_ci.csv', summaries)
    write_csv(folder / 'seed_pair_comparisons.csv', pairs)
    write_csv(folder / 'damage_vs_distortion.csv', distortion)
    write_csv(folder / 'matching_checks.csv', checks)
    atomic_json(folder / 'state.json', dict(complete=not smoke, smoke=smoke, models=aggregates, all_energy_matching_valid=all((x['energy_matching_valid'] for x in aggregates.values()))))
    lines = [f'Distortion-matched Stage H follow-up; smoke={smoke}; original Stage H unchanged.', f'{n} frozen TweetEval examples; {seed_count} paired random rank-7 targets; fixed layers 14 and 18.', 'Energy is mean token ||delta h||^2/||h||^2 within example. Partial emotion removal uses beta in [0,1], chosen without labels.', 'Matched aggregates average per-seed model metrics, not ensemble predictions. Paired bootstrap jointly resamples balanced examples and seed pairs.', 'Perturbation energy matching does not imply matching net residual-energy loss, pooled cosine, or perturbation direction.', 'Infeasible/out-of-tolerance cases are retained and flagged; do not claim a matched necessity comparison unless matching validity passes.', 'Pointwise 95% CIs; no multiplicity correction; previously observed sample, no untouched-test claim.']
    for r in summaries:
        lines.append(f'{r['model']} {r['condition']} {r['metric']}: performance={r['performance']:.4f} CI=[{r['performance_ci_low']:.4f},{r['performance_ci_high']:.4f}], damage={r['damage']:+.4f} CI=[{r['damage_ci_low']:+.4f},{r['damage_ci_high']:+.4f}], energy={r['energy_percent']:.5f}%; matching_valid={r['all_energy_matching_valid']}' + (f', excess vs random={r['excess_vs_random']:+.4f} CI=[{r['excess_ci_low']:+.4f},{r['excess_ci_high']:+.4f}]' if r['condition'].endswith('_matched') else ''))
    (folder / 'results_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plot(folder, distortion)
    print(f'Saved follow-up report: {folder}; matching valid={all((x['energy_matching_valid'] for x in aggregates.values()))}', flush=True)

def plot(folder, rows):
    os.environ['MPLCONFIGDIR'] = str(folder / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for include_full, name in [(False, 'matched_damage_vs_distortion'), (True, 'full_and_matched_damage_vs_distortion')]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
        for i, key in enumerate(KEYS):
            for j, (metric, title) in enumerate([('accuracy_damage_pp', 'Accuracy'), ('macro_f1_damage_pp', 'Macro-F1'), ('auroc_damage_pp', 'AUROC')]):
                ax = axes[i, j]
                for condition, color in [('random_full', '#777777'), ('emotion_raw_matched', '#D17A22'), ('emotion_denoised_matched', '#148C8C')]:
                    selected = [r for r in rows if r['model'] == key and r['condition'] == condition]
                    ax.scatter([r['energy_percent'] for r in selected], [r[metric] for r in selected], s=18, alpha=0.65, color=color, label=condition)
                if include_full:
                    for condition, color in [('emotion_raw_full_reference', '#D17A22'), ('emotion_denoised_full_reference', '#148C8C')]:
                        selected = [r for r in rows if r['model'] == key and r['condition'] == condition]
                        ax.scatter([r['energy_percent'] for r in selected], [r[metric] for r in selected], s=110, marker='*', color=color, label=condition)
                ax.axhline(0, color='black', lw=0.5)
                ax.set_title(f'Qwen {key}: {title}')
                ax.set_xlabel('Measured perturbation energy (%)')
                ax.set_ylabel('Performance damage (pp)')
        axes[0, 0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(folder / f'{name}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)

def self_test():
    import torch
    h = torch.tensor([[[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]]])
    mask = torch.tensor([[True, False]])
    e = torch.tensor([[1.0, 0.0, 0.0]])
    r = torch.tensor([[0.0, 1.0, 0.0]])
    changed, target, actual, beta, feasible = matching_change(h, mask, e, r)
    assert not bool(feasible[0]) and float(beta[0]) == 1
    r = torch.tensor([[0.0, 0.0, 1.0]])
    changed, target, actual, beta, feasible = matching_change(h, mask, r, e)
    assert bool(feasible[0]) and 0 < float(beta[0]) < 1 and torch.allclose(target, actual, rtol=0.0001, atol=1e-07)
    assert torch.equal(changed[~mask], h[~mask])
    loss = (h.square().sum() - changed.square().sum()) / h.square().sum()
    assert not np.isclose(float(loss), float(actual[0]))
    print('Matching, infeasibility cap, masking, and perturbation-vs-net-energy checks passed.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'run', 'report', 'all'], default='prepare')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--match-tolerance', type=float, default=0.01)
    parser.add_argument('--output', type=Path, default=ART / 'stage_h_distortion_matched')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.bootstrap < 100 or not 0 < args.match_tolerance < 0.1:
        parser.error('Use 100+ bootstrap draws and relative tolerance between 0 and 0.1.')
    args.output = args.output.resolve()
    if args.output == SOURCE or SOURCE in args.output.parents:
        parser.error('Use a new output folder outside original Stage H.')
    p, config, samples = prepare(args)
    if args.phase == 'smoke':
        execute(p, config, samples, args, True)
        report(config, samples, args, True)
    if args.phase in ('run', 'all'):
        execute(p, config, samples, args)
    if args.phase in ('report', 'all'):
        report(config, samples, args)
    for path, signature in config['source_sha256'].items():
        if sha(ROOT / path) != signature:
            raise ValueError('Original source changed during follow-up.')
    print('Original Stage H source hashes unchanged.', flush=True)
if __name__ == '__main__':
    main()
