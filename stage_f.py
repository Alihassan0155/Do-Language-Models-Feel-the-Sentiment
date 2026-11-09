"""Stage F: training-only emotion/sentiment geometry, no model forwards."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import argparse
import csv
import json
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MPLCONFIGDIR'] = str(WORKSPACE / 'visualizations' / '.mplconfig')
import numpy as np
from stage_e_common import ROOT, ART, E, MODELS, DATASETS, atomic_json, sha
from stage_e_pipeline import assert_protocol, load_hidden, emotion_inputs, random_directions, fit_pca

def basis(vectors, rtol=1e-05):
    _, s, vt = np.linalg.svd(np.asarray(vectors, dtype=np.float64), full_matrices=False)
    rank = int(np.sum(s > s[0] * rtol)) if s[0] > 0 else 0
    return (vt[:rank].T, s)

def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        raise ValueError('Zero-length direction; cosine is undefined.')
    return float(np.clip(np.dot(a, b) / denominator, -1, 1))

def geometry(sentiment, vectors):
    u, singular = basis(vectors)
    inside = u @ (u.T @ sentiment)
    outside = sentiment - inside
    valence = vectors[:4].mean(axis=0) - vectors[4:].mean(axis=0)
    norm = np.linalg.norm(sentiment)
    if norm <= 0:
        raise ValueError('Zero sentiment direction.')
    alignment = cosine(sentiment, valence)
    coverage = float(np.dot(inside, inside) / np.dot(sentiment, sentiment))
    if not np.allclose(inside @ outside, 0, atol=1e-08 * norm ** 2):
        raise ValueError('Projection is not orthogonal.')
    return (dict(rank=u.shape[1], cosine_valence=alignment, angle_degrees=float(np.degrees(np.arccos(alignment))), coverage=coverage, explained_percent=100 * coverage, norm_sentiment=float(norm), norm_inside=float(np.linalg.norm(inside)), norm_outside=float(np.linalg.norm(outside))), u, inside, outside, valence, singular)

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plots(rows, comparisons, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight'})
    labels = ['Qwen 0.5B', 'Qwen 1.5B', 'Qwen 3B', 'BERT-large']

    def save(fig, name):
        fig.savefig(output / (name + '.png'), dpi=300)
        fig.savefig(output / (name + '.pdf'))
        plt.close(fig)
    for field, ylabel, filename in [('explained_percent', 'Sentiment direction covered (%)', 'emotion_coverage'), ('cosine_valence', 'Cosine: valence vs sentiment', 'valence_alignment')]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
        for ax, dataset in zip(axes, DATASETS):
            x = np.arange(4)
            for offset, space, color in [(-0.18, 'raw', '#7B61A8'), (0.18, 'denoised', '#148C8C')]:
                values = [next((r[field] for r in rows if r['model'] == key and r['dataset'] == dataset and (r['space'] == space))) for key in MODELS]
                ax.bar(x + offset, values, width=0.34, label=space.title(), color=color)
            ax.set_xticks(x, labels, rotation=20, ha='right')
            ax.set_title('SST-2' if dataset == 'sst2' else 'TweetEval (binary)')
            ax.axhline(0, color='black', lw=0.6)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(frameon=False)
        fig.tight_layout()
        save(fig, filename)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, (dataset, space) in zip(axes.flat, [(d, s) for d in DATASETS for s in ('raw', 'denoised')]):
        selected = [r for r in comparisons if r['dataset'] == dataset and r['space'] == space]
        for j, (field, label, color) in enumerate([('emotion_coverage', 'Emotion', '#148C8C'), ('random_mean', 'Random', '#888888'), ('pca_coverage', 'PCA', '#D99238')]):
            values = [next((r[field] for r in selected if r['model'] == key)) * 100 for key in MODELS]
            error = [next((r['random_sd'] for r in selected if r['model'] == key)) * 100 for key in MODELS] if field == 'random_mean' else None
            ax.bar(np.arange(4) + (j - 1) * 0.25, values, 0.24, yerr=error, capsize=3, label=label, color=color)
        ax.set_xticks(np.arange(4), labels, rotation=20, ha='right')
        ax.set_title(f'{dataset} / {space} (rank matched)')
        ax.set_ylabel('Coverage (%)')
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, 'rank_matched_controls')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        v = np.array([[1.0, 0, 0], [0, 1, 0]])
        u, _ = basis(v)
        s = np.array([3.0, 4.0, 12.0])
        inside = u @ (u.T @ s)
        assert np.allclose(inside, [3, 4, 0])
        assert np.isclose(np.dot(inside, inside) / np.dot(s, s), 25 / 169)
        assert np.isclose(cosine(s, s), 1)
        assert basis(np.vstack([v, v.sum(axis=0)]))[0].shape[1] == 2
        assert np.allclose(random_directions(20, 0, 7) @ random_directions(20, 0, 7).T, np.eye(7))
        print('Geometry self-tests passed.')
        return
    p = json.loads((E / 'protocol.json').read_text(encoding='utf-8'))
    assert_protocol(p)
    for key in MODELS:
        for dataset in DATASETS:
            h, y, _, _ = load_hidden(p, key, dataset, 'train')
            del h, y
    folder = ART / 'stage_f'
    figures = ROOT / 'visualizations' / 'stage_f'
    folder.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rows, controls, comparisons, stability, provenance = ([], [], [], [], {})
    for key in MODELS:
        _, info = emotion_inputs(key)
        vectors = {space: np.asarray(np.load(ART / subfolder / key / filename)[:, info['layer'] - 1], dtype=np.float64) for space, subfolder, filename in [('raw', 'stage_b_raw_vectors', 'emotion_directions_raw.float32.npy'), ('denoised', 'stage_b_final_vectors', 'emotion_directions_final.float32.npy')]}
        sentiments = {}
        for dataset in DATASETS:
            print(f'Stage F: {key}/{dataset}, layer {info['layer']}', flush=True)
            h, y, _, meta = load_hidden(p, key, dataset, 'train')
            means = [np.mean(h[y == label], axis=0, dtype=np.float64) for label in (0, 1)]
            sent = means[1] - means[0]
            sentiments[dataset] = sent
            pca = fit_pca(h, p['seed'])
            target = folder / key / dataset
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / 'sentiment_direction.npy', sent)
            np.save(target / 'class_means_negative_positive.npy', np.stack(means))
            provenance[f'{key}/{dataset}'] = dict(layer=info['layer'], train_hidden_sha256=meta['hidden_sha256'], train_source_sha256=meta['source_sha256'], n_negative=int(sum(y == 0)), n_positive=int(sum(y == 1)), raw_sha256=info['raw_sha256'], final_sha256=info['final_sha256'])
            for space, v in vectors.items():
                metrics, u, inside, outside, valence, singular = geometry(sent, v)
                r = metrics['rank']
                if r < 1 or r > 8:
                    raise ValueError('Unexpected emotion rank.')
                rows.append(dict(model=key, dataset=dataset, layer=info['layer'], space=space, **metrics))
                np.savez(target / (space + '_geometry.npz'), emotion_basis=u, sentiment_inside=inside, sentiment_outside=outside, valence_direction=valence, singular_values=singular)
                random_values = []
                for seed in p['random_seeds']:
                    q = random_directions(info['width'], seed, r).T
                    coverage = float(np.linalg.norm(q.T @ sent) ** 2 / np.dot(sent, sent))
                    random_values.append(coverage)
                    controls.append(dict(model=key, dataset=dataset, space=space, rank=r, control='random', seed=seed, coverage=coverage))
                pc = pca.components_[:r].T
                pc_coverage = float(np.linalg.norm(pc.T @ sent) ** 2 / np.dot(sent, sent))
                controls.append(dict(model=key, dataset=dataset, space=space, rank=r, control='pca_train', seed=p['seed'], coverage=pc_coverage))
                comparisons.append(dict(model=key, dataset=dataset, space=space, rank=r, emotion_coverage=metrics['coverage'], random_mean=float(np.mean(random_values)), random_sd=float(np.std(random_values, ddof=1)), random_expected=r / info['width'], pca_coverage=pc_coverage))
            del h, y, pca
        stability.append(dict(model=key, layer=info['layer'], cosine_sst2_tweet=cosine(sentiments['sst2'], sentiments['tweet_eval_sentiment']), angle_degrees=float(np.degrees(np.arccos(cosine(sentiments['sst2'], sentiments['tweet_eval_sentiment']))))))
    write_csv(folder / 'geometry.csv', rows)
    write_csv(folder / 'controls.csv', controls)
    write_csv(folder / 'coverage_comparison.csv', comparisons)
    write_csv(folder / 'cross_dataset_stability.csv', stability)
    plots(rows, comparisons, figures)
    lines = ['Stage F: training-only geometry at Stage-C-selected layers; fixed Stage E T1.', 'TweetEval neutral excluded. Valence uses original vector magnitudes.', 'SVD rank tolerance=1e-5 relative to largest singular value; random SD is across seeds, not a confidence interval.', 'Coverage is squared direction norm explained, not classification accuracy or causal evidence.', '']
    for r in rows:
        c = next((c for c in comparisons if all((c[k] == r[k] for k in ('model', 'dataset', 'space')))))
        lines.append(f'{r['model']} {r['dataset']} {r['space']}: rank={r['rank']}, coverage={r['explained_percent']:.2f}%, cosine={r['cosine_valence']:.4f}, angle={r['angle_degrees']:.2f} deg, inside={r['norm_inside']:.4f}, outside={r['norm_outside']:.4f}; random={100 * c['random_mean']:.2f}% +/- {100 * c['random_sd']:.2f}%, PCA={100 * c['pca_coverage']:.2f}%')
    lines += ['', 'Cross-dataset sentiment-direction cosine:']
    lines += [f'{r['model']}: {r['cosine_sst2_tweet']:.4f}' for r in stability]
    (folder / 'stage_f_results_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_json(folder / 'metadata.json', dict(complete=True, split='train only', protocol_sha256=sha(E / 'protocol.json'), rank_relative_tolerance=1e-05, random_seeds=p['random_seeds'], valence='mean original positive vectors minus mean original negative vectors', inputs=provenance, geometry_sha256=sha(folder / 'geometry.csv')))
    print(f'Results: {folder}\nFigures: {figures}', flush=True)
if __name__ == '__main__':
    main()
