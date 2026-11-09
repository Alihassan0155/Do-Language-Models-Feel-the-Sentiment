"""Build local transfer configuration from newly generated source heads.

No historical identity, fitted coefficient, dataset hash or target outcome
is embedded here. This preparation reads locally generated source outputs.
"""
import csv
import hashlib
import json
from pathlib import Path
from release_paths import WORKSPACE

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    root = WORKSPACE
    target = root / 'artifacts/stage_k_amazon_transfer'
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'transfer_compatibility.json').exists():
        raise FileExistsError('Transfer configuration is already frozen')
    settings, inventory, tracked = ([], [], {})
    for source, folder in [('sst2', 'stage_i_sst2'), ('tweeteval', 'stage_i_tweeteval_matched_controls')]:
        base = root / 'artifacts' / folder
        protocol = json.loads((base / 'protocol.json').read_text())
        for key in ('0.5b', '3b'):
            identity = protocol['models'][key]
            if source == 'sst2':
                prompt = protocol['prompt']
            else:
                prompt = protocol['margin_scoring']['models'][key]['prompt']
            cache = root / '.cache/huggingface/hub' / ('models--' + identity['model_id'].replace('/', '--')) / 'snapshots' / identity['checkpoint_commit']
            if not cache.is_dir():
                raise FileNotFoundError('Acquire the exact model snapshot in the workspace cache first')
            vectors = {name: str(root / 'artifacts' / folder_name / key / filename) for name, folder_name, filename in [('raw', 'stage_b_raw_vectors', 'emotion_directions_raw.float32.npy'), ('denoised', 'stage_b_final_vectors', 'emotion_directions_final.float32.npy')]}
            settings.append(dict(source=source, model=key, identity=identity, prompt=prompt, vectors=vectors, snapshot=str(cache)))
            methods = ['sentiment_only', 'raw', 'denoised', 'pca_rank7'] + [f'random_rank7_seed_{s}' for s in range(2000, 2050)]
            for method in methods:
                classifier = base / key / 'classifiers' / method / 'model.joblib'
                if source == 'tweeteval' and method == 'sentiment_only':
                    classifier = root / 'artifacts/stage_i_sentiment_correction' / key / 'correction_model.joblib'
                elif source == 'tweeteval' and method in ('raw', 'denoised'):
                    classifier = root / 'artifacts/stage_i_emotion_guided_correction' / key / 'classifiers' / method / 'model.joblib'
                tracked[str(classifier)] = sha(classifier)
                import joblib
                pipe = joblib.load(classifier)
                c = pipe.named_steps['classifier'].C
                inventory.append(dict(source=source, model=key, method=method, classifier_path=str(classifier), sha256=sha(classifier), C=c, feature_count=pipe.n_features_in_))
            for path in (base / key).rglob('*'):
                if path.is_file():
                    tracked[str(path)] = sha(path)
    with (target / 'source_model_inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    (target / 'transfer_compatibility.json').write_text(json.dumps({'source_settings': settings, 'verified_source_sha256': tracked}, indent=2))
if __name__ == '__main__':
    main()
