"""ASD-STE117: frozen genuine binary prompt; new forwards, resumable scores."""
from release_paths import WORKSPACE, CODE_ROOT, source_path, model_revision
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys, json, csv, sqlite3, hashlib, copy, importlib.util, time, argparse
from pathlib import Path
from types import SimpleNamespace
OUT = WORKSPACE / 'artifacts/asd_ste117_binary_prompt'
ROOT = WORKSPACE
sys.path.insert(0, str(CODE_ROOT))
import numpy as np
from prompt_templates import PROMPT_TEMPLATES
from stage_e_common import MODELS
ART = ROOT / 'artifacts'
DS = 'tweet_eval_sentiment'
KEYS = ('0.5b', '3b')

def js(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    p = source_path(p)
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, indent=2) + '\n', encoding='utf-8')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, source_path(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def prepare():
    old = js(ART / 'stage_h/protocol.json')
    matched = js(ART / 'stage_h_distortion_matched/protocol.json')
    prompt = PROMPT_TEMPLATES['sst2']['template_1']
    assert prompt == 'Classify the sentiment of this text as positive or negative.\nReply with only one label: positive or negative.\n\nText: {text}' and 'neutral' not in prompt
    allrows = [json.loads(x) for x in (ART / 'stage_e/data/tweet_eval_sentiment/test.jsonl').read_text(encoding='utf-8').splitlines()]
    sample = js(ART / 'stage_h/samples.json')[DS]
    assert sample == js(ART / 'stage_h_distortion_matched/samples.json')[DS]
    assert len(allrows) == 6347 and len({r['id'] for r in allrows}) == 6347
    assert len(sample) == 600 and len({r['id'] for r in sample}) == 600 and (np.bincount([r['label'] for r in sample]).tolist() == [300, 300])
    official = {r['id']: r for r in allrows}
    assert all((all((r[k] == official[r['id']][k] for k in ('text', 'label', 'source_split', 'source_row_index'))) for r in sample))
    assert old['random_seeds'] == matched['random_seeds'] == list(range(2000, 2050))
    paths = [ROOT / x for x in ('prompt_templates.py', 'stage_d.py', 'stage_h.py', 'stage_h_distortion_matched.py', 'stage_g_confirmation.py', 'stage_e_pipeline.py', 'stage_e_common.py')]
    paths += [ART / x for x in ('stage_e/protocol.json', 'stage_e/data/tweet_eval_sentiment/test.jsonl', 'stage_h/protocol.json', 'stage_h/samples.json', 'stage_h/scores.sqlite3', 'stage_h_distortion_matched/protocol.json', 'stage_h_distortion_matched/samples.json', 'stage_h_distortion_matched/scores.sqlite3', 'stage_h_distortion_matched/matching.sqlite3')]
    for dirname in ('binary_reanalysis_20261019', 'stage_i_sentiment_correction', 'stage_i_emotion_guided_correction', 'stage_i_tweeteval_matched_controls', 'stage_k_amazon_transfer'):
        paths += [p for p in (ART / dirname).rglob('*') if p.is_file() and (p.suffix in ('.joblib', '.csv', '.json', '.md') or p.name == 'scores.sqlite3')]
    for key in KEYS:
        paths += [ART / f'stage_h/{key}/{DS}/{x}' for x in ('bases.npz', 'bases_metadata.json')]
        paths += [ART / f'stage_b_raw_vectors/{key}/emotion_directions_raw.float32.npy', ART / f'stage_b_final_vectors/{key}/emotion_directions_final.float32.npy']
    digest = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    protocol = dict(task='ASD-STE117', prompt=prompt, prompt_origin='prompt_templates.py SST-2 template_1, exact string', chat='stage_d.token_batch, one user message, add_generation_prompt=True, add_special_tokens=False, left padding; text-offset mask excludes special/prompt/pad tokens', prediction='only negative/positive candidate tokens; full-vocabulary float32 log_softmax; M=float32(logPpositive)-float32(logPnegative); positive iff M>0, ties negative', baseline_n=6347, ablation_n=600, ablation_counts={'negative': 300, 'positive': 300}, models={k: old['models'][k] for k in KEYS}, new_prompt_for_both_models=prompt, layers={k: old['models'][k]['layer'] for k in KEYS}, batch_size=8, precision='FP16 forward; float32 logits and intervention arithmetic; deterministic CUDA, TF32 disabled', random_seeds=old['random_seeds'], centering='none', full_equation="h'=h-P_U h on original valid text tokens", matched_equation="h'=h-beta_i P_E h; beta_i recomputed on NEW binary-prompt activations against full random removal", matching=dict(iterations=16, beta_cap=1, relative_tolerance=matched['match_relative_tolerance'], absolute_tolerance=matched['match_absolute_tolerance'], energy='per-example mean valid-text-token ||delta h||^2/||h||^2; actual FP16-rounded intervention'), bootstrap=dict(draws=2000, full_example_seed=44, full_seeds_fixed=True, matched_example_seed=45, matched_paired_seed_seed=46, paired=True, class_stratified=True, nominal_no_multiplicity_correction=True), frozen='checkpoints, vectors/PCA denoising, historical subspace bases, layers, examples/order/batches, seeds, intervention formulas, matching tolerance/iterations; no outcome-based tuning', changed='binary prompt and strict two-label readout; all scores and matched betas/energies new', scope='retrospective prompt sensitivity; not independent validation; downstream stages not rerun', source_sha256=digest)
    if (OUT / 'protocol.json').exists():
        assert js(OUT / 'protocol.json') == protocol, 'Frozen protocol or source inputs changed'
    else:
        save(OUT / 'protocol.json', protocol)
        save(OUT / 'samples.json', {DS: sample})
        save(OUT / 'baseline_samples.json', allrows)
        save(OUT / 'engine_changes.json', {'engine': 'shipped binary-only engine; no textual patching'})
    if not (OUT / 'frozen_runner.json').exists():
        save(OUT / 'frozen_runner.json', dict(run117_sha256=sha(__file__), engine_sha256=sha(OUT / 'binary_engine.py'), protocol_sha256=sha(OUT / 'protocol.json'), samples_sha256=sha(OUT / 'samples.json'), baseline_samples_sha256=sha(OUT / 'baseline_samples.json'), frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    lock = js(OUT / 'frozen_runner.json')
    for path, field in [(Path(__file__), 'run117_sha256'), (OUT / 'binary_engine.py', 'engine_sha256'), (OUT / 'protocol.json', 'protocol_sha256'), (OUT / 'samples.json', 'samples_sha256'), (OUT / 'baseline_samples.json', 'baseline_samples_sha256')]:
        assert sha(path) == lock[field]
    p = copy.deepcopy(js(ART / 'stage_e/protocol.json'))
    for key in KEYS:
        p['models'][key]['prompts'][DS] = prompt
    config = dict(matched, centering='none', _root=str(ART / 'stage_h'))
    return (p, config, allrows, sample)

def preflight(p, sample):
    import torch, transformers, pyarrow
    from transformers import AutoConfig, AutoTokenizer
    assert torch.cuda.is_available()
    evidence = dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__, pyarrow=pyarrow.__version__, gpu=torch.cuda.get_device_name(0), models={})
    for key in KEYS:
        info = p['models'][key]
        cfg = AutoConfig.from_pretrained(MODELS[key], local_files_only=True, revision=model_revision(MODELS[key]))
        tok = AutoTokenizer.from_pretrained(MODELS[key], use_fast=True, local_files_only=True, revision=model_revision(MODELS[key]))
        assert cfg._commit_hash == info['checkpoint_commit']
        assert sha_token(tok) == info['tokenizer_sha256']
        assert cfg.hidden_size == info['width'] and cfg.num_hidden_layers == info['total_layers']
        ids = {k: tok.encode(k, add_special_tokens=False) for k in ('negative', 'positive')}
        assert all((len(x) == 1 for x in ids.values()))
        prompt = p['models'][key]['prompts'][DS]
        rendered = tok.apply_chat_template([{'role': 'user', 'content': prompt.format(text=sample[0]['text'])}], tokenize=False, add_generation_prompt=True)
        assert prompt.format(text=sample[0]['text']) in rendered and 'neutral' not in prompt
        evidence['models'][key] = dict(commit=cfg._commit_hash, tokenizer_sha256=sha_token(tok), candidate_tokens=ids, rendered_first_example=rendered, chat_template=tok.chat_template, prompt_repr=repr(prompt))
    save(OUT / 'runtime_preflight.json', evidence)

def sha_token(tok):
    return hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()

def run_phase(phase, p, config, allrows, sample):
    H = module('ste117_engine', OUT / 'binary_engine.py')
    H.DATASETS = (DS,)

    def frozen_bases(p, key, dataset, cfg, folder):
        source = ART / f'stage_h/{key}/{DS}'
        meta = js(source / 'bases_metadata.json')
        assert sha(source / 'bases.npz') == meta['bases_sha256']
        with np.load(source / 'bases.npz') as z:
            names = ['emotion_raw', 'emotion_denoised'] + [f'random_rank7_seed_{s}' for s in config['random_seeds']]
            if phase == 'smoke':
                names = ['emotion_raw', 'emotion_denoised', 'random_rank7_seed_2000']
            values = {name: z[name].copy() for name in names}
            anchor = z['anchor'].copy()
        assert not np.any(anchor)
        return (values, anchor, meta)

    def baseline_bases(p, key, dataset, cfg, folder):
        return ({}, np.zeros(p['models'][key]['width'], np.float32), dict(model_role='primary'))
    if phase == 'baseline':
        H.build_bases = baseline_bases
        folder = OUT / 'baseline'
        folder.mkdir(exist_ok=True)
        H.execute(p, config, {DS: allrows}, SimpleNamespace(output=folder, batch_size=8), KEYS)
        return
    folder = OUT / 'smoke/full' if phase == 'smoke' else OUT / 'full'
    folder.mkdir(parents=True, exist_ok=True)
    if phase in ('full', 'smoke'):
        H.build_bases = frozen_bases
        H.execute(p, config, {DS: sample[:8] if phase == 'smoke' else sample}, SimpleNamespace(output=folder, batch_size=8), KEYS)
    if phase in ('matched', 'smoke'):
        import stage_h_distortion_matched as M
        M.H = H
        M.ORIGINAL_REMOVE = H.remove
        source = folder if phase == 'smoke' else OUT / 'full'

        def copy_new_reference(dest, cfg, n):
            db = H.database(dest)
            orig = sqlite3.connect('file:' + (source / 'scores.sqlite3').as_posix() + '?mode=ro', uri=True)
            seeds = cfg['random_seeds'][:1] if phase == 'smoke' else cfg['random_seeds']
            for key in KEYS:
                for name in ['baseline', 'emotion_raw', 'emotion_denoised'] + [f'random_rank7_seed_{s}' for s in seeds]:
                    records = orig.execute('select * from scores where model=? and dataset=? and method=? and example_index<? order by example_index', (key, DS, name, n)).fetchall()
                    assert len(records) == n
                    with db:
                        db.executemany('insert or ignore into scores values (?,?,?,?,?,?,?,?,?,?,?,?,?)', records)
            orig.close()
            db.close()
        M.copy_reference = copy_new_reference
        args = SimpleNamespace(output=OUT / 'smoke' if phase == 'smoke' else OUT / 'matched', batch_size=8, bootstrap=2000, match_tolerance=0.01)
        M.execute(p, config, {DS: sample}, args, phase == 'smoke')
        print('STE117 reference scores came ONLY from newly computed binary-prompt full stage; inherited historical-reuse message above does not apply.', flush=True)
        db = sqlite3.connect(args.output / ('smoke/matching.sqlite3' if phase == 'smoke' else 'matching.sqlite3'))
        total, bad = db.execute('select count(*),sum(feasible=0 or abs(actual_energy-target_energy)>0.01*target_energy+1e-7) from matching').fetchone()
        db.close()
        save(OUT / ('smoke_validation.json' if phase == 'smoke' else 'matching_validation.json'), dict(comparisons=total, failures=bad, smoke=phase == 'smoke'))
        if phase == 'smoke':
            assert total == 32 and bad == 0, 'Smoke energy matching failed'

def preserve(phase):
    for rel, digest in js(OUT / 'protocol.json')['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    save(OUT / ('preservation_' + phase + '.json'), dict(historical_inputs_unchanged=True, phase=phase, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['prepare', 'smoke', 'baseline', 'full', 'matched', 'all'], required=True)
    args = ap.parse_args()
    p, c, allrows, sample = prepare()
    preflight(p, sample)
    with (OUT / 'commands.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(executable=sys.executable, argv=sys.argv, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))) + '\n')
    phases = ['smoke', 'baseline', 'full', 'matched'] if args.phase == 'all' else [] if args.phase == 'prepare' else [args.phase]
    for phase in phases:
        print('START', phase, flush=True)
        run_phase(phase, p, c, allrows, sample)
        preserve(phase)
        print('COMPLETE', phase, flush=True)
    preserve(args.phase)
if __name__ == '__main__':
    main()
