"""Run a fresh, resumable sequential Stage E experiment (E1-E5)."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import shutil
import sys
from pathlib import Path
from stage_e_common import ROOT, E, MODELS, atomic_json

def reset_outputs():
    workspace = ROOT.resolve()
    targets = [workspace / 'artifacts' / 'stage_e', workspace / 'visualizations' / 'stage_e']
    for target in targets:
        resolved = target.resolve()
        if str(resolved).casefold() != str(target).casefold() or target.is_symlink():
            raise RuntimeError(f'Unsafe reset target: {target} resolves to {resolved}')
        if not resolved.is_relative_to(workspace):
            raise RuntimeError('Reset target leaves the project.')
        print(f'Verified Stage E reset target: {resolved}', flush=True)
    for target in targets:
        if target.exists():
            shutil.rmtree(target)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'extract', 'fit', 'evaluate', 'report', 'all'], default='all')
    parser.add_argument('--reset', action='store_true', help='Delete only generated Stage E artifacts and Stage E plots; earlier stages untouched.')
    parser.add_argument('--model', choices=[*MODELS, 'all'], default='all')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, help='Extraction smoke test only; resume extraction without this flag.')
    parser.add_argument('--template', choices=['template_1'], default='template_1')
    parser.add_argument('--save-hidden', action='store_true', help='Compatibility flag; fresh selected-layer states are always saved.')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size and limit must be positive.')
    if args.limit is not None and args.phase != 'extract':
        parser.error('--limit is allowed only with --phase extract.')
    if args.model != 'all' and args.phase in ('all', 'evaluate', 'report'):
        parser.error('Fit/extract subsets separately; evaluation/report/all require all models for complete comparison results.')
    if args.reset:
        reset_outputs()
    from prepare_stage_e import prepare
    from stage_e_pipeline import protocol, assert_protocol, extract_models, fit_all, evaluate_all, report
    prepare(seed=42, val_fraction=0.2)
    p = protocol(args.batch_size)
    assert_protocol(p)
    view = dict(p)
    if args.model != 'all':
        view['models'] = {args.model: p['models'][args.model]}
    state = E / 'state.json'
    try:
        if args.phase in ('extract', 'all'):
            atomic_json(state, dict(phase='extracting', model=args.model, complete=False))
            extract_models(view, args.batch_size, args.limit)
            atomic_json(state, dict(phase='extracted' if args.limit is None else 'smoke_extraction', complete=False))
        if args.phase in ('fit', 'all'):
            atomic_json(state, dict(phase='fitting', model=args.model, complete=False))
            fit_all(view)
            atomic_json(state, dict(phase='fitted', complete=False))
        if args.phase in ('evaluate', 'all'):
            atomic_json(state, dict(phase='evaluating', complete=False))
            evaluate_all(p)
            atomic_json(state, dict(phase='evaluated', complete=False))
        if args.phase in ('report', 'all'):
            report(p)
            atomic_json(state, dict(phase='complete', complete=True))
        if args.phase == 'prepare':
            atomic_json(state, dict(phase='prepared', complete=False))
            print(f'Prepared locked Stage E protocol: {E / 'protocol.json'}', flush=True)
    except Exception as error:
        previous = json.loads(state.read_text(encoding='utf-8')) if state.exists() else {}
        atomic_json(state, dict(phase=previous.get('phase', args.phase), complete=False, error_type=type(error).__name__, error=str(error)))
        raise
if __name__ == '__main__':
    main()
