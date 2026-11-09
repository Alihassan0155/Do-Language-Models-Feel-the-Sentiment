"""Prepared Stage3B runner. Explicit --execute-full required; no outcome evaluation."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import sys, argparse
sys.dont_write_bytecode = True
from stage_k_inference import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-full', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=64)
    a = parser.parse_args()
    assert a.execute_full, 'Full inference not authorized by default; pass --execute-full only after approval'
    assert a.batch_size > 0 and a.chunk_size > 0
    state = js(P / 'state.json')
    assert state['complete'] and state['stage3b_ready'] and state['all_source_checks_passed']
    for filename, expected in state['verified_code_sha256'].items():
        assert sha(P / filename) == expected, 'Verified inference code changed: ' + filename
    for filename, expected in state['verified_input_sha256'].items():
        assert sha(K / filename) == expected, 'Frozen input/config changed: ' + filename
    rows = verify_dataset()
    for key in ('0.5b', '3b'):
        e = Extractor(key)
        try:
            for s in settings():
                if s['model'] == key:
                    run_chunks(e, s, rows, K / 'stage3b_inference' / (s['source'] + '_' + key), batch_size=a.batch_size, chunk_size=a.chunk_size)
        finally:
            e.close()
            del e
if __name__ == '__main__':
    main()
