from release_paths import WORKSPACE, CODE_ROOT, source_path
'Verify completed pilot chunk resume without new Qwen forward passes.'
import sys
sys.dont_write_bytecode = True
from stage_k_inference import *
from run_stage3a import select_pilot
rows = select_pilot(verify_dataset())
results = []
for s in settings():
    folder = P / 'pilot_features' / (s['source'] + '_' + s['model'])
    before = sha(folder / 'completion.json')
    old = js(folder / 'completion.json')['benchmark']

    class NoForward:
        cfg = s['identity']
        mask_sha = mask_reference()[1]
        load_seconds = old['model_load_seconds']

        def batch(self, *args, **kwargs):
            raise AssertionError('Resume attempted forbidden forward')
    arrays, b = run_chunks(NoForward(), s, rows, folder, batch_size=4, chunk_size=8)
    assert sha(folder / 'completion.json') == before and b == old and (len(arrays['ids']) == 32)
    for name, shape in [('sentiment_only', (32, 1)), ('raw', (32, 9)), ('denoised', (32, 9)), ('pca_rank7', (32, 8)), ('random_rank7', (32, 50, 8)), ('pooled', (32, s['identity']['width']))]:
        assert arrays[name].shape == shape
    bad = [dict(r) for r in rows]
    bad[0]['evaluation_id'] = 'mismatched_id_for_resume_check'
    try:
        run_chunks(NoForward(), s, bad, folder, batch_size=4, chunk_size=8)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched resume ID accepted')
    assert sha(folder / 'completion.json') == before
    results.append(dict(configuration=s['source'] + '_' + s['model'], passed=True, chunk_hashes_verified=4, benchmark_preserved=True, new_forward_passes=0, wrong_id_rejected=True))
atomic_json(P / 'resume_verification.json', dict(passed=True, configurations=results))
assert sha(D / 'amazon_test_20000.jsonl') == dataset_sha()
print('RESUME VERIFICATION PASSED: all4 configurations; zero forward passes; benchmark preserved')
state = js(P / 'state.json')
state['resume_verification_passed'] = True
atomic_json(P / 'state.json', state)
