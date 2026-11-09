"""Read-only Stage3B full cache verification; zero model forwards/outcome metrics."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import sys
sys.dont_write_bytecode = True
from run_stage3b_full import *
assert g.js(O / 'run_state.json')['complete']
rows = g.verify_dataset()
lengths = list(csv.DictReader((g.D / 'token_length_report.csv').open(encoding='utf-8')))
checks = []
for s in g.settings():
    name = s['source'] + '_' + s['model']
    checks.append(validate_config(O / 'features' / name, rows, [int(v[name]) for v in lengths]))
    print(name + ':20,000 IDs,313 chunks and every feature verified', flush=True)
assert sum((c['n'] for c in checks)) == 80000 and g.js(O / 'integrity_checks.json')['passed']
tracked = g.js(O / 'preflight.json')['asset_hashes']
for p, h in tracked.items():
    assert g.sha(p) == h, 'Frozen artifact changed: ' + p
state = g.js(P / 'state.json')
for p, h in state['verified_code_sha256'].items():
    assert g.sha(P / p) == h
for p, h in state['verified_input_sha256'].items():
    assert g.sha(K / p) == h
print('READ-ONLY FULL VERIFICATION PASSED;80,000 outputs;1252 chunks;no forwards or metrics')
