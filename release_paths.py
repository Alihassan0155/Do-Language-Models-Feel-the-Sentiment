"""Separate publishable implementation from locally generated research material."""
import os
import json
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('EMOTIONVECTORS_WORKSPACE', str(CODE_ROOT / 'work'))).expanduser().resolve()

def model_revision(identifier):
    """Require locally recorded user-selected revisions for provider model IDs."""
    if Path(str(identifier)).is_dir():
        return None
    path = WORKSPACE / 'model_revisions.json'
    if not path.is_file():
        raise FileNotFoundError('Acquire models with load_assets.py --model ... --revision ... first')
    revisions = json.loads(path.read_text(encoding='utf-8'))
    name = str(identifier)
    if name == 'bert-large-uncased':
        name = 'google-bert/bert-large-uncased'
    if name not in revisions:
        raise ValueError('No reviewed model revision recorded for ' + name)
    return revisions[name]

def source_path(path):
    """Map legacy source references without redirecting data/artifact reads."""
    p = Path(path)
    if p.suffix != '.py':
        return p
    if p.is_relative_to(CODE_ROOT) and (not p.is_relative_to(WORKSPACE)) and p.exists():
        return p
    try:
        relative = p.relative_to(WORKSPACE)
    except ValueError:
        return p
    if relative.parts and relative.parts[0] == 'artifacts':
        relative = Path('pipeline', *relative.parts[1:])
    candidate = CODE_ROOT / relative
    return candidate if candidate.is_file() else p
