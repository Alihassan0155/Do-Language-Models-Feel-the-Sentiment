"""Launch one reviewed source entry point with an explicit private workspace."""
import argparse
import os
from pathlib import Path
import runpy
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('script', help='Relative Python source path inside this release')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    source = (code / args.script).resolve()
    if not source.is_relative_to(code) or source.suffix != '.py' or (not source.is_file()):
        parser.error('script must be an existing Python source within this release')
    workspace = args.workspace.expanduser().resolve()
    if workspace == code or code.is_relative_to(workspace):
        parser.error('workspace must not contain the release source directory')
    os.environ['EMOTIONVECTORS_WORKSPACE'] = str(workspace)
    os.environ.setdefault('HF_HOME', str(workspace / '.cache' / 'huggingface'))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(source.parent))
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [str(source), *arguments]
    runpy.run_path(str(source), run_name='__main__')
if __name__ == '__main__':
    main()
