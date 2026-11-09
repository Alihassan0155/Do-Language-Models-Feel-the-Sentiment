"""Compatibility entry point for all sequential Stage E probes and controls."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import sys
from run_stage_e import main
if __name__ == '__main__':
    main(['--phase', 'fit', *sys.argv[1:]])
