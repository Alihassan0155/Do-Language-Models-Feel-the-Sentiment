"""Compatibility entry point for the sequential Stage E extractor."""
from release_paths import WORKSPACE, CODE_ROOT, source_path
import sys
from run_stage_e import main
if __name__ == '__main__':
    main(['--phase', 'extract', *sys.argv[1:]])
