"""Allow `python -m text_theme_analyzer ...`."""

import sys
from pathlib import Path

from text_theme_analyzer.cli import _absorb_expanded_glob_args, _warn_if_glob_expanded, main

# Recover from shell-expanded globs (see cli._absorb_expanded_glob_args).
# The user's command `--include "**/*.md"` may have been expanded by the
# shell into a list of file paths before Python saw it. This repair
# step folds them back into the include list so Click doesn't see them
# as unexpected extra arguments.
_original_argv = sys.argv[1:]
sys.argv[1:] = _absorb_expanded_glob_args(sys.argv[1:])

# Warn the user if a glob was eaten.
if _original_argv and len(sys.argv) > 1:
    rest = sys.argv[1:]
    if rest and rest[0] == "analyze" and len(rest) >= 2:
        try:
            ip = Path(rest[1])
        except (IndexError, ValueError):
            ip = None
        if ip is not None:
            _warn_if_glob_expanded(_original_argv, ip)

if __name__ == "__main__":
    sys.exit(main())
