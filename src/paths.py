"""Every directory the pipeline reads or writes, defined once.

Each module used to derive its own output paths from `__file__`, with four
different idioms for the same directory and two that disagreed about where
the run store lived. That is how a figure ends up written somewhere the
script that consumes it does not look. Import from here instead.

Nothing in this file depends on the current working directory, so a module
may be run from anywhere and the entry points that `os.chdir` for the
solver's sake cannot move an output from under its reader.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

# Raw results. Content-addressed by config; the expensive thing in the repo
# and the only directory whose loss costs GPU time. Gitignored.
RESULTS = ROOT / "results"

# Generator scratch: everything the runners draw along the way. Regenerable,
# gitignored, safe to delete wholesale.
OUT = ROOT / "out"
FIG = OUT / "figs"
TAB = OUT / "tables"
TEX = TAB / "tex"

# The figures the manuscript actually cites, and nothing else. Populated by
# sync.py from main.tex's \includegraphics, so the set is derived rather
# than curated. Tracked.
FIGURES = ROOT / "figures"

# Inputs and the manuscript.
DATA = ROOT / "data"
OVERLEAF = ROOT / "overleaf"
MAIN_TEX = OVERLEAF / "main.tex"


def ensure():
    """Make the output directories. Cheap, idempotent, called by the runners."""
    for d in (RESULTS, FIG, TAB, TEX, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
