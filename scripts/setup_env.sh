#!/usr/bin/env bash
# Install the dependencies and report what this machine can run.
#
# Deliberately does not create or activate an environment. Which tool you use
# -- venv, conda, uv, the system interpreter -- is your business, and a script
# that picks one for you is a script that fights your setup. Activate whatever
# you like, then run this inside it.
#
#   python -m venv .venv && . .venv/bin/activate   # or conda, or uv, or none
#   bash scripts/setup_env.sh
#
#   PY=python3.12 bash scripts/setup_env.sh        # a specific interpreter
#   bash scripts/setup_env.sh --check              # report only, install nothing

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
CHECK=0; [[ "${1:-}" == "--check" ]] && CHECK=1

command -v "$PY" >/dev/null || { echo "no interpreter '$PY' on PATH"; exit 1; }
echo "interpreter: $("$PY" -c 'import sys;print(sys.executable, sys.version.split()[0])')"

if ((!CHECK)); then
    echo
    echo "== installing requirements.txt"
    "$PY" -m pip install -r requirements.txt || {
        echo "pip install failed -- fix the above and rerun"; exit 1; }
fi

echo
echo "== what this machine can run"
# Each line: module, what it is needed for, and whether its absence is fatal.
"$PY" - <<'PYEOF'
import importlib, shutil, sys

NEEDED = [
    ("numpy",      "everything",                              True),
    ("scipy",      "Wilcoxon p-values and t-intervals",       False),
    ("matplotlib", "every figure",                            True),
    ("numba",      "the 2D solver's inner loops",             False),
    ("astra",      "all 3D geometry (cone-beam projectors)",  False),
]
missing_hard = []
for name, why, fatal in NEEDED:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"  yes  {name:<11} {v:<10} {why}")
    except Exception:
        print(f"  NO   {name:<11} {'':<10} {why}"
              + ("   <-- required" if fatal else "   (optional)"))
        if fatal:
            missing_hard.append(name)

# ASTRA needs a CUDA device, not just the package. Report the distinction:
# "astra imports" and "astra can project" are different claims.
gpu = shutil.which("nvidia-smi")
print(f"\n  {'yes' if gpu else 'no '}  CUDA GPU    "
      + ("nvidia-smi found" if gpu else
         "no nvidia-smi -- the 3D stages cannot run here"))

print("\nWithout scipy, intervals fall back to the normal approximation and "
      "p-values are omitted.\nWithout numba the 2D solver still runs, slowly."
      "\nWithout ASTRA and a GPU, the 2D stages run and the 3D stages fail; "
      "the published\n3D numbers are in results/ and the figures redraw from "
      "them without a GPU.")
sys.exit(1 if missing_hard else 0)
PYEOF
rc=$?

echo
if ((rc)); then
    echo "Something required is missing -- see above."
else
    echo "Ready. Run the pipeline with:  bash scripts/pipeline.sh"
fi
exit $rc
