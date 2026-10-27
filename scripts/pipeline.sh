#!/usr/bin/env bash
# Run every experiment end to end. Designed to be started once inside tmux and
# left alone for hours.
#
#     tmux new -s ct
#     ./run_all.sh 2>&1 | tee logs/run_all_$(date +%Y%m%d_%H%M).log
#     # Ctrl-b d to detach, tmux attach -t ct to come back
#
# Safe to re-run. The 2D stages go through runstore, which keys results on a
# config hash: unchanged config returns the stored arrays instantly, changed
# methodology gets a new key. So an interrupted run resumes where it stopped
# rather than starting over. The 3D stages are not yet on runstore (see
# STATUS.md item 2) and DO recompute, which is why they are staged last.
#
# Env knobs:
#   STAGES="2d 3d"   which stage groups to run (default: all)
#   ITERS=5000       2D long-run length
#   SEEDS="1 2 3"    seeds for the multi-seed stability checks
#   REPS=30          Poisson noise realizations for the 2D statistics stage
#   MIN_REPS=20      floor enforced by the audit stage; nothing reported may
#                    rest on fewer realizations than this
#   ITERMAX2D=2000   iteration budget for the 2D statistics stage; also the
#                    budget the tables and the audit are read at
#   ITERMAX3D=2000   iteration budget for the 3D statistics stage
#   REPS3D=10        Poisson noise realizations for the 3D statistics stage
#   PY=python        interpreter
#   SKIP_SLOW=1      skip the 20000-iteration reference runs and 3D sweeps

set -uo pipefail            # NOT -e: one failing stage must not kill the night

cd "$(dirname "$0")/.."
PY="${PY:-python}"
ITERS="${ITERS:-5000}"
SEEDS="${SEEDS:-1 2 3 4 5}"
REPS="${REPS:-30}"
# 3D realizations are separately tunable: each is a full single+two GPU solve,
# so the 2D count of 30 would be a multi-day stage on its own. It may not go
# below MIN_REPS -- no reported number may rest on fewer realizations than
# that, and the audit stage at the end fails the run if one does.
REPS3D="${REPS3D:-20}"
MIN_REPS="${MIN_REPS:-20}"
# One budget per dimension, set here and threaded through the runs, the tables
# and the audit. Defining it in three places is how a figure ends up captioned
# at one iteration count and computed at another.
ITERMAX2D="${ITERMAX2D:-2000}"
ITERMAX3D="${ITERMAX3D:-2000}"
# Realizations for the two sweeps that were historically single-realization.
REPS_TOL="${REPS_TOL:-20}"
REPS_BAND="${REPS_BAND:-20}"
STAGES="${STAGES:-diag 2d 3d analysis}"
SKIP_SLOW="${SKIP_SLOW:-0}"
# figs/ and tables/ are gitignored, so a fresh clone has neither.
mkdir -p logs out/figs out/tables results figures

START=$(date +%s)
FAILED=()
declare -a TIMINGS=()        # assigned, not just declared: `declare -a X`
                             # leaves X unset, so ${#X[@]} trips `set -u`

hdr() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

# Run a stage, keep going on failure, record how long it took. A stage that dies
# at 3am must not silently look like a stage that was never reached, so failures
# are collected and reported in the summary at the end.
stage() {
    local name="$1"; shift
    hdr "$name"
    echo "\$ $*"
    local t0 t1
    t0=$(date +%s)
    if "$@"; then
        t1=$(date +%s)
        TIMINGS+=("$(printf '%-42s %6ss  ok' "$name" "$((t1 - t0))")")
    else
        local rc=$?
        t1=$(date +%s)
        TIMINGS+=("$(printf '%-42s %6ss  FAILED (rc=%s)' "$name" "$((t1 - t0))" "$rc")")
        FAILED+=("$name")
        echo ">>> STAGE FAILED: $name (rc=$rc) -- continuing"
    fi
}

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# --------------------------------------------------------------------------
hdr "environment"
$PY - <<'EOF'
import platform, sys
print("python  ", sys.version.split()[0], "|", platform.platform())
for m in ("numpy", "scipy", "numba", "matplotlib", "astra"):
    try:
        mod = __import__(m)
        print(f"{m:9s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"{m:9s} MISSING ({type(e).__name__})")
try:
    import astra
    print("astra CUDA:", astra.astra.use_cuda())
except Exception as e:
    print("astra CUDA: unavailable —", e)
EOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi: not found (3D stages will fail)"
git rev-parse HEAD 2>/dev/null | sed 's/^/commit  /'

# --------------------------------------------------------------------------
# Diagnostics. Cheap, and they gate the interpretation of everything else --
# if the certified step sizes are not trustworthy, no downstream number is.
# --------------------------------------------------------------------------
if has_stage diag; then
    stage "2D operator stability diagnostic" \
        $PY src/stability.py

    # Feeds tab:stability's margin column and fig:bandsplit(b). Neither is
    # derivable from the diagnostic above: it reports one s, this sweeps s.
    stage "2D stability margin vs step scale" \
        $PY src/stability_curve.py

    # Feeds fig:mechanism(b), the gain-vs-band-content panel. A property of
    # the phantoms and geometry alone, so it needs no reconstruction.
    stage "2D band energy per phantom" \
        $PY src/band_energy.py
fi

# --------------------------------------------------------------------------
# 2D. CPU only, no ASTRA, no GPU. This is the headline result: the long-run
# collapse study that turns "lower RMSE at iteration 500" into "same solution
# N times faster". Runs first because it needs nothing from the GPU.
# --------------------------------------------------------------------------
if has_stage 2d; then
    stage "2D convergence, all phantoms (500 iters)" \
        $PY src/convergence2d.py --phantom all

    # THE statistics run: REPS independent Poisson realizations per phantom,
    # paired single-vs-two on each realization, reported as mean difference with
    # a 95% CI plus median/IQR and a Wilcoxon p. i0 defaults to 'auto', scaling
    # with each phantom's peak line integral -- a fixed i0 clamps zero-count
    # rays on the thick phantoms (Defrise needs ~8e6 where Shepp-Logan needs
    # ~1e3) and silently biases those line integrals low.
    stage "2D Poisson noise statistics (${REPS} realizations, all phantoms)" \
        $PY src/noise2d.py --phantom all \
            --reps "$REPS" --iters "$ITERMAX2D"
fi

# --------------------------------------------------------------------------
# 3D. Requires ASTRA + CUDA. Staged last: it is the part that is NOT yet
# resumable, so if the night runs short the cheap resumable work is already done.
# --------------------------------------------------------------------------
if has_stage 3d; then
    # THE 3D statistics run. Until now every 3D number in the paper came from a
    # single realization, and the 3D seed sweep below measures step-size jitter,
    # NOT measurement noise -- the same distinction the 2D stages draw. This is
    # the 3D counterpart of the 2D Poisson stage: solver seed fixed, only the
    # noise seed varying, arms paired within each realization. It runs first in
    # the 3D block because it is the only 3D stage on runstore and so the only
    # one that resumes rather than restarting if the night runs short.
    # itermax is the published budget, and the per-band tolerance mode is the
    # one every reported 3D number uses. A longer run reproduces a shorter one
    # exactly -- PDHG's trajectory does not depend on where it is stopped --
    # so the budget here sets the ceiling, and anything shorter can still be
    # read off the stored trace afterwards.
    stage "3D VICTRE Poisson noise statistics (${REPS3D} realizations)" \
        $PY src/noise3d.py --down 4 \
            --reps "$REPS3D" --itermax "$ITERMAX3D" \
            --eps-mode noise_per_band

    # fig:dbt. Writes its slices to results/figures3d/ so the figure can be
    # restyled later on a machine with no GPU.
    stage "3D DBT reconstruction panel" \
        $PY src/figures3d.py --down 4 --itermax "$ITERMAX3D" \
            --eps-mode noise_per_band

    stage "3D VICTRE (500 iters)" \
        $PY src/victre3d.py --down 4 --itermax 500


    if [[ "$SKIP_SLOW" != "1" ]]; then
        # Multi-seed VICTRE. Same caveat as the 2D seed sweep: this probes
        # step-size sensitivity, not noise statistics. --tag keeps the runs
        # from overwriting each other's figures and tables.
        for s in $SEEDS; do
            stage "3D VICTRE seed $s (stability check)" \
                $PY src/victre3d.py --down 4 --itermax 300 \
                    --seed "$s" --tag "seed$s"
        done

        # s-sweep: without s=1 there is no control separating "the split
        # helped" from "the amplification helped", and without s=8 nothing
        # shows what happens past the certified bound. The 2D story has all
        # three; 3D has only ever run s=4.
        for slo in 1 4 8; do
            stage "3D VICTRE sigma_lo scale s=$slo" \
                $PY src/victre3d.py --down 4 --itermax 300 \
                    --slo "$slo" --tag "s$slo"
        done
    fi
fi

# --------------------------------------------------------------------------
# Analysis. Cheap, CPU-only, and dependent on the stages above, so it runs
# last and can be re-run on its own with STAGES=analysis after a pull.
# --------------------------------------------------------------------------
if has_stage analysis; then
    stage "2D tolerance sweep (${REPS_TOL} realizations)" \
        $PY src/tolerance.py --phantom breast \
            --reps "$REPS_TOL"

    stage "2D band-resolved error (${REPS_BAND} realizations)" \
        $PY src/bands.py --phantom breast \
            --reps "$REPS_BAND"

    stage "paper figures" \
        $PY src/figures.py

    # fig:gallery and fig:ladder. Reconstructions are read from the noise
    # store, so this is a redraw at the published budget, not a new solve.
    stage "reconstruction galleries" \
        $PY src/galleries.py --itermax "$ITERMAX2D" --eps-mode noise_per_band

    stage "manuscript tables" \
        $PY src/tables.py \
            --itermax-2d "$ITERMAX2D" --itermax-3d "$ITERMAX3D"

    # The manuscript carries inlined copies so the Overleaf bundle is
    # self-contained; this is what keeps them from drifting from the store.
    stage "sync tables into the manuscript" \
        $PY src/sync.py

    # Last, and allowed to fail the run: this is the check that no reported
    # number rests on too few realizations. A pipeline that produced the
    # tables but cannot pass this has not finished its job.
    stage "realization-count audit" \
        $PY src/audit.py --min-reps "$MIN_REPS" \
            --itermax-2d "$ITERMAX2D" --itermax-3d "$ITERMAX3D"
fi

# --------------------------------------------------------------------------
hdr "summary"
# Guarded: expanding an empty array under `set -u` errors on bash < 4.4.
((${#TIMINGS[@]})) && printf '%s\n' "${TIMINGS[@]}"
printf '\ntotal wall-clock: %s s\n' "$(( $(date +%s) - START ))"
if ((${#FAILED[@]})); then
    printf '\n\033[1;31m%s stage(s) FAILED:\033[0m\n' "${#FAILED[@]}"
    printf '  - %s\n' "${FAILED[@]}"
    echo
    echo "Re-running is cheap for the 2D stages: runstore returns completed"
    echo "runs instantly, so only the failed work recomputes."
    exit 1
fi
echo "all stages completed"
echo "scratch  -> out/figs/, out/tables/   (regenerable, gitignored)"
echo "cited    -> figures/                  (written by src/sync.py)"
echo "raw      -> results/                  (runstore, gitignored)"
