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
#   REPS=30          Poisson noise realizations for the statistics stage
#   PY=python        interpreter
#   SKIP_SLOW=1      skip the 20000-iteration reference runs and 3D sweeps

set -uo pipefail            # NOT -e: one failing stage must not kill the night

cd "$(dirname "$0")"
PY="${PY:-python}"
ITERS="${ITERS:-5000}"
SEEDS="${SEEDS:-1 2 3 4 5}"
REPS="${REPS:-30}"
STAGES="${STAGES:-diag 2d 3d}"
SKIP_SLOW="${SKIP_SLOW:-0}"
mkdir -p logs

START=$(date +%s)
FAILED=()
declare -a TIMINGS

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
        $PY paper/experiments/stability_diagnostic.py
    stage "2D dyadic stability diagnostic" \
        $PY paper/experiments/dyadic_stability_diagnostic.py
    if [[ "$SKIP_SLOW" != "1" ]]; then
        # Answers "is npower large enough for tau*lambda_max(M)<1 to hold?"
        # Needs a GPU; harmless to attempt without one (it will just fail).
        stage "3D power-iteration convergence (synthetic)" \
            $PY paper/experiments/check_power_convergence.py --geom synthetic
        stage "3D power-iteration convergence (VICTRE)" \
            $PY paper/experiments/check_power_convergence.py --geom victre --down 4
    fi
fi

# --------------------------------------------------------------------------
# 2D. CPU only, no ASTRA, no GPU. This is the headline result: the long-run
# collapse study that turns "lower RMSE at iteration 500" into "same solution
# N times faster". Runs first because it needs nothing from the GPU.
# --------------------------------------------------------------------------
if has_stage 2d; then
    stage "2D convergence, all phantoms (500 iters)" \
        $PY paper/experiments/run_2d_convergence.py --phantom all

    stage "2D epsilon sensitivity" \
        $PY paper/experiments/eps_loosening.py

    stage "2D dyadic convergence" \
        $PY paper/experiments/dyadic_convergence.py

    # The collapse study. ref-iters defaults to 4x, so this also runs a
    # 20000-iteration reference per phantom -- the expensive part.
    if [[ "$SKIP_SLOW" == "1" ]]; then
        stage "2D long-run collapse (short ref)" \
            $PY paper/experiments/run_2d_longrun.py --phantom all \
                --iters "$ITERS" --ref-iters "$ITERS"
    else
        stage "2D long-run collapse (${ITERS} iters, 4x ref)" \
            $PY paper/experiments/run_2d_longrun.py --phantom all --iters "$ITERS"
    fi

    # Multi-seed repeat. NOTE what this does and does not measure: the forward
    # data here is NOISELESS, so the only stochastic element is the
    # power-iteration start vector. This measures step-size sensitivity, not
    # statistical spread -- it is a stability check, not error bars. The
    # noise-statistics stage below is where error bars come from.
    for s in $SEEDS; do
        stage "2D long-run seed $s (stability check, defrise)" \
            $PY paper/experiments/run_2d_longrun.py --phantom defrise \
                --iters "$ITERS" --seed "$s"
    done

    # THE statistics run: REPS independent Poisson realizations per phantom,
    # paired single-vs-two on each realization, reported as mean difference with
    # a 95% CI plus median/IQR and a Wilcoxon p. i0 defaults to 'auto', scaling
    # with each phantom's peak line integral -- a fixed i0 clamps zero-count
    # rays on the thick phantoms (Defrise needs ~8e6 where Shepp-Logan needs
    # ~1e3) and silently biases those line integrals low.
    stage "2D Poisson noise statistics (${REPS} realizations, all phantoms)" \
        $PY paper/experiments/run_2d_noise_stats.py --phantom all \
            --reps "$REPS" --iters 500
fi

# --------------------------------------------------------------------------
# 3D. Requires ASTRA + CUDA. Staged last: it is the part that is NOT yet
# resumable, so if the night runs short the cheap resumable work is already done.
# --------------------------------------------------------------------------
if has_stage 3d; then
    stage "3D analytic defrise" \
        $PY paper/experiments/run_3d_analytic.py --full

    stage "3D analytic breast" \
        $PY paper/experiments/run_3d_breast.py --itermax 300

    stage "3D VICTRE (500 iters)" \
        $PY paper/experiments/run_3d_victre.py --down 4 --itermax 500

    stage "3D VICTRE lesion detectability" \
        $PY paper/experiments/run_3d_lesion.py --down 4 --itermax 200

    if [[ "$SKIP_SLOW" != "1" ]]; then
        # Multi-seed VICTRE. Same caveat as the 2D seed sweep: this probes
        # step-size sensitivity, not noise statistics. --tag keeps the runs
        # from overwriting each other's figures and tables.
        for s in $SEEDS; do
            stage "3D VICTRE seed $s (stability check)" \
                $PY paper/experiments/run_3d_victre.py --down 4 --itermax 300 \
                    --seed "$s" --tag "seed$s"
        done

        # s-sweep: without s=1 there is no control separating "the split
        # helped" from "the amplification helped", and without s=8 nothing
        # shows what happens past the certified bound. The 2D story has all
        # three; 3D has only ever run s=4.
        for slo in 1 4 8; do
            stage "3D VICTRE sigma_lo scale s=$slo" \
                $PY paper/experiments/run_3d_victre.py --down 4 --itermax 300 \
                    --slo "$slo" --tag "s$slo"
        done
    fi
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
echo "figures -> paper/experiments/figs/    tables -> paper/experiments/tables/"
echo "raw arrays -> paper/experiments/results/  (gitignored)"
