# Channel-Preconditioned PDHG for Accelerating Low-Frequency Convergence in Limited-Angle Tomosynthesis

Taro Iyadomi, Ricardo Parada, Anna Kim, Lily Jiang, Emil Sidky, and William Chang

Single- versus two-channel Chambolle–Pock reconstruction with directional total
variation for limited-angle tomography, with digital breast tomosynthesis as the
motivating case. Splitting the data fidelity into low- and high-frequency
channels and amplifying the low-frequency dual step accelerates low-frequency
convergence; a preconditioned analysis certifies the amplified step under the
sharp condition `tau * lambda_max(M) < 1`.

## Layout

```
src/        every module the paper depends on, and nothing else
scripts/    setup_env.sh (dependencies), pipeline.sh (the whole run)
results/    raw results, content-addressed by config   (gitignored)
figures/    the figures the manuscript cites, and only those
out/        generator scratch: everything drawn along the way (gitignored)
data/       phantoms and the VICTRE downsample cache
```

`figures/` is written by `src/sync.py`, which reads the manuscript's
`\includegraphics` calls — so "which figures does the paper use" is answered by
the paper, not by a list someone maintains. `out/` can be deleted at any time.

## Requirements

- Python 3.12+
- **2D experiments**: CPU only — numpy, matplotlib, and optionally scipy and
  numba. No GPU.
- **3D experiments**: additionally ASTRA and an NVIDIA GPU. ASTRA's 3D
  projectors are CUDA-only, so there is no CPU fallback.

Activate whatever environment you prefer — venv, conda, uv, the system
interpreter — then:

```bash
bash scripts/setup_env.sh           # install requirements.txt, report what is usable
bash scripts/setup_env.sh --check   # report only, install nothing
```

It deliberately does not create or activate an environment for you. It prints
one line per dependency saying what that dependency buys and whether its
absence is fatal, so a machine without a GPU is told which stages it cannot run
rather than discovering it three hours in.

ASTRA is not on PyPI and so is not in `requirements.txt`:

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox   # or your own route
```

## The VICTRE phantom

The source `.raw` is 601 MB and is not tracked, so a fresh clone cannot build
the 3D phantom on its own. The downsampled volume the experiments actually
consume is far smaller — 2.5 MB at `--down 4`, since the volume is label-derived
and holds only a handful of distinct attenuation values — and is cached to
`data/victre_cache/`.

If `data/victre_cache/` is already in the repo, there is nothing to do. To
(re)build it on a machine that has the `.raw` (no GPU needed):

```bash
python src/victre3d.py --cache-only --down 4
```

Build one cache per downsampling factor you intend to use. Committing the cache
lets the phantom travel with the repo; otherwise copy `data/victre_cache/` to
the target machine. Loading prefers the cache and falls back to the `.raw`,
erroring clearly if neither is present.

The published runs use one particular VICTRE phantom, which is the default. It
is the name of a file somebody downloaded, not a property of the method, so it
is overridable:

```bash
python src/victre3d.py --phantom-id p_12345678      # or VICTRE_PHANTOM=p_12345678
```

The id is part of the cache key, so two phantoms at the same downsampling
factor never collide.

## Running the experiments

`scripts/pipeline.sh` runs everything end to end — diagnostics, 2D, then 3D. Verify the
environment first; this prints versions, CUDA availability, and the GPU, then
exits without running anything:

```bash
STAGES=none bash scripts/pipeline.sh
```

You want `astra CUDA: True` and your GPU listed. Then start the real run under
`tmux` so it survives disconnection:

```bash
mkdir -p logs
tmux new -s ct
bash scripts/pipeline.sh 2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M).log
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t ct`. Follow along
from another shell with `tail -f logs/pipeline_*.log`.

**If it dies partway, just run it again.** The 2D stages key each result on a
hash of its config, so completed runs return instantly and only unfinished work
recomputes. A failing stage does not abort the script — failures are collected
and listed in the summary, so one broken stage overnight does not cost the rest.
The 3D stages do not yet have this and will recompute in full, which is why they
are staged last.

### Knobs

| Variable | Default | Meaning |
|---|---|---|
| `STAGES` | `diag 2d 3d analysis` | which stage groups to run; `STAGES=none` is the env check |
| `REPS` | `30` | Poisson realizations for the 2D statistics stage |
| `REPS3D` | `20` | Poisson realizations for the 3D statistics stage |
| `REPS_TOL` / `REPS_BAND` | `20` | realizations for the two sweeps |
| `MIN_REPS` | `20` | floor the audit stage enforces on anything the paper reports |
| `ITERMAX2D` / `ITERMAX3D` | `2000` | iteration budget, and the budget the tables and audit are read at |
| `SEEDS` | `1 2 3 4 5` | seeds for the multi-seed stability checks |
| `SKIP_SLOW` | `0` | `1` skips the slowest sweeps |
| `PY` | `python` | interpreter |

```bash
STAGES=2d bash scripts/pipeline.sh                  # CPU only, no GPU needed
SKIP_SLOW=1 bash scripts/pipeline.sh                # quick pass to check everything runs
STAGES=3d SEEDS="1 2 3" bash scripts/pipeline.sh    # GPU work only, fewer seeds
```

A first `SKIP_SLOW=1 STAGES=diag bash scripts/pipeline.sh` is worth the few minutes — it
confirms the environment end to end before committing to a long run.

### Individual experiments

```bash
# Operator-level stability diagnostic
python src/stability.py

# 2D convergence: single vs two-channel, all phantoms
python src/convergence2d.py --phantom all

# Sensitivity to the low-band tolerance (20 realizations)
python src/tolerance.py --phantom breast --reps 20

# Error resolved by frequency band
python src/bands.py --phantom breast --reps 20
```

## Output

| Where | What | Tracked |
|---|---|---|
| `results/` | raw arrays, one `.npz` per run with a sibling `.json` manifest | no |
| `out/figs`, `out/tables` | every figure and table a runner draws | no |
| `figures/` | the figures the manuscript cites | yes |

Each manifest records the full config, package versions, git commit, GPU and
wall-clock, so a result stays auditable months later instead of being just a
picture. The config hash is the filename, which is why a rerun is free and why
two runs that differ in any parameter can never overwrite each other.

Nothing is keyed by realization count or iteration budget in its *name* —
those live in the file's own header and in the manifest. One sweep, one file.

If you run the heavy stages elsewhere, copy `results/` back and rerun the
analysis stages locally; every figure and table redraws from the store without
a GPU:

```bash
STAGES=analysis bash scripts/pipeline.sh
```

### Checking the paper against the results

```bash
python src/audit.py --itermax-2d 2000 --itermax-3d 2000   # realization counts
python src/sync.py --check                                # tables and figures in the .tex
```

`audit.py` fails if anything the manuscript reports rests on fewer than 20
noise realizations, and names what it could not check rather than passing
silently. `sync.py` rewrites the manuscript's inlined tables and copies cited
figures into `figures/`; `--check` reports drift and changes nothing.

## Two kinds of repeated runs — they are not interchangeable

**Seed sweeps are stability checks, not statistics.** With noiseless data the
only stochastic element is the power-iteration start vector, so a seed sweep
measures how sensitive the certified step sizes are to it. Worth knowing — it is
how `npower` was chosen — but it is not a sample over noise realizations and
must not be reported as error bars.

**Error bars come from `run_2d_noise_stats.py`**, which draws independent
Poisson realizations of the transmission measurement:

```bash
python src/noise2d.py --phantom all --reps 30
```

Both arms reconstruct the *same* realization, so the comparison is paired and
the per-realization difference removes variation the two arms share. The solver
seed is held fixed while only the noise seed varies, so the reported spread is
measurement noise rather than step-size jitter. Output gives mean paired
difference with a 95% CI, median and IQR, a Wilcoxon signed-rank p, and a win
rate.

Two settings matter and default to the right thing:

- `--i0 auto` scales incident photons with each phantom's peak line integral.
  A fixed `i0` clamps zero-count rays on thick phantoms — Defrise needs ~8e6
  where Shepp-Logan needs ~1e3 — and silently biases those line integrals low.
  Because each phantom then sits at its own dose, absolute RMSE is **not**
  comparable across phantoms; the paired within-phantom comparison is.
- The data tolerance `eps` is recalibrated to the noise from an independent
  draw. This is not cosmetic: at `i0=1e5` the calibrated `eps` is ~0.137 against
  a noiseless 0.001, so running noise at the noiseless tolerance puts the truth
  far outside the feasible set and produces meaningless reconstructions.

## References

Sidky, E. Y., et al. "Accurate volume image reconstruction for digital breast
tomosynthesis with directional-gradient and pixel sparsity regularization."
*Journal of Medical Imaging* 12.S1 (2025): S13013.

Chambolle, A., and Pock, T. "A first-order primal-dual algorithm for convex
problems with applications to imaging." *Journal of Mathematical Imaging and
Vision* 40.1 (2011): 120–145.
