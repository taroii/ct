# Channel-Preconditioned PDHG for Accelerating Low-Frequency Convergence in Limited-Angle Tomosynthesis

Taro Iyadomi, Ricardo Parada, Anna Kim, Lily Jiang, Emil Sidky, and William Chang

Single- versus two-channel Chambolle–Pock reconstruction with directional total
variation for limited-angle tomography, with digital breast tomosynthesis as the
motivating case. Splitting the data fidelity into low- and high-frequency
channels and amplifying the low-frequency dual step accelerates low-frequency
convergence; a preconditioned analysis certifies the amplified step under the
sharp condition `tau * lambda_max(M) < 1`.

This branch targets the journal paper (`paper/manuscripts/new/template.tex`).
The earlier CT-Meeting 2026 abstract and its deck are still in the tree under
`paper/manuscripts/abstract/` and `presentation/`.

## Requirements

- Python 3.13, conda
- **2D experiments**: CPU only — numpy, scipy, matplotlib, numba. No GPU.
- **3D experiments**: additionally ASTRA and an NVIDIA GPU. ASTRA's 3D
  projectors are CUDA-only, so there is no CPU fallback.

```bash
conda create -n ct python=3.13.7 -y
conda activate ct
pip install -r requirements.txt

# 3D only
conda install -c astra-toolbox -c nvidia astra-toolbox -y
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
python paper/experiments/run_3d_victre.py --cache-only --down 4
```

Build one cache per downsampling factor you intend to use. Committing the cache
lets the phantom travel with the repo; otherwise copy `data/victre_cache/` to
the target machine. Loading prefers the cache and falls back to the `.raw`,
erroring clearly if neither is present.

## Running the experiments

`run_all.sh` runs everything end to end — diagnostics, 2D, then 3D. Verify the
environment first; this prints versions, CUDA availability, and the GPU, then
exits without running anything:

```bash
STAGES=none ./run_all.sh
```

You want `astra CUDA: True` and your GPU listed. Then start the real run under
`tmux` so it survives disconnection:

```bash
mkdir -p logs
tmux new -s ct
./run_all.sh 2>&1 | tee logs/run_all_$(date +%Y%m%d_%H%M).log
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t ct`. Follow along
from another shell with `tail -f logs/run_all_*.log`.

**If it dies partway, just run it again.** The 2D stages key each result on a
hash of its config, so completed runs return instantly and only unfinished work
recomputes. A failing stage does not abort the script — failures are collected
and listed in the summary, so one broken stage overnight does not cost the rest.
The 3D stages do not yet have this and will recompute in full, which is why they
are staged last.

### Knobs

| Variable | Default | Meaning |
|---|---|---|
| `STAGES` | `diag 2d 3d` | which stage groups to run; `STAGES=none` is the env check |
| `ITERS` | `5000` | 2D long-run length |
| `SEEDS` | `1 2 3 4 5` | seeds for the multi-seed stability checks |
| `SKIP_SLOW` | `0` | `1` skips the 20000-iteration reference runs and the 3D sweeps |
| `PY` | `python` | interpreter |

```bash
STAGES=2d ./run_all.sh                  # CPU only, no GPU needed
SKIP_SLOW=1 ./run_all.sh                # quick pass to check everything runs
STAGES=3d SEEDS="1 2 3" ./run_all.sh    # GPU work only, fewer seeds
```

A first `SKIP_SLOW=1 STAGES=diag ./run_all.sh` is worth the few minutes — it
confirms the environment end to end before committing to a long run.

### Individual experiments

```bash
# Operator-level stability diagnostic
python paper/experiments/stability_diagnostic.py

# 2D convergence: single vs two-channel, all phantoms
python paper/experiments/run_2d_convergence.py --phantom all

# 2D long-run: do the s = 1, 4, 8 curves converge to a common limit?
python paper/experiments/run_2d_longrun.py --phantom all --iters 5000

# How many power iterations the certified bound needs, for a given geometry
python paper/experiments/check_power_convergence.py --geom victre --down 4
```

## Output

Figures land in `paper/experiments/figs/` and result tables in
`paper/experiments/tables/`. Raw arrays go to `paper/experiments/results/`
(not tracked): each `.npz` has a sibling `.json` manifest recording the full
config, package versions, git commit, GPU, and wall-clock, so a result stays
auditable months later instead of being just a picture.

To bring results back from a server:

```bash
rsync -av server:ct/paper/experiments/figs/   paper/experiments/figs/
rsync -av server:ct/paper/experiments/tables/ paper/experiments/tables/
```

## Two kinds of repeated runs — they are not interchangeable

**Seed sweeps are stability checks, not statistics.** With noiseless data the
only stochastic element is the power-iteration start vector, so a seed sweep
measures how sensitive the certified step sizes are to it. Worth knowing — it is
how `npower` was chosen — but it is not a sample over noise realizations and
must not be reported as error bars.

**Error bars come from `run_2d_noise_stats.py`**, which draws independent
Poisson realizations of the transmission measurement:

```bash
python paper/experiments/run_2d_noise_stats.py --phantom all --reps 30
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
