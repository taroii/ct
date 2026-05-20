# CT-Meeting 2026 talk — rough draft

15-min oral + 5-min Q/A. Engineering audience (per Emil's framing: visuals
heavy, math light).

## Build

```bash
cd presentation
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex   # second pass for refs
```

Output: `main.pdf` (42 pages — 28 main + appendix + refs).

`graphicspath` covers `figs/` (paper-config presentation figures),
`../final_figures/` (legacy script outputs), and
`../final_figures/victre/` (3D CBCT slices).

## Structure

| Pages   | Section                                       |
| ------- | --------------------------------------------- |
| 1       | Title                                         |
| 2       | One-line takeaway                             |
| 3-4     | Motivation (LAR, geometry)                    |
| 5-6     | Setup (PDHG, single-channel)                  |
| 7-8     | The problem (visual + spectral)               |
| 9-10    | Idea (two-channel, algorithm)                 |
| 11-15   | 2D results (paper-config 256^2)               |
| 16      | Persistence across image sizes                |
| 17-19   | Analytic breast (intro + xy ladder + ROI ladder) |
| 20-21   | Analytic head (intro + central brain ROI ladder) |
| 22-23   | Analytic jaw  (intro + mouth/teeth ROI ladder)   |
| 24-25   | Shepp-Logan (intro + convergence: methods tie)|
| 26-27   | VICTRE (intro + glandular ROI iter ladder)    |
| 28      | 2D noise sweep (paper config 256^2)           |
| 29-30   | Conclusion + future work                      |
| 31      | Q&A                                           |
| 32-42   | Appendix (math, RMSE, refs)                   |

## Story

1. **Motivation:** LAR is unavoidable (dose, geometry); 50° arcs are typical.
2. **PDHG:** alternates a dual (sinogram) step and a primal (image) step;
   stability bounds the step sizes.
3. **The problem:** single-channel Hann1/2 weighting under-drives the
   low-frequency band — the very modes that are hardest to invert.
4. **The idea:** split the sinogram into hi/lo channels, each with its own
   dual variable and step size. Same constrained DTV+L1 objective.
5. **Result:** at clinical iteration counts, two-channel resolves
   recognizable soft-tissue structure several iterations earlier; clearest
   on the most under-determined grids (512^2). Same fixed point.

## Figures (paper config: c_hi=4, c_lo=8)

Regenerate with:
```bash
python scripts/presentation_iter_ladder.py                       # 2D 256^2 ladder
python scripts/_run_multires.py                                  # 2D 512/256/128 sweep
python scripts/presentation_persistence_plot.py                  # 1x3 conv. plot
python scripts/presentation_ct2_phantom_ladder.py --phantom breast
python scripts/presentation_ct2_phantom_ladder.py --phantom head
python scripts/presentation_ct2_phantom_ladder.py --phantom jaw
python scripts/presentation_shepp_logan_ladder.py                # 3D Shepp-Logan
python scripts/presentation_victre_ladder.py                     # 3D VICTRE breast
python scripts/presentation_noise_sweep.py                       # 2D Poisson sweep
python scripts/_phantom_gap_audit.py                             # print RMSE gaps
```

Cached snapshots in `cache/`: `iter_ladder_paper_256.pkl`,
`multiresolution_results.pkl`, `ct2_{breast,head,jaw}_recon.pkl`,
`shepp_logan_recon.pkl`, `victre_iter_ladder.pkl`,
`noise_sweep_256_paper.pkl`. Delete the relevant cache file (or pass
`--force`) to recompute.

Approximate wall times on this machine: 2D ladder + noise sweep ~1 min each,
analytic breast ~7 min, analytic head ~11 min (larger detector),
analytic jaw ~10 min, Shepp-Logan ~10 min, VICTRE ~23 min (largest volume).

## What's still rough

- **DBT geometry schematic** (slide 3): plain text box. Replace with a
  labelled source-arc / detector / breast-cross-section drawing.
- **Adjoint test** is loose for the wider-detector phantoms
  (analytic head rel=1.04, jaw rel=0.68, Shepp-Logan rel=0.92) but recons
  still produce informative results. Worth tightening before the talk.
- **Slide pacing**: 28 main slides for 15 min is tight ($\sim$32 s each).
  Consider dropping 1-2 of {head, jaw, Shepp-Logan} for live delivery.
- **Persistence figure** (slide 15): 23pt vbox overflow — cosmetic, the
  caption bullets sit lower than ideal.
- **Section divider slides** (Motivation/Results/etc.) are blank by
  metropolis default; can be tightened or removed.

## Parameter setup (post-pivot, 2026-05-19)

After team discussion we reverted to the paper's empirical step-size
heuristic:

- `sigma_hi = stepbalance / ||K||`, `tau = 1 / (stepbalance * ||K||)`,
  `sigma_lo = 4 * sigma_hi`, with `||K||` from an *unweighted* joint
  power iteration. This sits at the boundary of the CP theorem
  (`tau * sigma_hi * ||K||^2 = 1`) without strictly satisfying it once
  `sigma_lo > sigma_hi`.
- This is what reproduces the paper's $\sim$22% iter-500 RMSE
  improvement at $256^{2}$.
- In 3D (analytic phantoms, Shepp-Logan, VICTRE), the literal 2D heuristic diverges
  in two-channel mode. We inflate `||K||` by `sqrt(sigma_lo/sigma_hi)`
  in `scripts/victre_reconstruction.py` to recover stability while
  preserving the LF/HF ratio. Documented in the backup slide.
- The theoretical convergence-condition enforcement (commit `09afde4`)
  is parked under future work in the appendix; the constraint
  `tau * ||Sigma^(1/2) K||^2 < 1` will move back into the algorithm
  for the journal extension.

## Honest framing notes

- The visual story (early-iteration LF wobble in single-channel,
  resolved by two-channel) is intact and is the right lead for an
  engineering audience.
- The paper's abstract claim (improvement increases as the grid
  coarsens: 19/22/61% at 512/256/128) reproduces under this setup.
  It can be stated as observed.
- 128² shows the *biggest* two-channel gap (62%), not the smallest --
  the cleanly periodic 128² phantom seems to benefit most from the
  spectral split.
- 3D analytic phantoms (breast, head, jaw) all show a clear gap
  (iter-500: +16, +19, +17%; iter-200: +35, +14, +13%).
- 3D Shepp-Logan and VICTRE show **no gap** — two-channel is slightly
  worse (SL: -8% iter-500; VICTRE: -26% iter-500). Both have
  anatomically distributed structure rather than discrete inserts.
  Need to decide whether to keep them as "negative-control" slides or
  drop them.
