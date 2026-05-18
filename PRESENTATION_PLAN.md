# CT Meeting 2026 - Presentation & Experiments Plan

## Status (updated 2026-05-17)

The 2D regression in `final_figures/rmse_table.txt` has been diagnosed and the
talk's story re-grounded:

- Commit `09afde4` was a correct fix that brought two-channel into compliance
  with the CP convergence theorem; the paper's headline numbers came from a
  theorem-violating step-size scheme.
- Verified via 2000-iter sweeps at 256x256: at full convergence two-channel
  underperforms single-channel regardless of `eps_lo` ratio (0.05 - 1.25) or
  PoU filter cutoff (c=2 - 16). The asymptotic two-channel deficit is real.
- BUT: at iters 100-300 (the practical regime), two-channel produces visibly
  cleaner low-frequency reconstructions. Quantified by Gaussian-LP error maps:
  iter 200 LF RMSE 0.090 vs 0.129 (-30%); iter 100 LF RMSE 0.129 vs 0.174 (-26%).
  Crossover ~iter 500; thereafter single-channel keeps refining.

Talk re-framed around the early-iteration / trajectory advantage, with full-
convergence parity disclosed and the dyadic k>=3 manuscript positioned as the
principled path to closing the asymptotic gap.

Current figures (all in `final_figures/`, 256x256, PoU c=4, 2000 iter):

- `early_iter_recon_256.png` - reconstructions at iters 50/100/200/500, soft-
  tissue window. Two-channel rows visibly smoother in the LF band.
- `early_iter_lferror_256.png` - Gaussian sigma=8 px LP error maps. The
  centerpiece visual; bottom row is consistently fainter at iters 100-200.
- `late_iter_lferror_256.png` - iters 200/500/1000/2000 showing single
  catching up by ~500 and overtaking by 2000. Use for honest disclosure.
- `early_convergence_256.png` - first 500 iters; two-channel sits below
  single from iter ~150-350.
- `full_convergence_256.png` - all 2000 iters showing the long-term tail.
- `eps_lo_sweep_256.{png,txt}` - confirms eps_lo isn't the lever.
- `pou_cutoff_sweep_256.{png,txt}` - confirms cutoff isn't either.

Cached intermediate snapshots: `cache/early_iteration_snapshots_256.pkl`.

## Context

- Paper accepted to CT Meeting (conference in ~3 weeks, talk date TBD).
- Accepted for ORAL: 15 min presentation + 5 min Q/A, solo.
- Audience: engineers / CT practitioners, not optimization theorists.
- Emil's guidance (paper/emil_advice.md):
  - Prioritize motivation, visual results, and improvements over math.
  - Lots of images; separate panels for 10/20/50/100 iterations.
  - Pretend the audience are not experts: intro -> main result -> experimental results.
  - Spend a lot of time motivating the problem (LAR, PDHG/Chambolle-Pock).
  - Image RMSE is a weak metric here (angular range is tiny) - lean on visuals.
  - Core ask: find a setting where single-channel visibly shows low-frequency
    blur/wobble and two-channel resolves it. Zooming in is fine.
  - Get more empirical results, including discrete phantoms (Shepp-Logan), noise.
- Plan: math details go in appendix slides, referenced only if Q/A needs them.

---

## Part 1: Presentation

### Framing

- Tell it as a story: limited-angle scanning is clinically necessary (dose,
  scanner geometry) -> reconstruction is hard -> low frequencies converge
  slowly and wobble -> we split the data-fidelity term to fix exactly that.
- Lead every method idea with a picture, not an equation.
- The "single channel low-frequency blur vs two-channel fixed" image pair is
  the centerpiece. Everything builds toward it and refers back to it.
- Keep the optimization vocabulary light: "data-fidelity term", "step size",
  "frequency band". Defer singular values / stability proof to appendix.

### Slide structure (target ~15 min)

1. Title + one-line takeaway. (~0.5 min)
2. Motivation: why limited-angle? (~3-4 min)
   - Clinical drivers: dose reduction, scanner/space constraints, DBT geometry.
   - Concrete use cases: lung imaging, kidney stone imaging, breast.
   - Why 2D doesn't really make sense - it's fundamentally a 3D image.
3. The reconstruction setup. (~2 min)
   - PDHG / Chambolle-Pock at a high level: iterative, alternates image and
     data updates. One schematic, no proofs.
   - Sidky et al. single-channel approach: weighted (Hann^1/2) data fidelity.
4. The problem, shown visually. (~2-3 min)
   - Single-channel reconstruction as a function of iteration count
     (10/20/50/100/...): fine detail settles, low frequencies stay blurry/wobble.
   - This IS the motivation - let the images make the argument.
5. The idea: two-channel fidelity. (~2 min)
   - Split the sinogram residual into low-pass and high-pass bands.
   - Give the low-frequency band its own constraint + larger step size.
   - One clean schematic. Math lives in the appendix.
6. Results. (~4-5 min)
   - Side-by-side single vs two-channel reconstructions (+ zoomed ROI).
   - Iteration-series comparison: two-channel settles low frequencies faster.
   - Convergence curves (linear + log-log).
   - Limited-angle window sweep (50 / 90 / wider): improvement persists.
   - 3D VICTRE slices if ready - the big visual payoff.
7. Conclusion + future work. (~1 min)

### Appendix slides (for Q/A only)

- Constrained sparsity-regularized formulation (Eq. 3 / Eq. 4).
- Algorithm 1 (split-fidelity PDHG with He-Yuan relaxation).
- Singular-value / spectral-gap argument for why low frequencies stall.
- PDHG stability condition and per-channel step-size selection (power iteration).
- Full parameter tables (alpha, beta, cutoffs, sigma ratios, eps ratios).
- RMSE tables across resolutions.

### Formatting notes

- Big images, minimal text per slide, consistent grayscale window across
  every reconstruction comparison so panels are honest.
- Always label: resolution, iteration count, angular range, single vs two.
- Difference maps with a fixed symmetric color range.
- Have appendix slide numbers memorized so you can jump there live.

---

## Part 2: Experiments

### What's done

- 2D pipeline: scripts/compare_methods_multiresolution.py runs single vs
  two-channel at 512/256/128, with caching. Produces convergence_subplots.png,
  convergence_combined.png, rmse_table.txt.
- 2D plotting: scripts/plot_reconstruction_comparison.py produces the 2x2
  reconstruction images and 256 convergence variants.
- 2D figures currently in final_figures/ (convergence + reconstruction_2x2).
- 3D VICTRE pipeline: scripts/victre_reconstruction.py - loads/downsamples the
  VICTRE phantom, simulates a 25-view 50-deg cone-beam scan, runs single and
  two-channel CBCT recon, outputs final_figures/victre/ slices + convergence.
- ct-2 repo: analytic 3D phantom builder (phantom3d.py) + breast/head/jaw
  phantoms. NOTE: the demo scripts only do FORWARD PROJECTION + display - there
  is no reconstruction wired up in ct-2 yet.
- 2D digital breast phantom data is present (data/phantoms_from_paper/*.npy).

### Resolved / re-scoped issues

- **2D RMSE regression** - diagnosed (see Status above). The fix is to retell
  the story around the early-iteration trajectory rather than the asymptote.
- **3D VICTRE** - still shows no advantage in the prior fan-beam-on-2D-slices
  run; user will redo as proper CBCT from their PC where the `.raw` lives.

### What needs to be done

#### A. The motivation visual (DONE for 256, paper phantom)
- `final_figures/early_iter_recon_256.png` and `early_iter_lferror_256.png`
  deliver Emil's "main thing": at iters 100-200 the single-channel LF blur is
  visibly resolved earlier by two-channel (26-30% lower LF RMSE).
- Future polish: add a zoomed ROI panel on a homogeneous breast region for an
  even cleaner soft-tissue wobble comparison.

#### B. Other phantoms (next)
- Reuse the snapshot-capable `run_reconstruction_for_mfact` and rerun the
  early-iteration visualization on:
  - Shepp-Logan (not yet in the repo; need to add a phantom-builder).
  - The ct-2 analytic 3D phantoms (breast / head / jaw) once a recon pipeline
    is wired up there - currently projection-only.
  - 2D phantom variants from `data/phantoms_from_paper/` at indices other
    than `imagenumber=3` to confirm the LF-advantage isn't phantom-specific.
- Optional noise sweep: existing `addnoise=0` switch in
  `scripts/compare_methods_multiresolution.py`, wire up to test Poisson noise.

#### C. Dyadic k>=3 implementation (after other phantoms)
- Implement the dyadic shell construction from `paper/dyadic ct.tex` Sec III.A
  (eq `dyadic-filters`) and the geometric `sigma_i = 2^i sigma_0` schedule.
- Likely lives as a new function `run_reconstruction_dyadic(mfact, k, ...)` in
  `compare_methods_multiresolution.py` or as a parallel module.
- Real chance this closes the asymptotic gap; even if it doesn't, the talk
  benefits from showing the dyadic depth-vs-RMSE trend.

#### D. Limited-angle window sweep
- Run single vs two-channel at multiple arcs (e.g. 50 / 90 / 180-deg+fan).
- Show the low-frequency improvement persists as the window widens; frame any
  angular reduction as dose/risk benefit.
- Current code is fixed at 25 views / 50-deg arc - needs the geometry
  parameterized as a sweep.

#### E. 3D VICTRE experiments
- Get the .raw phantom onto the machine.
- Re-tune two-channel params so 3D shows a visible improvement.
- Deliver: single vs two-channel slice panels (xy/xz/yz), difference maps,
  convergence curve. This is the strongest possible visual aid - prioritize.

#### F. ct-2 analytic 3D phantoms
- Port a reconstruction pipeline into ct-2 (currently projection-only) so the
  breast/head/jaw analytic phantoms can be reconstructed single vs two-channel.
- Reuse the astra-based projector/recon from scripts/victre_reconstruction.py.
- These give clean, noise-free 3D structures to show improvement on beyond VICTRE.

#### G. Metrics
- Image RMSE is weak here (Emil) - keep the tables in the appendix but don't
  lead with them.
- The Gaussian-LP LF RMSE quantity introduced in
  `scripts/visualize_early_iterations.py` is the right metric for the talk's
  story; print it under each panel in figures.

### Figure deliverables checklist

- [x] 2D single-channel + two-channel iteration series at 256 (early_iter_recon_256.png)
- [x] 2D LF-error maps at early + late iters (early_iter_lferror_256.png, late_iter_lferror_256.png)
- [x] 2D convergence curve, early window + full (early_convergence_256.png, full_convergence_256.png)
- [x] 2D eps_lo and PoU cutoff sweeps confirming the trajectory-not-tolerance story
- [ ] 2D iteration series at 512 and 128 (rerun the visualization script with --mfact)
- [ ] 2D zoomed ROI comparison on a homogeneous breast region
- [ ] Limited-angle sweep comparison (50 / 90 / wider arcs)
- [ ] Other phantom families (Shepp-Logan, ct-2 analytic, alternate breast slices)
- [ ] Dyadic k>=3 results (3- and 4-channel) at 256
- [ ] 3D VICTRE slice panels (xy/xz/yz), single vs two-channel + difference (user's PC)
- [ ] 3D VICTRE convergence curve (user's PC)
- [ ] RMSE / LF-RMSE tables (appendix)

---

## Open questions / decisions

- Confirm the exact talk date within the conference window (affects schedule).
- Is the VICTRE .raw retrievable before the talk, or should ct-2 analytic
  phantoms be the primary 3D story instead?
- Acceptable to present updated 2D numbers if they differ from the paper, or
  must the figures exactly reproduce the published table?
- Which single phantom/setting becomes the "hero" motivation image?
