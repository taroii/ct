# CT Meeting 2026 - Presentation & Experiments Plan

## Manual To-Do List:

Round 3 follow-ups (post-second-review):
- [x] Rename slide 11 to "2D fan-beam: single vs two-channel".
- [x] Strip redundant caption text (256x256, 25 views, 50 deg) across all figure slides where \geomnote already carries that information.
- [x] Drop slide 14 (3D analytic breast phantom intro three-view) entirely.
- [x] Revert head/jaw 100 deg figures to iter 300 instead of iter 500 (RMSE rises after iter ~100-200). Ladder now shows GT/iter-100/iter-300; convergence x-axis clipped to 0-300. Scripts: `_regen_ct2_ladders.py`, `_regen_ct2_convergence_arc100.py`. New caption numbers: head -16% (0.554 vs 0.463), jaw -12% (0.494 vs 0.436).
- [x] Summary slide: "2D paper breast" -> "2D breast phantom".
- [x] Move design-space material from live deck to appendix and expand to two slides: (a) "The per-band tolerance ratio" explaining what r is and what tuning does, (b) "Tolerance-ratio sweep across phantoms" with the redesigned single-panel figure and the cross-phantom story.
- [x] Future work rewritten: "Dyadic multi-channel fidelity" bullet now tied to the dyadic ct.tex manuscript (partition-of-unity bank, sigma_i = 2^i sigma_0, operator norm linear in k). LF-tolerance bullet rewritten for a first-time audience now that the design-space context is in the appendix. The "Realistic voxelized anatomy" bullet has been omitted -- see Future-work-deferred section below.
- [x] Clear out the rest of the appendix. Now only "Backup: per-band tolerance tuning" (two slides) + References.

Future-work bullets deferred (omitted from live deck for the practice run, may be re-added):
- Realistic voxelized anatomy / VICTRE breast. Original phrasing: "On voxelized glandular-structure phantoms the two-channel gain at 50 deg is front-loaded; sustaining it is ongoing." Cut because the result is honest-but-weak and the talk does not currently include a VICTRE figure. Re-add if Sidky asks for a discussion of how the method translates to anatomically realistic phantoms.

Pre-practice-run priorities (Sidky practice 2026-05-27):

Cuts (do first, biggest time-budget impact):
- [x] Drop VICTRE and Shepp-Logan entirely (live deck and appendix). Remove the slides, the figure references, and the future-work line that names VICTRE.
- [x] Consolidate 4 design-knob slides into 1 live "design space" slide. v2: replaced the spaghetti convergence plot with an iter-100-RMSE-reduction vs r line chart (2D vs 3D, paper config and 2D-best marked). Sweep detail kept in appendix.
- [x] 2D paper breast: keep iteration ladder + convergence only. Convergence figure regenerated linear-only (`scripts/_regen_2d_convergence_linear.py`).
- [x] 3D breast: keep ladder + convergence. Ladder bumped to include iter 500 (`scripts/_regen_ct2_ladders.py`).
- [x] Wider-arc: superseded by the per-phantom redesign. Live deck now has two slides ("Analytic jaw phantom" and "Analytic head phantom") each pairing a 3-column ladder (GT / iter 100 / iter 500) with the convergence plot for that phantom at 100°. Multi-arc-convergence slide removed; head/jaw 50° and 75° material stays in appendix.
- [x] Clean up appendix after the above cuts.

Edits:
- [x] Em-dash sweep across the whole deck.
- [x] Fill the placeholder DBT geometry schematic on the "Why limited-angle?" slide. v2 TikZ: bigger compressed-breast half-ellipse between detector and a top paddle, two extreme rays defining the 50° wedge, three source dots on a dashed arc.
- [x] Rework the "Reconstruction with PDHG (Chambolle-Pock)" slide. Current bullets are too dense for ~30 s of delivery. Either compress to one sentence + reference to backup, or split the operator-norm line out entirely.
- [x] Rename "Design knob 1" to "Band-cutoff robustness" (or similar) and remove every "PoU" mention from the live deck.
- [x] Summary slide: switch the -45% / -25% numbers to the iter-100 paper-config numbers from the 2D and 3D breast results that survive the cut, so the live numbers and the summary agree.
- [x] Future-work slide: drop the VICTRE-named bullet; reword the "realistic voxelized anatomy" point without naming VICTRE.

Speaker-side / script:
- [ ] Decide whether to retitle away from "Limited-Angle DBT" toward "Limited-Angle Tomography" (or similar). The wider-arc head/jaw slides land better under a non-DBT-specific title; Emil also suggested de-emphasising DBT.
- [ ] Speaker notes: explain empirical phenomena out loud (why iter 5 is uniformly blurry, why iter 20 is the inflection). Otherwise drop iters 5 and 20 from the ladder and start at iter 50.
- [ ] Speaker notes: pre-empt the 3D semi-convergence question. Say out loud that both methods rise after iter ~100 and the comparison is at a stopping point, not asymptotic.
- [ ] Speaker notes for the design-space slide: plan to say "I condensed the design-knob material because of the time cap and because the tolerance optimum doesn't transfer between 2D and 3D; happy to expand if you'd prefer."

Experiments still open:
- [ ] Test wider angles on the analytic phantoms ([50, 100] degree range). If we do better at higher angles, we can reduce the DBT framing and write it as a more general LAR setting. 

---

# old

## Reconciliation (2026-05-21, desktop) — two threads aligned

Two threads have been running in parallel:
- **Laptop thread** (this file's recent notes): scaffolded the band /
  eps sweep direction and ran H1, H2, H6 on 2D paper-breast.
- **Desktop thread** (not previously written here): rebuilt
  `presentation/main.tex` against an in-person review with Emil; ran
  the 75- and 100-deg analytic-phantom reconstructions.

### Desktop changes since this file was last updated

Slide-by-slide rebuild of `presentation/main.tex` (now 65 pages):
- Motivation reworked: removed the takeaway slide; added a "What is
  limited-angle reconstruction?" slide (the 180-deg+fan sampling fact
  now lives there, not under "Space"); old slide 4 became "Two
  reconstruction settings: 2D and 3D".
- Reconstruction setup reworked: removed "in one slide", "Step sizes
  are not free", and the strict stability inequality that the talk
  doesn't follow; added "Why not gradient descent?"; intuitive
  primal/dual alternation. Footnote citation on the single-channel slide.
- "Wobble" removed from every slide; AI-flavoured phrasings replaced
  with plain academic sentences; "reproduces the paper's abstract" line
  removed.
- Two-channel iteration diagram redrawn in tikz (clean loop with
  feedback arrow).
- Difference plots converted to **magnitude** (black 0, white largest),
  full slice for the 3D phantoms; the 2D "Where is the error?" slide
  was updated to match. Colorbar added with labels.
- Display windows widened to stop recons saturating white: 2D 1.0 ->
  1.6; breast vmax 0.6; head vmax 2.6; jaw vmax 3.0.
- All figure suptitles dropped (slide title carries the name); every
  figure slide gets a small italic generation note via a new
  `\geomnote` macro: "modality, 25 views, 50 deg arc, grid".
- Head/jaw iteration-ladder slides switched from ROI crop to full
  slice; reconstruction-error and convergence slides added for each
  of breast/head/jaw.
- Added `--arc` argument to `presentation_ct2_phantom_ladder.py`; ran
  75-deg and 100-deg reconstructions for breast/head/jaw. 18 wide-arc
  slides added under a new section "Wider-arc reconstructions
  (exploratory)". Notable: at 100 deg the breast problem becomes
  well-conditioned and **single-channel wins** (RMSE 0.010 vs
  two-channel 0.021) -- crossover into the well-conditioned regime.
- Shepp-Logan and VICTRE moved out of the main flow into the appendix
  with a 50-deg paper-vs-tuned summary table; conclusion/future work
  rewritten accordingly.
- New helper scripts under `scripts/_*`: `_regen_ct2_figs.py`,
  `_regen_convergence_figs.py`, `_error_style_variants.py`,
  `_sweep_ct2_arcs.py`, `_inspect_arc_gaps.py`, `_inspect_display_ranges.py`.

### Reconciliation with Emil's 2026-05-22 scope cut

Emil asked for band-selection and data-error results, focused on **2D
and 3D breast only**. None of the 2D/3D band-selection or eps sweep
results are in the deck yet; the deck still contains head, jaw,
Shepp-Logan (appendix), VICTRE (appendix), and 75/100-deg head/jaw.
Forward plan:

1. **Slides for H1 (2D band sweep).** Figures already at
   `final_figures/H1_cutoff_{recon,error,lferror,convergence}_256.png`.
   Owner: either thread.
2. **Slides for H2 (2D eps sweep, r=15 headline).** Figures at
   `final_figures/H2_eps_*`. **Headline:** -25% iter-100 RMSE vs paper
   r=1.25, -45% vs single-channel. Owner: either thread.
3. **Run H3 (3D ct-2 breast cutoff sweep).**
   `scripts/sweep_cutoff_visual_ct2breast.py` -- CUDA required ->
   **desktop owns**.
4. **Run H4 (3D ct-2 breast eps sweep).**
   `scripts/sweep_eps_visual_ct2breast.py` -- CUDA required ->
   **desktop owns**.
5. **Trim head/jaw from main flow** -- move to appendix.
   Owner: either thread.
6. **Decide on wide-arc 75/100 slides.** Per scope-cut, head/jaw
   75/100 also move to appendix; breast 75/100 stays. The breast-100
   "single wins" result is honest and worth keeping (shows where the
   method's limited-angle benefit ends).
7. **If H4 corroborates r=15 on 3D ct-2 breast**, regenerate the 3D
   breast main-flow figures at r=15 so the deck shows the tuned
   configuration rather than the paper default. Decide after H4 lands.

### H3 + H4 findings (2026-05-21, desktop -- CUDA box)

H3 -- 3D ct-2 breast, c_hi=4 fixed, eps_lo/eps_hi=1.25, c_lo in {4, 6, 8,
12, 16}, itermax=500, 50-deg arc:

- All five c_lo values give image RMSE within +-0.005 at every iteration
  count. PoU (c_lo=4) and non-PoU (c_lo=8) indistinguishable.
  iter-100 RMSE 0.0504-0.0535 across the sweep; iter-500 0.0956-0.0974.
- Confirms the 2D H1 result on 3D: **band selection is robust on both**.
  Designer does not need to fine-tune the filter shape.
- Figures: `presentation/figs/ct2_breast_H3_cutoff_{recon,error,convergence}.png`;
  summary in `ct2_breast_H3_cutoff_summary.txt`; cache at
  `cache/ct2_breast_H3_cutoff_sweep.pkl`.

H4 -- 3D ct-2 breast, c_hi=4, c_lo=8 fixed, r=eps_lo/eps_hi in {0.25, 0.5,
1.0, 1.25, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0}:

- **r=15 does NOT transfer to 3D.** On 3D breast the paper's r=1.25 is at
  or near the iter-100+ optimum; tightening to r=0.25 wins iter-200/500
  by ~3%, loosening to r >= 5 progressively hurts iter 100+.
- Key numbers vs single (0.0773 at iter 100):
    r=0.25  | 0.0508 | iter 100 (-34% vs single, best iter-500=0.0931)
    r=1.25  | 0.0508 | iter 100 (paper config; iter-500=0.0960)
    r=15    | 0.0525 | iter 100 (worse than paper at iter 100+;
                                  only wins iter-50 by ~6%)
    r=30    | 0.0557 | iter 100 (loosest tested; iter-500=0.1221)
- The 3D semi-convergence rise dominates regardless of r: every two-channel
  curve has a minimum near iter 60-80 and rises thereafter. Loose r
  produces a slightly lower minimum that rebounds faster; tight r holds
  the iter-500 RMSE slightly lower.
- Figures: `presentation/figs/ct2_breast_H4_eps_{recon,error,convergence}.png`;
  summary in `ct2_breast_H4_eps_summary.txt`; cache at
  `cache/ct2_breast_H4_eps_sweep.pkl`.

**Implication for the talk.** The 2D and 3D phantoms have different
sensitivity to r. Honest design-space picture:

- Band design (c_lo, PoU vs not): **robust on both 2D and 3D.**
- LF tolerance (r = eps_lo/eps_hi): **tunable on 2D paper-breast**
  (r=15, -25% iter-100 vs paper), **near-optimal at the paper config
  on 3D ct-2 breast** (no big gain available).
- Two-channel vs single (at clinical iter counts): solid on both
  phantoms (~30-45% RMSE reduction at iter 100).

So the talk does NOT regenerate the 3D breast main-flow figures at r=15
(item 7 above is dropped). Keep the paper config on the 3D side, and
present H4 as "the design landscape on 3D breast is forgiving in the
neighborhood of the paper config."

### Deck restructure landed (2026-05-21, desktop, post-H3/H4)

`presentation/main.tex` reorganised to Emil's scope. Now 75 pages total:

- **Main flow (~34 pages incl section pages):**
  Motivation -> Setup -> Problem -> Idea -> Results: 2D paper breast
  (iter ladder / soft-tissue ROI / "Where is the error?" / convergence /
  persistence) -> **2D design knobs (NEW: H1 band, H2 eps conv, H2 eps
  headline recon)** -> 3D analytic breast (intro / mid-axial ladder /
  ROI / **reconstruction error NEW** / convergence) -> **3D design
  knobs (NEW: H3 band, H4 eps)** -> Conclusion (Summary updated, Future
  work updated).
- **Appendix (~38 pages):**
  H6 (2D eps_hi supplementary, NEW) -> Backup: 3D analytic head and
  jaw (8 slides MOVED from main) -> Backup: wider-arc reconstructions
  (18 slides MOVED) -> Backup: Shepp-Logan and VICTRE -> Backup:
  method details (objective / Algorithm 1 / step-size / full-resolution
  RMSE) -> References.

The H2 recon figure (`H2_eps_recon_256.png`) is too tall to render
legibly on a slide (16+ ratio rows). New focused figure
`final_figures/H2_eps_headline_recon_256.png` (3 rows: single / paper
r=1.25 / tuned r=15) generated by `scripts/_d3_headline_recon.py`;
this is the figure D3 references. Tuned-r=15 recon snapshots cached at
`cache/iter_ladder_tuned_r15_256.pkl`.

Summary slide now reads:
- LAR DBT bottleneck.
- Two-channel splits and accelerates LF.
- Confirmed on 2D paper breast and 3D analytic breast at 50 deg.
- Design space: band selection robust on both; LF tolerance tunable on
  2D (r=15, -25% iter-100 vs paper, -45% vs single), near-optimal at
  the paper r=1.25 on 3D breast.

Future work now: multi-band fidelity, theorem-compliant step sizes,
**LF-tolerance scaling (NEW: 2D vs 3D r-optimum gap)**, VICTRE / realistic
voxelized anatomy.

### Remaining work
- User wants to revisit head/jaw to see if any design-space tuning can
  push them into a stronger "favorable" zone -- to be tackled after the
  base presentation is signed off.

### Handoff convention
- Laptop: 2D experiment runs (CPU ok), slide editing, plan upkeep.
- Desktop: anything CUDA (H3, H4, 3D recons), slide editing, plan upkeep.
- Both threads edit `presentation/main.tex` and update this file on
  material changes.
- 2D caches: `cache/iter_ladder_paper_256.pkl`. 3D ct-2:
  `cache/ct2_<name>_recon[_arc<N>].pkl`.

### What the sections below now describe
The "Slide structure (target ~15 min)" in Part 1 is older than the
current deck (the deck is more developed). Treat it as motivation
for the current `presentation/main.tex` rather than a literal
description. The H1/H2/H6 findings in Section H **are** current.

---

## Status (updated 2026-05-22) — Emil's second email

After seeing the slides, Emil wrote:

> "After seeing the slides, it occurred to me that it would be good to see
> more results centered on the impact of band selection and data-error
> constraint values. Maybe have more results along those lines and focus
> only on the 2d and 3d breast phantoms."

Reading this as two technical asks plus a scope-narrow:

1. **Band selection** — show how the choice of filter design changes the
   reconstruction. Knobs in our two-channel code: `cutoffparm` (c_hi),
   `cutoffparm_lo` (c_lo), and whether they form a partition of unity. The
   current talk shows one specific design (c_hi=4 / c_lo=8, non-PoU);
   the audience can't see the design space.
2. **Data-error constraints** — show how the per-band tolerances
   `eps_hi`, `eps_lo` change the reconstruction. This is the lever the
   manuscript already calls out as the one that shifts the fixed point
   (sigma_lo only changes the trajectory). Right now eps_lo/eps_hi = 1.25
   is the only data point in the deck.
3. **Scope** — focus on **2D paper breast phantom + 3D analytic breast** only.
   Drop head/jaw/Shepp-Logan from the main flow (keep available in figures
   directory; can stay as backup slides).

Project pivot: **moved away from dyadic k>=3.** Convergence guarantees for
the dyadic case were never proven in the abstract, so the talk and journal
extension stay inside the two-channel formulation. References to k>=3 in
older sections of this plan are marked `[DROPPED]` rather than deleted, so
the option is still visible to the user.

What we already have that's reusable:
- `scripts/sweep_eps_lo.py` produces RMSE-curves + table over an eps_lo
  ratio sweep at 2D; needs an extension that *renders the actual
  reconstructions* at a fixed iteration count across the sweep.
- `scripts/sweep_pou_cutoff.py` does the same for PoU matched cutoffs c;
  same extension needed.
- `scripts/visualize_early_iterations.py` already knows how to capture
  snapshots from `run_reconstruction_for_mfact` and lay them out as a
  ladder — its `make_recon_grid` / `make_error_grid` /
  `make_lf_error_grid` helpers are directly reusable.
- 3D analytic breast pipeline:
  `scripts/presentation_ct2_breast_ladder.py` already reconstructs and
  saves snapshots for the 50/75/100-deg arc figures in the deck. Needs a
  band-sweep / eps-sweep driver that varies cutoff or eps_lo and rerenders.

What's needed (mapped to deliverables — see new Section H below).

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
principled path to closing the asymptotic gap. **[2026-05-22 UPDATE: dyadic
extension has been dropped from the talk and journal extension scope, since
convergence guarantees were never proven in the abstract; keep the manuscript
as background only.]**

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

#### C. Dyadic k>=3 implementation (after other phantoms) **[DROPPED 2026-05-22]**
- Implement the dyadic shell construction from `paper/dyadic ct.tex` Sec III.A
  (eq `dyadic-filters`) and the geometric `sigma_i = 2^i sigma_0` schedule.
- Likely lives as a new function `run_reconstruction_dyadic(mfact, k, ...)` in
  `compare_methods_multiresolution.py` or as a parallel module.
- Real chance this closes the asymptotic gap; even if it doesn't, the talk
  benefits from showing the dyadic depth-vs-RMSE trend.
- **Why dropped:** convergence guarantees were never proven for k>=3 in the
  abstract / accepted paper; presenting dyadic results would overclaim.
  Kept here for visibility; do not pursue for the talk.

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

#### H. Band selection and data-error constraint results (2026-05-22, per Emil)

Emil's follow-up email after seeing the slides:

> "it would be good to see more results centered on the impact of band
> selection and data-error constraint values. Maybe have more results
> along those lines and focus only on the 2d and 3d breast phantoms."

Two design knobs to make visible, on two phantoms.

**Knob 1 — Band selection.** The choice of the high/low channel filter design:
- Cutoff sweep: `c_hi`, `c_lo`. Two views: matched-cutoff PoU (c_hi = c_lo)
  with c in {2, 4, 8, 16}; and paper-style non-PoU where c_lo > c_hi (e.g.
  c_hi=4 fixed, c_lo in {6, 8, 12, 16}).
- PoU vs non-PoU at fixed scale (e.g. c_hi=4, c_lo=8 paper config vs
  c_hi=c_lo=4 matched PoU). One slide showing visually what the PoU
  construction buys (or doesn't).

**Knob 2 — Data-error constraints.** The `eps_hi`, `eps_lo` per-band radii:
- eps_lo/eps_hi sweep at fixed band design (default c_hi=4, c_lo=8). Ratios
  in {0.25, 0.5, 1.0, 1.25, 2.0} should span "tighten LF" to "loosen LF"
  relative to the paper config.
- (Optional) eps_hi sweep at fixed eps_lo to show symmetry — likely just for
  appendix.

**Phantoms.**
1. 2D paper digital breast (`data/phantoms_from_paper/...`, imagenumber=3),
   single representative resolution (256 most likely — fastest, already the
   reference in `presentation_iter_ladder.py`).
2. 3D ct-2 analytic breast at 50-deg arc (matches the current main-flow 3D
   slide). Pipeline already lives in `scripts/presentation_ct2_breast_ladder.py`.

**Presentation style.** Crucial difference from the existing sweep figures
(which are just RMSE-vs-iter curves): for each knob value, *render the
reconstruction itself* at one or two fixed iteration counts (e.g. iter 100
and iter 500) as a row in a grid. The audience needs to see the visual
consequence of each knob setting, not just an RMSE number. Mirror this with
a difference-map row underneath, same color scale across the row.

**Concrete new scripts to add:**
1. `scripts/sweep_cutoff_visual_2d.py` — vary the cutoff (PoU and/or
   non-PoU) on 2D paper-breast, run 256 to N=500, render a recon-row +
   error-row grid plus the RMSE curve. Reuse helpers from
   `visualize_early_iterations.py`.
2. `scripts/sweep_eps_visual_2d.py` — same shape for the eps_lo/eps_hi ratio
   sweep.
3. `scripts/sweep_cutoff_visual_ct2breast.py` — equivalent on the 3D ct-2
   breast at 50-deg, mid-axial slice rendered as the row image. Reuse the
   reconstruction loop from `presentation_ct2_breast_ladder.py`.
4. `scripts/sweep_eps_visual_ct2breast.py` — eps version on 3D ct-2 breast.

**Decisions outstanding (ask user before running):**
- Which iteration counts to render per knob value? Default proposal:
  iter 100 (early-advantage regime) + iter 500 (where the deck currently
  reports headline numbers).
- Cutoff sweep extent: spend more effort on PoU matched sweep, or on
  non-PoU varying c_lo? Or both?
- For the 3D ct-2 sweeps: stick to the existing 50-deg arc or also include
  the 75-deg point where the gap is more dramatic?
- VICTRE: Emil said "breast phantoms" plural — should we read VICTRE in
  *or* read it as ct-2 analytic breast only? VICTRE currently lives in
  backup; if it's meant to be in the main story, the sweep work doubles.

**Findings (2026-05-22, from H1 + H2 runs at 256x256, itermax=500):**

H1 -- band-selection (c_lo sweep, c_hi=4 fixed, eps_lo/eps_hi=1.25):

- The two-channel image RMSE is **nearly flat** across c_lo in {4, 6, 8, 12, 16}
  at every iteration count. Numbers within +-0.002 RMSE of each other.
  This includes the PoU case (c_lo=4 with c_hi=4); PoU is theoretically tidier
  but visually indistinguishable from the paper's non-PoU c_lo=8.
- LF RMSE @ iter 200 reaches its minimum at **c_lo=16** (0.0598) and degrades
  past that (c_lo=24 -> 0.0627, c_lo=32 -> 0.0664). Beyond c_lo=16 the early-
  iter image RMSE also degrades; only the iter-500 asymptote benefits.
- **Story for the talk:** the design is robust over a wide c_lo plateau; the
  paper's c_lo=8 sits in the middle of it.

H2 -- data-error constraint (eps_lo/eps_hi sweep, c_hi=4, c_lo=8 fixed):

- Image RMSE has a **smooth unimodal interior optimum at r in [15, 20]**.
  Improves through r=10..15, plateaus 15..20, degrades 20..50, then
  collapses at r=10^6 (the "no LF constraint" limit gives RMSE ~ 0.55 --
  the method is unusable without an LF constraint).
- Best-found point: r=15 wins iter-100/200/500 and LF RMSE; r=20 wins
  iter-50 by a hair. Going with r=15 as the headline.
- Headline at r=15 vs paper config (r=1.25):
    iter  50: 0.1430 vs 0.1855 (-23%)
    iter 100: 0.1215 vs 0.1615 (-25%)
    iter 200: 0.1035 vs 0.1192 (-13%)
    iter 500: 0.0732 vs 0.0768  (-5%)
    LF RMSE @ iter 200: 0.0518 vs 0.0621 (-17%)
- Headline at r=15 vs single-channel @ iter 100: 0.1215 vs 0.2202 (-45%).
- **Interpretation:** the LF channel needs to be *loose but not vacuous*.
  The paper's r=1.25 is too tight; r >> 20 strands the LF correction
  entirely. The clinically useful window is r in [10, 20].
- Smooth curve over 16 r values from 0.25 to 30 (plus 50, 10^6) available
  in `final_figures/H2_eps_convergence_256.png`; flat plateau in the
  optimum window plus the catastrophic r=10^6 collapse make this an
  excellent "what's the right slack?" slide.

H6 -- eps_hi scale sweep (at r=10 fixed, c_hi=4, c_lo=8 fixed):

- Also has an interior optimum, at eps_hi/eps ~ 1..2. Tightening to 0.5
  hurts early iters; loosening to 4 hurts uniformly.
- Headline at the best-tested point (eps_hi/eps=2) vs paper (eps_hi/eps=1):
    iter  50: 0.1413 vs 0.1506 (-6%)
    iter 100: 0.1222 vs 0.1252 (-2%)
- Cross-check caveat: H6 (eps_hi/eps=2, r=10) and H2 (r=20, eps_hi/eps=1)
  are the *same parameter point* but produce slightly different curves
  due to the unseeded `np.random.randn` power-iteration init. The
  difference is 1-3% in RMSE; trends are unaffected. Seedable in
  compare_methods_multiresolution.py if bit-identity is needed.

**Combined picture (for the talk's "designer view"):**

1. **Band selection (c_lo)** -- broad plateau c_lo in [4, 16]; PoU and
   non-PoU indistinguishable. "Method is robust to filter shape."
2. **LF/HF balance (r = eps_lo/eps_hi)** -- interior optimum at r ~ 20;
   paper's r=1.25 leaves ~25% RMSE on the table at clinical iter counts.
3. **Overall scale (eps_hi)** -- paper's eps_hi=cm.eps is near-optimal;
   small benefit from eps_hi=2*cm.eps but not robust at iter 500.
4. **At the best 2D config** (r=15, eps_hi=cm.eps): iter-100 RMSE 0.1215
   vs paper config 0.1615 (-25%) vs single-channel 0.2202 (-45%).

**Scope cut for the talk (per Emil "focus only on 2D + 3D breast"):**
- Head and jaw 3D analytic phantoms move from main flow to backup.
- Shepp-Logan stays in backup (already is).
- Wider-arc (75/100-deg) results stay available as backup since they
  buttress the LAR framing, but step back from the main flow if needed.
- These cuts are noted here but **not yet applied to main.tex** — pending
  user direction.

### Figure deliverables checklist

- [x] 2D single-channel + two-channel iteration series at 256 (early_iter_recon_256.png)
- [x] 2D LF-error maps at early + late iters (early_iter_lferror_256.png, late_iter_lferror_256.png)
- [x] 2D convergence curve, early window + full (early_convergence_256.png, full_convergence_256.png)
- [x] 2D eps_lo and PoU cutoff sweeps confirming the trajectory-not-tolerance story
- [ ] 2D iteration series at 512 and 128 (rerun the visualization script with --mfact)
- [ ] 2D zoomed ROI comparison on a homogeneous breast region
- [ ] Limited-angle sweep comparison (50 / 90 / wider arcs)
- [ ] Other phantom families (Shepp-Logan, ct-2 analytic, alternate breast slices)
- [~] Dyadic k>=3 results (3- and 4-channel) at 256 — **DROPPED 2026-05-22**
- [ ] 3D VICTRE slice panels (xy/xz/yz), single vs two-channel + difference (user's PC)
- [ ] 3D VICTRE convergence curve (user's PC)
- [ ] RMSE / LF-RMSE tables (appendix)
- [scaffolded] **H1. 2D paper-breast cutoff sweep -- `scripts/sweep_cutoff_visual_2d.py` (non-PoU, c_hi=4, c_lo in {6,8,12,16}, iters 50/100/200/500). Not yet run.**
- [scaffolded] **H2. 2D paper-breast eps_lo/eps_hi sweep -- `scripts/sweep_eps_visual_2d.py` (ratios {0.25, 0.5, 1.0, 1.25, 2.0}). Not yet run.**
- [scaffolded] **H3. 3D ct-2 breast (50-deg) cutoff sweep -- `scripts/sweep_cutoff_visual_ct2breast.py`. Needs astra-toolbox **with CUDA** (the 3D module only exposes `*_gpu` calls). Confirmed unrunnable on this macOS box even after astra 2.4.1 install via conda-forge.**
- [scaffolded] **H4. 3D ct-2 breast (50-deg) eps sweep -- `scripts/sweep_eps_visual_ct2breast.py`. Same CUDA constraint as H3.**
- [observed in H1] **H5. PoU vs non-PoU at fixed scale** -- H1 with c_lo=4 (PoU) and c_lo=8 (paper non-PoU) shows the two are numerically indistinguishable (within +-0.002 RMSE everywhere). No further work needed for the talk.
- [ ] **H6. 2D paper-breast eps_hi sweep at r=eps_lo/eps_hi=10 fixed -- `scripts/sweep_eps_hi_visual_2d.py` (eps_hi/eps in {0.5, 1, 2, 4}). Probes the overall scale.**

---

## Open questions / decisions

- Confirm the exact talk date within the conference window (affects schedule).
- Is the VICTRE .raw retrievable before the talk, or should ct-2 analytic
  phantoms be the primary 3D story instead?
- Acceptable to present updated 2D numbers if they differ from the paper, or
  must the figures exactly reproduce the published table?
- Which single phantom/setting becomes the "hero" motivation image?
- (2026-05-22) For Section H: which iteration counts per knob value (default
  100 + 500), how wide the cutoff and eps sweeps should be, whether VICTRE
  is in scope for the sweeps, and whether to include the 75-deg arc point
  on 3D ct-2 breast.
- (2026-05-22) Once Section H lands: are head, jaw, Shepp-Logan, and wider
  arcs trimmed from the main flow of `presentation/main.tex` or kept?
