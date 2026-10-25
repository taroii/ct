"""The 3D figure: matched slices, an error row, and the convergence curves.

The existing `recon3d_victre_slices.png` shows ground truth / single / two side
by side, and at journal column width the two reconstructions are hard to tell
apart -- the reader is asked to take the RMSE table's word for it. What makes
the difference legible is the error image |recon - truth| on a shared scale,
which is where the low-frequency shading the method removes actually shows up.

The noise-statistics store keeps only RMSE traces, not volumes, so that row
cannot be built from it; this re-runs ONE realization with snapshots.

Config is taken from run_3d_noise_stats so the curves here are the same
experiment as the table, not a lookalike: same geometry, s, npower, tolerance
mode, dose rule and noise seed. As a guard, the RMSE traces are compared
against the stored run for that realization and the mismatch is printed -- if
it is not ~0, the config has drifted and the figure must not be used.

The slices and error images are also written to a small .npz so the figure can
be restyled WITHOUT another GPU run (the volumes are ~40 MB; the slices ~1 MB).
Restyling needs no ASTRA, so it works on a laptop -- hence the solver imports
live inside the run path, not at module scope.

Note the two RMSE scales this figure carries: the image panels are scored on
the central SLICE, the convergence curves on the whole VOLUME (what the paper's
tables report). They are different numbers for the same run and are labelled as
such, so neither can be mistaken for the other.

    python paper/experiments/fig_3d_victre.py                      # ~30 GPU-min
    python paper/experiments/fig_3d_victre.py --itermax 300        # ~15 GPU-min
    python paper/experiments/fig_3d_victre.py --from-npz PATH      # no GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIG = HERE / "figs"
TAB = HERE / "tables"
OUT = HERE / "results" / "fig_3d_victre"
STORE = HERE / "results" / "3d_noise_stats"
NOISE_SEED = 1000


def _itertag(itermax):
    """Keep runs at different budgets in different files.

    The npz and the png used to be named by seed and tolerance mode alone, so
    a longer rerun silently overwrote the shorter one it was supposed to be
    compared against -- and the slices it overwrote cost GPU hours to make.
    600 keeps the bare name so the published figure's path does not move.
    """
    return "" if itermax == 600 else f"_it{itermax}"


def stored_trace(noise_seed, itermax, eps_mode, s):
    """The run_3d_noise_stats traces for this realization, for the guard."""
    for jp in sorted(STORE.glob("*.json")):
        # macOS writes ._-prefixed AppleDouble sidecars next to real files when
        # a tree is copied to another filesystem; they are not manifests. More
        # generally, one unreadable file must not cost a completed GPU run --
        # the guard is a cross-check, not the product.
        if jp.name.startswith("._"):
            continue
        try:
            c = json.loads(jp.read_text())["config"]
        except Exception as e:
            print(f"  (skipping unreadable manifest {jp.name}: "
                  f"{type(e).__name__})")
            continue
        # itermax is deliberately not matched -- see below, the common prefix
        # of a longer run must reproduce the stored one exactly.
        if (c.get("noise_seed") == noise_seed
                and c.get("eps_mode") == eps_mode and c.get("s") == s):
            with np.load(jp.with_suffix(".npz")) as d:
                return (np.asarray(d["ierrs_single"], float),
                        np.asarray(d["ierrs_two"], float))
    return None


def make_panel(gt, rs, rt, si, tw, s, itermax, out_path):
    """Draw the figure from slices alone. No solver, no GPU, no ASTRA."""
    rmse = lambda a: float(np.sqrt(((a - gt) ** 2).mean()))   # noqa: E731
    vmax = float(np.percentile(gt, 99.5))
    emax = float(np.percentile(np.abs(np.concatenate([(rs - gt).ravel(),
                                                      (rt - gt).ravel()])), 99.5))

    fig = plt.figure(figsize=(11.2, 5.9))
    # Colorbars get their own grid columns. Attaching them to an image axis
    # steals width from the neighbouring cell, which is what clipped the
    # convergence panel's title and label in the first version of this figure.
    # wspace has to clear BOTH colorbar labels and the curve panel's y-label,
    # which sit on either side of the same gap.
    outer = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.5], wspace=0.34,
                             left=0.035, right=0.965, top=0.86, bottom=0.10)
    gi = outer[0, 0].subgridspec(2, 4, width_ratios=[1, 1, 1, 0.062],
                                 wspace=0.07, hspace=0.26)

    # Titles name the panel; the slice RMSE goes inside the panel as an
    # inset, the way the 2D galleries carry theirs, so that the titles stay
    # short and the two figures read the same way.
    for col, (img, title, sc) in enumerate((
            (gt, "Ground truth", None),
            (rs, "Single-channel", rmse(rs)),
            (rt, f"Two-channel ($s={s:g}$)", rmse(rt)))):
        ax = fig.add_subplot(gi[0, col])
        im = ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if sc is not None:
            ax.text(0.03, 0.97, f"{sc:.4f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color="w",
                    bbox=dict(boxstyle="square,pad=0.18", fc="k", ec="none",
                              alpha=0.55))
    cb = fig.colorbar(im, cax=fig.add_subplot(gi[0, 3]))
    cb.set_label(r"$\mu$ (cm$^{-1}$)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.add_subplot(gi[1, 0]).axis("off")

    for col, (img, title) in enumerate(
            ((np.abs(rs - gt), "Error, single-channel"),
             (np.abs(rt - gt), "Error, two-channel")), start=1):
        ax = fig.add_subplot(gi[1, col])
        im = ax.imshow(img, cmap="magma", vmin=0.0, vmax=emax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gi[1, 3]))
    cb.set_label("Absolute error", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(outer[0, 1])
    it = np.arange(1, len(si) + 1)
    ax.semilogx(it, si, "-", color="0.15", lw=1.8, label="single-channel")
    ax.semilogx(it, tw, "-", color="#1f77b4", lw=1.8,
                label=f"two-channel ($s={s:g}$)")
    imin = int(np.argmin(si))
    ax.plot(imin + 1, si[imin], "o", color="0.15", ms=5, mfc="white", mew=1.4)
    # Park the label in the empty band between the two curves at late
    # iterations, so neither the arrow nor the text crosses a trace.
    ax.annotate(f"single-channel minimum\n(iter {imin + 1}, then rises)",
                xy=(imin + 1, si[imin]),
                xytext=(len(si), 0.5 * (si[-1] + tw[-1])),
                ha="right", fontsize=7, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8,
                                shrinkB=4))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Image RMSE (whole volume)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Volume RMSE vs iteration", fontsize=9)

    fig.suptitle(f"3D DBT VICTRE reconstruction, iteration {itermax}",
                 y=0.965, fontsize=11)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"slice  RMSE single {rmse(rs):.5f} | two {rmse(rt):.5f} "
          f"({100 * (rmse(rs) - rmse(rt)) / rmse(rs):+.1f}%)")
    print(f"volume RMSE single {si[-1]:.5f} | two {tw[-1]:.5f} "
          f"({100 * (si[-1] - tw[-1]) / si[-1]:+.1f}%)   <- matches the table")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--itermax", type=int, default=600,
                    help="match the stored per-band run so the guard can compare")
    ap.add_argument("--npower", type=int, default=200)
    ap.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    ap.add_argument("--eps-mode", default="noise_per_band",
                    choices=["noise", "noise_per_band"])
    ap.add_argument("--from-npz", default=None,
                    help="restyle the figure from a saved .npz; no GPU needed")
    ap.add_argument("--s", type=float, default=4.0,
                    help="only used by --from-npz for a .npz saved before "
                         "s was recorded in it")
    args = ap.parse_args()

    for d in (FIG, TAB, OUT):
        d.mkdir(parents=True, exist_ok=True)

    if args.from_npz:
        with np.load(args.from_npz) as d:
            s = float(d["s"]) if "s" in d.files else float(args.s)
            it = int(d["itermax"])
            make_panel(d["gt"], d["single"], d["two"],
                       np.asarray(d["ierrs_single"], float),
                       np.asarray(d["ierrs_two"], float),
                       s, it,
                       FIG / f"recon3d_victre_panel_{args.eps_mode}"
                             f"{_itertag(it)}.png")
        return

    # Solver imports live here: restyling must work without ASTRA.
    import recon3d as r3                                        # noqa: E402
    from run_3d_victre import load_phantom, VICTRE_GEOM         # noqa: E402
    from run_3d_noise_stats import (auto_i0, SOLVER_SEED,       # noqa: E402
                                    HERO_S, EPS_CAL_DRAWS, EPS_CAL_K)

    phantom, dx_cm = load_phantom(args.down)
    i0 = auto_i0(phantom, dx_cm)
    cfg = dict(r3.DEFAULTS)
    cfg.update(itermax=args.itermax, sigma_lo_scale=HERO_S, seed=SOLVER_SEED,
               noise_seed=args.noise_seed, i0=i0, eps_mode=args.eps_mode,
               eps_factor=1.1, npower=args.npower,
               eps_cal_draws=EPS_CAL_DRAWS, eps_cal_k=EPS_CAL_K)
    print(f"3D VICTRE figure: i0={i0:.6g}, seed={args.noise_seed}, "
          f"{args.eps_mode}, itermax={args.itermax}")

    res = r3.reconstruct(phantom, dx_cm, cfg=cfg, geom=dict(VICTRE_GEOM),
                         snapshot_iters=[args.itermax])
    si = np.asarray(res["single"]["ierrs"], float)
    tw = np.asarray(res["two"]["ierrs"], float)

    mid = phantom.shape[0] // 2
    gt = np.asarray(phantom[mid], float)
    rs = np.asarray(res["single"]["recon"][mid], float)
    rt = np.asarray(res["two"]["recon"][mid], float)
    npz = (OUT / f"slices_seed{args.noise_seed}_{args.eps_mode}"
                 f"{_itertag(args.itermax)}.npz")
    np.savez_compressed(npz, gt=gt, single=rs, two=rt, ierrs_single=si,
                        ierrs_two=tw, i0=i0, itermax=args.itermax, s=HERO_S,
                        noise_seed=args.noise_seed, eps_mode=args.eps_mode)
    print(f"saved {npz} (restyle the figure from this without a GPU)")

    # Only now, with the GPU work safely on disk, run the cross-check.
    st = stored_trace(args.noise_seed, args.itermax, args.eps_mode, HERO_S)
    if st is None:
        print("GUARD: no stored run to compare against -- cannot confirm this "
              "matches the published table.")
    else:
        m = min(len(si), len(st[0]))
        d1, d2 = (float(np.abs(si[:m] - st[0][:m]).max()),
                  float(np.abs(tw[:m] - st[1][:m]).max()))
        print(f"GUARD: first {m} iterations vs stored run -- "
              f"single {d1:.2e}, two {d2:.2e}"
              + ("  OK" if max(d1, d2) <= 1e-6 else
                 "  *** MISMATCH: config has drifted, do not use this figure ***"))

    make_panel(gt, rs, rt, si, tw, HERO_S, args.itermax,
               FIG / f"recon3d_victre_panel_{args.eps_mode}"
                     f"{_itertag(args.itermax)}.png")


if __name__ == "__main__":
    main()
