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

Also writes a small .npz of the slices and error images so the figure can be
restyled later WITHOUT another GPU run (the volumes are ~40 MB; the saved
slices are ~1 MB).

    python paper/experiments/fig_3d_victre.py                 # ~30 GPU-min
    python paper/experiments/fig_3d_victre.py --itermax 300   # ~15 GPU-min
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recon3d as r3                                        # noqa: E402
from run_3d_victre import load_phantom, VICTRE_GEOM         # noqa: E402
from run_3d_noise_stats import (auto_i0, SOLVER_SEED, HERO_S,  # noqa: E402
                                EPS_CAL_DRAWS, EPS_CAL_K)

FIG = HERE / "figs"
TAB = HERE / "tables"
OUT = HERE / "results" / "fig_3d_victre"
STORE = HERE / "results" / "3d_noise_stats"
NOISE_SEED = 1000


def stored_trace(noise_seed, itermax, eps_mode):
    """The run_3d_noise_stats traces for this realization, for the guard."""
    for jp in sorted(STORE.glob("*.json")):
        # A single unreadable manifest must not cost a completed GPU run: the
        # guard is a cross-check, not the product.
        try:
            c = json.loads(jp.read_text())["config"]
        except Exception as e:
            print(f"  (skipping unreadable manifest {jp.name}: "
                  f"{type(e).__name__})")
            continue
        if (c.get("noise_seed") == noise_seed and c.get("itermax") == itermax
                and c.get("eps_mode") == eps_mode and c.get("s") == HERO_S):
            with np.load(jp.with_suffix(".npz")) as d:
                return (np.asarray(d["ierrs_single"], float),
                        np.asarray(d["ierrs_two"], float))
    return None


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
    args = ap.parse_args()

    for d in (FIG, TAB, OUT):
        d.mkdir(parents=True, exist_ok=True)

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
    np.savez_compressed(OUT / f"slices_seed{args.noise_seed}_{args.eps_mode}.npz",
                        gt=gt, single=rs, two=rt, ierrs_single=si, ierrs_two=tw,
                        i0=i0, itermax=args.itermax)
    print(f"saved {OUT / f'slices_seed{args.noise_seed}_{args.eps_mode}.npz'} "
          f"(restyle the figure from this without a GPU)")

    # Only now, with the GPU work safely on disk, run the cross-check.
    st = stored_trace(args.noise_seed, args.itermax, args.eps_mode)
    if st is None:
        print("GUARD: no stored run to compare against -- cannot confirm this "
              "matches the published table.")
    else:
        d1, d2 = (float(np.abs(si - st[0]).max()),
                  float(np.abs(tw - st[1]).max()))
        print(f"GUARD: max |diff| vs stored run -- single {d1:.2e}, two {d2:.2e}"
              + ("  OK" if max(d1, d2) <= 1e-6 else
                 "  *** MISMATCH: config has drifted, do not use this figure ***"))

    # ---- figure: matched window top row, shared error scale bottom row ----
    vmax = float(np.percentile(gt, 99.5))
    emax = float(np.percentile(np.abs(np.concatenate([(rs - gt).ravel(),
                                                      (rt - gt).ravel()])), 99.5))
    rmse = lambda a: float(np.sqrt(((a - gt) ** 2).mean()))   # noqa: E731

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], hspace=0.12,
                          wspace=0.08)
    for col, (img, title) in enumerate(((gt, "ground truth"),
                                        (rs, f"single  (RMSE {rmse(rs):.4f})"),
                                        (rt, f"two-channel $s={HERO_S:g}$  "
                                             f"(RMSE {rmse(rt):.4f})"))):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        if col == 2:
            fig.colorbar(im, ax=ax, fraction=0.046).set_label(
                r"$\mu$ (cm$^{-1}$)", fontsize=7)
    # error row: the point of the figure
    fig.add_subplot(gs[1, 0]).axis("off")
    for col, (img, title) in enumerate(((np.abs(rs - gt), "|single $-$ truth|"),
                                        (np.abs(rt - gt), "|two $-$ truth|")),
                                       start=1):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(img, cmap="magma", vmin=0.0, vmax=emax)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        if col == 2:
            fig.colorbar(im, ax=ax, fraction=0.046).set_label(
                "absolute error", fontsize=7)

    ax = fig.add_subplot(gs[:, 3])
    it = np.arange(1, len(si) + 1)
    ax.semilogx(it, si, "k-", lw=1.8, label="single-channel")
    ax.semilogx(it, tw, "-", color="#1f77b4", lw=1.8,
                label=f"two-channel $s={HERO_S:g}$")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("convergence (this realization)", fontsize=9)

    fig.suptitle(f"3D DBT VICTRE, central slice at iteration {args.itermax}; "
                 f"error images share one scale", fontsize=10)
    p = FIG / f"recon3d_victre_panel_{args.eps_mode}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")
    print(f"single RMSE {rmse(rs):.5f} | two RMSE {rmse(rt):.5f} "
          f"({100*(rmse(rs)-rmse(rt))/rmse(rs):+.1f}%)")


if __name__ == "__main__":
    main()
