"""Lesion-detectability test: does the certified two-channel method reveal an
inserted mass earlier than single-channel?

The VICTRE phantom p_-72912147 has no inserted lesion, so we insert a synthetic
spherical mass (signal-known-exactly) into the fibroglandular region, reconstruct
single vs certified two-channel (DBT, ramp-filtered fidelity), and compare a
zoomed ROI across iterations plus mass contrast-to-noise (CNR) vs iteration.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "experiments"))
import recon3d  # noqa: E402
from run_3d_victre import load_phantom, FIG, TAB  # noqa: E402


def insert_mass(phantom, dx_cm, mu, diam_cm):
    """Place a spherical mass at the centroid of the dense (fibroglandular)
    tissue. Returns (phantom_with_mass, center_zyx, radius_voxels)."""
    idx = np.argwhere(phantom > 0.34)            # glandular-ish
    cz, cy, cx = idx.mean(0).round().astype(int)
    r = max(2, int(round(diam_cm / 2 / dx_cm)))
    zz, yy, xx = np.ogrid[:phantom.shape[0], :phantom.shape[1], :phantom.shape[2]]
    sph = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    out = phantom.copy()
    out[sph] = mu
    print(f"inserted mass: center (z,y,x)=({cz},{cy},{cx}), r={r} vox "
          f"({2*r*dx_cm*10:.1f} mm diam), mu={mu}")
    return out, (cz, cy, cx), r


def cnr(roi, r):
    """Contrast-to-noise of the central disk (mass) vs a surrounding annulus."""
    n = roi.shape[0]
    c = n // 2
    yy, xx = np.ogrid[:n, :n]
    d = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    m = roi[d <= r]
    bg = roi[(d > 1.6 * r) & (d < 2.8 * r)]
    return float((m.mean() - bg.mean()) / (bg.std() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--itermax", type=int, default=150)
    ap.add_argument("--mu", type=float, default=0.55)
    ap.add_argument("--diam", type=float, default=1.2)   # cm
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames; required when sweeping "
                         "seeds, since runs otherwise overwrite each other")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    phantom, dx_cm = load_phantom(args.down)
    phantom, (cz, cy, cx), r = insert_mass(phantom, dx_cm, args.mu, args.diam)

    snaps = [s for s in (10, 20, 40, 70, 100, 150, 200) if s <= args.itermax]
    cfg = dict(itermax=args.itermax, npower=200, sigma_lo_scale=4.0,
               eps_hi_ratio=1.0, eps_lo_ratio=1.25, seed=args.seed)
    geom = dict(orbit="dbt", det_rows=480, det_cols=480, det_spacing=0.04,
                nviews=25, arc_deg=50.0, sod=65.0, odd=5.0)
    res = recon3d.reconstruct(phantom, dx_cm, cfg=cfg, geom=geom,
                              snapshot_iters=snaps)

    half = int(3.5 * r)
    def roi(vol):
        return vol[cz, cy - half:cy + half, cx - half:cx + half]
    gt_roi = roi(phantom)
    vmax = float(np.percentile(gt_roi, 99.5))

    # --- ROI iteration ladder: single (top) vs two-channel (bottom) ---
    n = len(snaps)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.7 * (n + 1), 3.7))
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    for row in (0, 1):
        axes[row, 0].imshow(gt_roi, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    axes[0, 0].set_title("ground truth", fontsize=9)
    axes[0, 0].set_ylabel("single", fontsize=10)
    axes[1, 0].set_ylabel("two-channel", fontsize=10)
    for j, it in enumerate(snaps):
        rs = roi(res["single"]["snaps"][it])
        rt = roi(res["two"]["snaps"][it])
        axes[0, j + 1].imshow(rs, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        axes[0, j + 1].set_title(f"iter {it}", fontsize=9)
        axes[1, j + 1].imshow(rt, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    fig.suptitle(f"Inserted mass ({2*r*dx_cm*10:.0f} mm) -- ROI vs iteration",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"recon3d_lesion_ladder{tag}.png", dpi=200, bbox_inches="tight")

    # --- CNR vs iteration ---
    cnr_s = [cnr(roi(res["single"]["snaps"][it]), r) for it in snaps]
    cnr_t = [cnr(roi(res["two"]["snaps"][it]), r) for it in snaps]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(snaps, cnr_s, "ks-", lw=1.8, label="single-channel")
    ax.plot(snaps, cnr_t, "bo-", lw=1.8, label="two-channel (certified)")
    ax.axhline(cnr(gt_roi, r), color="0.6", ls="--", lw=1, label="ground truth")
    ax.set_xlabel("iteration"); ax.set_ylabel("mass CNR")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"recon3d_lesion_cnr{tag}.png", dpi=200)

    lines = [f"# Inserted-mass detectability (VICTRE DBT, {2*r*dx_cm*10:.0f} mm "
             f"mass, mu={args.mu})\n", "| iter | single CNR | two-channel CNR |",
             "|---|---|---|"]
    print("\n=== mass CNR ===")
    for it, a, b in zip(snaps, cnr_s, cnr_t):
        lines.append(f"| {it} | {a:.2f} | {b:.2f} |")
        print(f"  iter {it:4d}: single CNR {a:.2f}  two CNR {b:.2f}")
    (TAB / f"recon3d_lesion{tag}.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote figs/tables. GT mass CNR = {cnr(gt_roi, r):.2f}")


if __name__ == "__main__":
    main()
