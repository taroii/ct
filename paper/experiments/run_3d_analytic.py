"""3D CBCT certified two-channel test on an analytic phantom.

Builds a Defrise-style analytic phantom (a stack of disks inside a cylinder plus
a few high-contrast spheres) -- the depth (z) direction carries strong low-
frequency content, which is exactly what limited-angle cone-beam degrades. Runs
single-channel vs certified two-channel (recon3d) and reports image-RMSE
convergence plus central slices.

Usage:
    python run_3d_analytic.py            # validation res (fast)
    python run_3d_analytic.py --full     # larger res
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

FIG = PAPER / "experiments" / "figs"
TAB = PAPER / "experiments" / "tables"


def defrise_phantom(nz, ny, nx):
    """Cylinder (LF background) + a z-stack of alternating dense/lucent disks
    (strong z low-frequency structure) + a few high-contrast spheres (HF)."""
    z = np.linspace(-1, 1, nz)[:, None, None]
    y = np.linspace(-1, 1, ny)[None, :, None]
    x = np.linspace(-1, 1, nx)[None, None, :]
    vol = np.zeros((nz, ny, nx), np.float32)
    cyl = (x ** 2 + y ** 2) <= 0.7 ** 2          # in-plane support
    vol[np.broadcast_to(cyl & (np.abs(z) <= 0.85), vol.shape)] = 0.4   # bg
    # stack of disks along z (each ~3 voxels thick), alternating +/- contrast
    ndisk = 7
    for i in range(ndisk):
        zc = -0.7 + 1.4 * i / (ndisk - 1)
        disk = (np.abs(z - zc) <= 0.05) & (x ** 2 + y ** 2 <= 0.55 ** 2)
        vol[np.broadcast_to(disk, vol.shape)] = 0.9 if i % 2 == 0 else 0.1
    # high-contrast spheres
    for (zc, yc, xc, r) in [(-0.3, 0.3, 0.0, 0.08), (0.3, -0.25, 0.2, 0.06),
                            (0.0, 0.0, -0.35, 0.05)]:
        sph = ((z - zc) ** 2 + (y - yc) ** 2 + (x - xc) ** 2) <= r ** 2
        vol[np.broadcast_to(sph, vol.shape)] = 1.0
    return np.ascontiguousarray(vol, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.full:
        shape = (128, 192, 192); det = (256, 256); itermax = 300; npower = 100
    else:
        shape = (80, 128, 128); det = (160, 160); itermax = 150; npower = 60
    dx_cm = 10.0 / shape[2]
    snaps = [10, 50, itermax]

    phantom = defrise_phantom(*shape)
    print(f"phantom {phantom.shape} range [{phantom.min():.2f},{phantom.max():.2f}]")

    cfg = dict(itermax=itermax, npower=npower, sigma_lo_scale=4.0,
               eps_hi_ratio=1.0, eps_lo_ratio=1.25, seed=args.seed)
    geom = dict(det_rows=det[0], det_cols=det[1], nviews=25, arc_deg=50.0)
    res = recon3d.reconstruct(phantom, dx_cm, cfg=cfg, geom=geom,
                              snapshot_iters=snaps)

    si, tw = res["single"]["ierrs"], res["two"]["ierrs"]
    report = [10, 50, 100, 150, 200, 300]
    report = [r for r in report if r <= itermax]
    lines = [f"# 3D CBCT analytic (Defrise) image RMSE vs iteration "
             f"(shape {shape}, {geom['nviews']} views / {geom['arc_deg']:.0f} deg)\n",
             "| iter | single | two-channel (certified) |", "|---|---|---|"]
    print("\n=== image RMSE ===")
    for r in report:
        red = (si[r-1] - tw[r-1]) / si[r-1] * 100
        lines.append(f"| {r} | {si[r-1]:.5f} | {tw[r-1]:.5f} ({red:+.1f}%) |")
        print(f"  iter {r:4d}: single {si[r-1]:.5f}  two {tw[r-1]:.5f}  ({red:+.1f}%)")
    (TAB / "recon3d_analytic.md").write_text("\n".join(lines) + "\n")

    # convergence plot
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    it = np.arange(1, len(si) + 1)
    ax.semilogy(it, si, "k-", lw=2, label="single-channel")
    ax.semilogy(it, tw, "b-", lw=1.8, label="two-channel (certified)")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "recon3d_analytic_convergence.png", dpi=200)

    # central coronal (x,z) slice comparison at final iter
    mid_y = shape[1] // 2
    gt = phantom[:, mid_y, :]
    rs = res["single"]["recon"][:, mid_y, :]
    rt = res["two"]["recon"][:, mid_y, :]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, im, t in zip(axes, [gt, rs, rt],
                         ["ground truth", "single", "two-channel"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1.0, origin="lower", aspect="auto")
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "recon3d_analytic_slices.png", dpi=200)
    print(f"\nWrote figs to {FIG}")


if __name__ == "__main__":
    main()
