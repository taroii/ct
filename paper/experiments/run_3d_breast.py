"""3D DBT certified two-channel test on the analytic ct-2 breast phantom.

DBT cone-beam geometry (source arcs in y-z, flat detector below), with the band
split on the 2D detector (u,v) -- matched to the limited-angle missing direction.
Runs single-channel vs certified two-channel (recon3d) and reports image-RMSE
convergence plus central slices.
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
sys.path.insert(0, str(PAPER / "phantoms3d"))
import recon3d  # noqa: E402
from phantom3d import image3D  # noqa: E402
from breast_phantom_demo import build_breast_phantom  # noqa: E402

FIG = PAPER / "experiments" / "figs"
TAB = PAPER / "experiments" / "tables"
DX_CM = 0.15
CLIP = 0.6           # cap attenuation (soft tissue range) before peak-scaling


def breast_volume():
    """Embed the breast phantom on a cubic grid, return (nz, ny, nx) float32."""
    nx, ny, nz = 73, 80, 40          # x: chest-nipple, y: lateral, z: compression
    img = image3D(shape=(nx, ny, nz),
                  xlen=nx * DX_CM, ylen=ny * DX_CM, zlen=nz * DX_CM,
                  x0=-2.5, y0=-ny * DX_CM / 2, z0=-nz * DX_CM / 2)
    build_breast_phantom().embed_in(img)
    vol = np.ascontiguousarray(img.mat.transpose(2, 1, 0))   # -> (nz, ny, nx)
    vol = np.clip(vol, 0.0, CLIP)
    vol *= 1.0 / max(vol.max(), 1e-9)
    return vol.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--itermax", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--slo", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    phantom = breast_volume()
    print(f"breast volume {phantom.shape} range [{phantom.min():.2f},"
          f"{phantom.max():.2f}]  frac>0={np.mean(phantom>0):.3f}")

    snaps = [10, 50, args.itermax]
    cfg = dict(itermax=args.itermax, npower=200, eps=args.eps,
               sigma_lo_scale=args.slo, eps_hi_ratio=1.0, eps_lo_ratio=1.25,
               seed=args.seed)
    geom = dict(orbit="dbt", det_rows=256, det_cols=256, det_spacing=0.06,
                nviews=25, arc_deg=50.0, sod=65.0, odd=5.0)
    res = recon3d.reconstruct(phantom, DX_CM, cfg=cfg, geom=geom,
                              snapshot_iters=snaps)

    si, tw = res["single"]["ierrs"], res["two"]["ierrs"]
    report = [r for r in (10, 50, 100, 150, 200, 300) if r <= args.itermax]
    lines = [f"# 3D DBT breast: image RMSE vs iteration "
             f"(shape {phantom.shape}, 25 views / 50 deg, eps={args.eps})\n",
             "| iter | single | two-channel (certified) |", "|---|---|---|"]
    print("\n=== image RMSE ===")
    for r in report:
        red = (si[r-1] - tw[r-1]) / si[r-1] * 100
        lines.append(f"| {r} | {si[r-1]:.5f} | {tw[r-1]:.5f} ({red:+.1f}%) |")
        print(f"  iter {r:4d}: single {si[r-1]:.5f}  two {tw[r-1]:.5f} ({red:+.1f}%)")
    (TAB / "recon3d_breast.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    it = np.arange(1, len(si) + 1)
    ax.semilogy(it, si, "k-", lw=2, label="single-channel")
    ax.semilogy(it, tw, "b-", lw=1.8, label="two-channel (certified)")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "recon3d_breast_convergence.png", dpi=200)

    # central sagittal (y-z) slice: shows the depth (z) direction DBT degrades
    mid_x = phantom.shape[2] // 2
    gt = phantom[:, :, mid_x]
    rs = res["single"]["recon"][:, :, mid_x]
    rt = res["two"]["recon"][:, :, mid_x]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    for ax, im, t in zip(axes, [gt, rs, rt],
                         ["ground truth", "single", "two-channel"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1.0, origin="lower", aspect="auto")
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "recon3d_breast_slices.png", dpi=200)
    print(f"\nWrote figs to {FIG}")


if __name__ == "__main__":
    main()
