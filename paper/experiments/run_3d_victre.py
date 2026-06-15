"""3D DBT certified two-channel test on the VICTRE breast phantom.

Loads the VICTRE uncompressed phantom (.raw), maps tissue labels to linear
attenuation, downsamples, and runs single vs certified two-channel (recon3d) in
DBT cone-beam geometry with the ramp-filtered Hann^{1/2} fidelity. Loader ported
from presentation/src/victre_reconstruction.py.
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

VICTRE = PAPER.parent / "data" / "victre_phantom"
MHD = VICTRE / "p_-72912147.mhd"
RAW = VICTRE / "p_-72912147.raw"
FIG = PAPER / "experiments" / "figs"
TAB = PAPER / "experiments" / "tables"

# VICTRE tissue label -> mu (cm^-1, ~30 keV); codes not listed -> air.
MU_TABLE = {0: 0.0, 1: 0.275, 2: 0.375, 29: 0.368, 33: 0.368, 40: 0.375,
            88: 0.368, 95: 0.368, 125: 0.368, 150: 0.368, 225: 0.368}
mu_lut = np.zeros(256, dtype=np.float32)
for k, v in MU_TABLE.items():
    mu_lut[k] = v


def parse_mhd(path):
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, val = [s.strip() for s in line.split("=", 1)]
            meta[k] = val
    dim = [int(s) for s in meta["DimSize"].split()]
    spacing = [float(s) for s in meta["ElementSpacing"].split()]
    return dim, spacing


def load_phantom(d):
    (nx, ny, nz), spacing = parse_mhd(MHD)
    dx_native = spacing[0]
    raw = np.memmap(RAW, dtype=np.uint8, mode="r", shape=(nz, ny, nx))
    nz_u, ny_u, nx_u = (nz // d) * d, (ny // d) * d, (nx // d) * d
    nz_d, ny_d, nx_d = nz_u // d, ny_u // d, nx_u // d
    phantom = np.zeros((nz_d, ny_d, nx_d), dtype=np.float32)
    for z0 in range(0, nz_u, 8 * d):                 # chunk over z to bound RAM
        z1 = min(z0 + 8 * d, nz_u)
        block = mu_lut[raw[z0:z1, :ny_u, :nx_u]]
        phantom[z0 // d:z1 // d] = block.reshape(
            (z1 - z0) // d, d, ny_d, d, nx_d, d).mean(axis=(1, 3, 5))
    dx_cm = dx_native * d / 10.0
    print(f"VICTRE downsampled {nx_d}x{ny_d}x{nz_d} @ {dx_cm*10:.2f} mm; "
          f"mu range [{phantom.min():.3f},{phantom.max():.3f}], "
          f"non-air {(phantom>1e-6).mean():.3f}")
    return np.ascontiguousarray(phantom), dx_cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down", type=int, default=6)
    ap.add_argument("--itermax", type=int, default=150)
    ap.add_argument("--slo", type=float, default=4.0)
    args = ap.parse_args()

    phantom, dx_cm = load_phantom(args.down)
    snaps = [10, 50, args.itermax]
    cfg = dict(itermax=args.itermax, npower=60, sigma_lo_scale=args.slo,
               eps_hi_ratio=1.0, eps_lo_ratio=1.25)        # ramp on by default
    geom = dict(orbit="dbt", det_rows=384, det_cols=384, det_spacing=0.05,
                nviews=25, arc_deg=50.0, sod=65.0, odd=5.0)
    res = recon3d.reconstruct(phantom, dx_cm, cfg=cfg, geom=geom,
                              snapshot_iters=snaps)

    si, tw = res["single"]["ierrs"], res["two"]["ierrs"]
    report = [r for r in (10, 50, 100, 150, 200) if r <= args.itermax]
    lines = [f"# 3D DBT VICTRE: image RMSE vs iteration "
             f"(shape {phantom.shape}, 25 views / 50 deg)\n",
             "| iter | single | two-channel (certified) |", "|---|---|---|"]
    print("\n=== image RMSE ===")
    for r in report:
        red = (si[r-1] - tw[r-1]) / si[r-1] * 100
        lines.append(f"| {r} | {si[r-1]:.5f} | {tw[r-1]:.5f} ({red:+.1f}%) |")
        print(f"  iter {r:4d}: single {si[r-1]:.5f}  two {tw[r-1]:.5f} ({red:+.1f}%)")
    (TAB / "recon3d_victre.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    it = np.arange(1, len(si) + 1)
    ax.semilogy(it, si, "k-", lw=2, label="single-channel")
    ax.semilogy(it, tw, "b-", lw=1.8, label="two-channel (certified)")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "recon3d_victre_convergence.png", dpi=200)

    vmax = float(np.percentile(phantom, 99.5))
    mid = phantom.shape[0] // 2                      # central z (compression) slice
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, im, t in zip(axes,
                         [phantom[mid], res["single"]["recon"][mid],
                          res["two"]["recon"][mid]],
                         ["ground truth", "single", "two-channel"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "recon3d_victre_slices.png", dpi=200)
    print(f"\nWrote figs to {FIG}")


if __name__ == "__main__":
    main()
