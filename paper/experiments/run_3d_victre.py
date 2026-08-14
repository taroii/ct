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


CACHE = PAPER.parent / "data" / "victre_cache"


def cache_path(d):
    return CACHE / f"victre_p-72912147_down{d}.npz"


def load_phantom(d, use_cache=True):
    """Downsampled VICTRE volume, from a small cache when available.

    The source .raw is 601 MB and gitignored, so a fresh checkout (e.g. on a
    compute server) cannot build the phantom. The downsampled volume the
    experiments actually consume is far smaller -- 2.5 MB at d=4, since the
    volume is label-derived and holds only a handful of distinct mu values --
    so it is cached to data/victre_cache/ and preferred when present, and is
    small enough to commit to git. That makes the
    phantom byte-identical across machines, which matters as much for
    reproducibility as the seeding does: a phantom rebuilt with a different
    numpy version could differ in the last bits and shift every RMSE.

    Build the cache once on a machine that has the .raw:
        python paper/experiments/run_3d_victre.py --cache-only --down 4
    then copy data/victre_cache/ to the server.
    """
    p = cache_path(d)
    if use_cache and p.exists():
        with np.load(p) as z:
            phantom, dx_cm = z["phantom"], float(z["dx_cm"])
        print(f"VICTRE from cache {p.name}: {phantom.shape} @ {dx_cm*10:.2f} mm")
        return np.ascontiguousarray(phantom), dx_cm
    if not RAW.exists():
        raise FileNotFoundError(
            f"Neither the downsample cache ({p}) nor the source raw ({RAW}) "
            f"is present. Build the cache on a machine that has the .raw with "
            f"'--cache-only --down {d}', then copy data/victre_cache/ across.")

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
    phantom = np.ascontiguousarray(phantom)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, phantom=phantom, dx_cm=np.float64(dx_cm),
                            down=np.int32(d), source=np.str_(RAW.name))
        print(f"  cached -> {p} ({p.stat().st_size/1e6:.1f} MB)")
    return phantom, dx_cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--itermax", type=int, default=150)
    ap.add_argument("--slo", type=float, default=4.0)
    ap.add_argument("--vmax", type=float, default=0.0,
                    help="display window max; 0 = auto from reconstructions")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds the power-iteration inits; identical seeds give "
                         "identical step sizes and hence identical runs")
    ap.add_argument("--cache-only", action="store_true",
                    help="build the downsample cache and exit (no GPU needed); "
                         "copy data/victre_cache/ to a machine without the .raw")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames; required when sweeping "
                         "seeds, since runs otherwise overwrite each other")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    phantom, dx_cm = load_phantom(args.down)
    if args.cache_only:
        print(f"cache ready: {cache_path(args.down)}")
        return
    snaps = [10, 50, args.itermax]
    cfg = dict(itermax=args.itermax, npower=200, sigma_lo_scale=args.slo,
               eps_hi_ratio=1.0, eps_lo_ratio=1.25,        # ramp on by default
               seed=args.seed)
    geom = dict(orbit="dbt", det_rows=480, det_cols=480, det_spacing=0.04,
                nviews=25, arc_deg=50.0, sod=65.0, odd=5.0)
    res = recon3d.reconstruct(phantom, dx_cm, cfg=cfg, geom=geom,
                              snapshot_iters=snaps)

    si, tw = res["single"]["ierrs"], res["two"]["ierrs"]
    report = [r for r in (10, 50, 100, 150, 200, 300, 400, 500) if r <= args.itermax]
    lines = [f"# 3D DBT VICTRE: image RMSE vs iteration "
             f"(shape {phantom.shape}, 25 views / 50 deg)\n",
             "| iter | single | two-channel (certified) |", "|---|---|---|"]
    print("\n=== image RMSE ===")
    for r in report:
        red = (si[r-1] - tw[r-1]) / si[r-1] * 100
        lines.append(f"| {r} | {si[r-1]:.5f} | {tw[r-1]:.5f} ({red:+.1f}%) |")
        print(f"  iter {r:4d}: single {si[r-1]:.5f}  two {tw[r-1]:.5f} ({red:+.1f}%)")
    (TAB / f"recon3d_victre{tag}.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    it = np.arange(1, len(si) + 1)
    ax.semilogy(it, si, "k-", lw=2, label="single-channel")
    ax.semilogy(it, tw, "b-", lw=1.8, label="two-channel (certified)")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / f"recon3d_victre_convergence{tag}.png", dpi=200)

    # Window the display so the recon's limited-angle overshoot does not clip
    # to white (the "light bloom"). Base vmax on the reconstructions, not the
    # phantom, and pad above their bright tail so tissue sits in mid-gray.
    if args.vmax > 0:
        vmax = args.vmax
    else:
        vmax = 1.15 * float(max(np.percentile(res["single"]["recon"], 99.5),
                                np.percentile(res["two"]["recon"], 99.5)))
    print(f"display window vmin=0 vmax={vmax:.3f}")
    mid = phantom.shape[0] // 2                      # central z (compression) slice
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, im, t in zip(axes,
                         [phantom[mid], res["single"]["recon"][mid],
                          res["two"]["recon"][mid]],
                         ["ground truth", "single", "two-channel"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / f"recon3d_victre_slices{tag}.png", dpi=200)
    print(f"\nWrote figs to {FIG}")


if __name__ == "__main__":
    main()
