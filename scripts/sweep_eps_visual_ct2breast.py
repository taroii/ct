"""
H4 -- 3D ct-2 analytic breast (50-deg arc), eps_lo/eps_hi ratio sweep,
rendered as mid-axial recon + error rows across iterations
(per Emil 2026-05-22).

Mirrors scripts/sweep_eps_visual_2d.py on the 3D breast phantom that backs
the deck's main 3D slide. Holds c_hi=4, c_lo=8 (paper config) and varies
the per-band tolerance ratio r = eps_lo / eps_hi in {0.25, 0.5, 1.0, 1.25, 2.0}.

Phantom, geometry, projector, operator norms, and sinogram filters R_hi,
R_lo are all built once and reused across the sweep -- only the eps ratios
change, so this is the cheapest of the four sweeps.

Outputs (under presentation/figs/):
  ct2_breast_H4_eps_recon.png    mid-axial recon panels
  ct2_breast_H4_eps_error.png    |recon - truth| panels
  ct2_breast_H4_eps_convergence.png
  ct2_breast_H4_eps_summary.txt
Cache:
  cache/ct2_breast_H4_eps_sweep.pkl

Run:  conda run --no-capture-output -n ct2 \
          python scripts/sweep_eps_visual_ct2breast.py [--force]
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ct-2"))

import victre_reconstruction as vr                # noqa: E402
import presentation_ct2_breast_ladder as ct2      # noqa: E402

# --- Defaults --------------------------------------------------------------
C_HI_FIXED = 4.0
C_LO_FIXED = 8.0
RATIO_DEFAULTS = [0.25, 0.5, 1.0, 1.25, 2.0]
SNAPSHOT_ITERS = [50, 100, 200, 500]
ITERMAX = 500

CACHE_PATH = ROOT / "cache" / "ct2_breast_H4_eps_sweep.pkl"
FIG_DIR = ROOT / "presentation" / "figs"

ERR_RANGE = 0.15


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    return vol[z-1:z+2].mean(axis=0)


def setup_once():
    phantom = ct2.build_breast_volume()
    vol_geom, proj_geom, geom_info = vr.build_geometry(phantom.shape, ct2.DX_CM)
    A, At = vr.make_projector(vol_geom, proj_geom)
    sino_shape = (geom_info["det_row_count"],
                  geom_info["nviews"],
                  geom_info["det_col_count"])
    vr.adjoint_test(A, At, phantom.shape, sino_shape)
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )
    R_hi, R_lo = vr.build_sinogram_filters(
        geom_info["det_col_count"], geom_info["det_spacing"],
        C_HI_FIXED, C_LO_FIXED,
    )
    return {
        "phantom": phantom, "geom_info": geom_info,
        "A": A, "At": At,
        "nusino": nusino,
        "nuxgrad": nuxgrad, "nuygrad": nuygrad, "nuzgrad": nuzgrad,
        "R_hi": R_hi, "R_lo": R_lo,
    }


def run_sweep(ratios, force=False):
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)-2} cached runs + single from {CACHE_PATH}")
        return results

    results = {}
    setup = setup_once()
    phantom = setup["phantom"]
    geom = setup["geom_info"]

    saved_itermax = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX
    vr.CONFIG["cutoffparm"] = C_HI_FIXED
    vr.CONFIG["cutoffparm_lo"] = C_LO_FIXED
    vr.CONFIG["eps_hi_ratio"] = 1.0
    try:
        print("\n=== single-channel reference ===")
        t0 = time.time()
        rs, is_, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, setup["A"], setup["At"],
            setup["nusino"], setup["nuxgrad"], setup["nuygrad"], setup["nuzgrad"],
            geom["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        print(f"  elapsed {time.time()-t0:.1f}s")
        results["single"] = {
            "snapshots": snaps_s, "ierrs": is_, "derrs": ds_, "tvs": ts_,
            "final_rmse": float(np.sqrt(((rs - phantom) ** 2).mean())),
        }
        results["phantom"] = phantom
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(results, f)

        for r in ratios:
            if r in results:
                print(f"[skip] eps_lo/eps_hi={r} already cached")
                continue
            vr.CONFIG["eps_hi_ratio"] = 1.0
            vr.CONFIG["eps_lo_ratio"] = float(r)
            print(f"\n=== two-channel: eps_lo/eps_hi={r} ===")
            t0 = time.time()
            rt, it_, dt_, tt_, snaps_t = vr.run_two_channel(
                phantom, setup["A"], setup["At"], setup["R_hi"], setup["R_lo"],
                setup["nusino"], setup["nuxgrad"], setup["nuygrad"], setup["nuzgrad"],
                geom["nrays"], snapshot_iters=SNAPSHOT_ITERS,
            )
            print(f"  elapsed {time.time()-t0:.1f}s")
            results[r] = {
                "ratio": r,
                "snapshots": snaps_t, "ierrs": it_,
                "derrs": dt_, "tvs": tt_,
                "final_rmse": float(np.sqrt(((rt - phantom) ** 2).mean())),
            }
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(results, f)
    finally:
        vr.CONFIG["itermax"] = saved_itermax

    return results


# --- Figures (parallel to H3) ---------------------------------------------
def _build_rows(results, ratios):
    rows = [("single (ref)", results["single"]["snapshots"])]
    for r in ratios:
        rows.append((f"two, eps_lo/eps_hi={r:g}", results[r]["snapshots"]))
    return rows


def _percentile_window(vol):
    vmax = float(np.percentile(vol[vol > 0], 99.0)) * 1.05
    return 0.0, vmax


def fig_recon_grid(results, ratios, iters, out_path, title=None):
    rows = _build_rows(results, ratios)
    phi = results["phantom"]
    slc_gt = _xy_slice(phi)
    vmin, vmax = _percentile_window(slc_gt)
    n = len(iters)
    fig, axes = plt.subplots(len(rows), n + 1, figsize=(1.9 * (n + 1), 1.9 * len(rows) + 0.5))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        _strip(ax)

    for r, (label, snaps) in enumerate(rows):
        axes[r, 0].imshow(slc_gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[r, 0].set_ylabel(label, fontsize=10)
        if r == 0:
            axes[r, 0].set_title("ground truth", fontsize=9)
        for i, it in enumerate(iters):
            col = i + 1
            slc = _xy_slice(snaps[it])
            axes[r, col].imshow(slc, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            if r == 0:
                axes[r, col].set_title(f"iter {it}", fontsize=9)
            axes[r, col].set_xlabel(f"RMSE {_rmse(snaps[it], phi):.3f}", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_error_grid(results, ratios, iters, out_path, title=None):
    rows = _build_rows(results, ratios)
    phi = results["phantom"]
    n = len(iters)
    fig, axes = plt.subplots(len(rows), n, figsize=(1.9 * n, 1.9 * len(rows) + 0.5))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        _strip(ax)

    for r, (label, snaps) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=10)
        for i, it in enumerate(iters):
            diff = _xy_slice(snaps[it]) - _xy_slice(phi)
            axes[r, i].imshow(diff, cmap="gray", vmin=-ERR_RANGE, vmax=ERR_RANGE, origin="lower")
            if r == 0:
                axes[r, i].set_title(f"iter {it}", fontsize=9)
            axes[r, i].set_xlabel(f"slice-RMSE {_rmse(_xy_slice(snaps[it]), _xy_slice(phi)):.3f}",
                                  fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_convergence(results, ratios, out_path, title=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ierrs_s = results["single"]["ierrs"]
    ax.semilogy(np.arange(1, len(ierrs_s) + 1), ierrs_s, "k-", lw=1.4, label="single (ref)")
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(ratios):
        ier = results[r]["ierrs"]
        color = cmap(i / max(1, len(ratios) - 1))
        ax.semilogy(np.arange(1, len(ier) + 1), ier, "-", color=color, lw=1.2,
                    label=f"two, eps_lo/eps_hi={r:g}")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def write_summary(results, ratios, iters, path):
    phi = results["phantom"]
    lines = [
        "H4 eps_lo/eps_hi sweep -- 3D ct-2 breast (50-deg arc, 144x144x32 @ 1.5 mm)",
        f"c_hi={C_HI_FIXED}, c_lo={C_LO_FIXED} fixed; itermax={ITERMAX}",
        "=" * 72, "",
        f"{'config':<28} | " + " | ".join(f"iter {it:>4}" for it in iters),
        "-" * 72,
    ]
    snaps_s = results["single"]["snapshots"]
    rmses_s = [_rmse(snaps_s[it], phi) for it in iters]
    lines.append(f"{'single (ref)':<28} | " + " | ".join(f"{v:.4f}    " for v in rmses_s))
    for r in ratios:
        snaps_t = results[r]["snapshots"]
        rmses_t = [_rmse(snaps_t[it], phi) for it in iters]
        lines.append(f"{'two eps_lo/eps_hi='+str(r):<28} | " + " | ".join(f"{v:.4f}    " for v in rmses_t))
    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="3D ct-2 breast eps_lo/eps_hi sweep (H4)")
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIO_DEFAULTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sweep(args.ratios, force=args.force)

    fig_recon_grid(
        results, args.ratios, SNAPSHOT_ITERS,
        FIG_DIR / "ct2_breast_H4_eps_recon.png",
        title=f"ct-2 breast 50deg -- eps_lo/eps_hi sweep (c_hi={C_HI_FIXED}, c_lo={C_LO_FIXED})",
    )
    fig_error_grid(
        results, args.ratios, SNAPSHOT_ITERS,
        FIG_DIR / "ct2_breast_H4_eps_error.png",
        title=f"ct-2 breast 50deg -- error maps (mid-axial), gray +-{ERR_RANGE}",
    )
    fig_convergence(
        results, args.ratios,
        FIG_DIR / "ct2_breast_H4_eps_convergence.png",
        title="ct-2 breast 50deg -- image RMSE vs iter, eps_lo/eps_hi sweep",
    )
    write_summary(results, args.ratios, SNAPSHOT_ITERS,
                  FIG_DIR / "ct2_breast_H4_eps_summary.txt")


if __name__ == "__main__":
    main()
