"""
H3 -- 3D ct-2 analytic breast (50-deg arc), non-PoU low-pass cutoff sweep,
rendered as mid-axial recon + error rows across iterations
(per Emil 2026-05-22).

Mirrors scripts/sweep_cutoff_visual_2d.py on the 3D breast phantom that
already backs the deck's main 3D slide (presentation_ct2_breast_ladder.py).
Fixes c_hi = 4, varies c_lo in {6, 8, 12, 16}; eps_lo/eps_hi = 1.25.

Phantom, geometry, projector, and operator norms are built once and reused
across the sweep. Only the sinogram filters R_hi/R_lo are rebuilt per c_lo
value. Single-channel is run once as a reference row.

Outputs (under presentation/figs/, alongside the existing ct2_breast_* set):
  ct2_breast_H3_cutoff_recon.png    mid-axial recon panels
  ct2_breast_H3_cutoff_error.png    |recon - truth| panels
  ct2_breast_H3_cutoff_convergence.png
  ct2_breast_H3_cutoff_summary.txt
Cache:
  cache/ct2_breast_H3_cutoff_sweep.pkl

Run:  conda run --no-capture-output -n ct2 \
          python scripts/sweep_cutoff_visual_ct2breast.py [--force]
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
import presentation_ct2_breast_ladder as ct2      # noqa: E402  (triggers grid-cache patch)

# --- Defaults --------------------------------------------------------------
C_HI_FIXED = 4.0
C_LO_DEFAULTS = [6.0, 8.0, 12.0, 16.0]
SNAPSHOT_ITERS = [50, 100, 200, 500]
ITERMAX = 500
EPS_LO_RATIO = 1.25

CACHE_PATH = ROOT / "cache" / "ct2_breast_H3_cutoff_sweep.pkl"
FIG_DIR = ROOT / "presentation" / "figs"

ERR_RANGE = 0.15


# --- Helpers ---------------------------------------------------------------
def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    return vol[z-1:z+2].mean(axis=0)


def setup_once():
    """Build phantom + geometry + projector + norms (cached on first call)."""
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
    return {
        "phantom": phantom,
        "geom_info": geom_info,
        "A": A, "At": At,
        "nusino": nusino,
        "nuxgrad": nuxgrad, "nuygrad": nuygrad, "nuzgrad": nuzgrad,
    }


def run_sweep(c_los, force=False):
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)-2} cached two-channel runs + single from {CACHE_PATH}")
        return results

    results = {}
    setup = setup_once()
    phantom = setup["phantom"]
    geom = setup["geom_info"]

    saved_itermax = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX
    vr.CONFIG["eps_lo_ratio"] = EPS_LO_RATIO
    vr.CONFIG["eps_hi_ratio"] = 1.0
    try:
        # Single-channel reference (independent of c_lo).
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

        for c_lo in c_los:
            if c_lo in results:
                print(f"[skip] c_lo={c_lo} already cached")
                continue
            vr.CONFIG["cutoffparm"] = C_HI_FIXED
            vr.CONFIG["cutoffparm_lo"] = float(c_lo)
            R_hi, R_lo = vr.build_sinogram_filters(
                geom["det_col_count"], geom["det_spacing"],
                C_HI_FIXED, float(c_lo),
            )
            print(f"\n=== two-channel: c_hi={C_HI_FIXED}, c_lo={c_lo} ===")
            t0 = time.time()
            rt, it_, dt_, tt_, snaps_t = vr.run_two_channel(
                phantom, setup["A"], setup["At"], R_hi, R_lo,
                setup["nusino"], setup["nuxgrad"], setup["nuygrad"], setup["nuzgrad"],
                geom["nrays"], snapshot_iters=SNAPSHOT_ITERS,
            )
            print(f"  elapsed {time.time()-t0:.1f}s")
            results[c_lo] = {
                "c_hi": C_HI_FIXED, "c_lo": c_lo,
                "snapshots": snaps_t, "ierrs": it_,
                "derrs": dt_, "tvs": tt_,
                "final_rmse": float(np.sqrt(((rt - phantom) ** 2).mean())),
            }
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(results, f)
    finally:
        vr.CONFIG["itermax"] = saved_itermax

    return results


# --- Figures ---------------------------------------------------------------
def _build_rows(results, c_los):
    rows = [("single (ref)", results["single"]["snapshots"])]
    for c in c_los:
        rows.append((f"two, c_lo={c:g}", results[c]["snapshots"]))
    return rows


def _percentile_window(vol):
    vmax = float(np.percentile(vol[vol > 0], 99.0)) * 1.05
    return 0.0, vmax


def fig_recon_grid(results, c_los, iters, out_path, title=None):
    rows = _build_rows(results, c_los)
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
            rmse_3d = _rmse(snaps[it], phi)   # volumetric RMSE (matches deck)
            axes[r, col].set_xlabel(f"RMSE {rmse_3d:.3f}", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_error_grid(results, c_los, iters, out_path, title=None):
    rows = _build_rows(results, c_los)
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


def fig_convergence(results, c_los, out_path, title=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ierrs_s = results["single"]["ierrs"]
    ax.semilogy(np.arange(1, len(ierrs_s) + 1), ierrs_s, "k-", lw=1.4, label="single (ref)")
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(c_los):
        ier = results[c]["ierrs"]
        color = cmap(i / max(1, len(c_los) - 1))
        ax.semilogy(np.arange(1, len(ier) + 1), ier, "-", color=color, lw=1.2,
                    label=f"two, c_lo={c:g}")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def write_summary(results, c_los, iters, path):
    phi = results["phantom"]
    lines = [
        "H3 cutoff sweep -- 3D ct-2 breast (50-deg arc, 144x144x32 @ 1.5 mm)",
        f"c_hi fixed at {C_HI_FIXED}, eps_lo/eps_hi = {EPS_LO_RATIO}, itermax={ITERMAX}",
        "=" * 72, "",
        f"{'config':<22} | " + " | ".join(f"iter {it:>4}" for it in iters),
        "-" * 72,
    ]
    snaps_s = results["single"]["snapshots"]
    rmses_s = [_rmse(snaps_s[it], phi) for it in iters]
    lines.append(f"{'single (ref)':<22} | " + " | ".join(f"{v:.4f}    " for v in rmses_s))
    for c in c_los:
        snaps_t = results[c]["snapshots"]
        rmses_t = [_rmse(snaps_t[it], phi) for it in iters]
        lines.append(f"{'two c_lo='+str(c):<22} | " + " | ".join(f"{v:.4f}    " for v in rmses_t))
    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="3D ct-2 breast non-PoU cutoff sweep (H3)")
    parser.add_argument("--c-los", type=float, nargs="+", default=C_LO_DEFAULTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sweep(args.c_los, force=args.force)

    fig_recon_grid(
        results, args.c_los, SNAPSHOT_ITERS,
        FIG_DIR / "ct2_breast_H3_cutoff_recon.png",
        title=f"ct-2 breast 50deg -- non-PoU cutoff sweep (c_hi={C_HI_FIXED}, vary c_lo)",
    )
    fig_error_grid(
        results, args.c_los, SNAPSHOT_ITERS,
        FIG_DIR / "ct2_breast_H3_cutoff_error.png",
        title=f"ct-2 breast 50deg -- error maps (mid-axial slice), gray +-{ERR_RANGE}",
    )
    fig_convergence(
        results, args.c_los,
        FIG_DIR / "ct2_breast_H3_cutoff_convergence.png",
        title="ct-2 breast 50deg -- image RMSE vs iter, non-PoU cutoff sweep",
    )
    write_summary(results, args.c_los, SNAPSHOT_ITERS,
                  FIG_DIR / "ct2_breast_H3_cutoff_summary.txt")


if __name__ == "__main__":
    main()
