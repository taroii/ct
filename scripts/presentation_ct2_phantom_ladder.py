"""
Generic ct-2 analytic-phantom iteration-ladder driver for the CT-Meeting
2026 talk. Reuses scripts/victre_reconstruction.py for projector / norms
/ PDHG; swaps in different ct-2 phantom builders and per-phantom volume
boxes / detector sizes.

Run:
    python scripts/presentation_ct2_phantom_ladder.py --phantom breast
    python scripts/presentation_ct2_phantom_ladder.py --phantom head
    python scripts/presentation_ct2_phantom_ladder.py --phantom jaw

Outputs (under presentation/figs/):
    ct2_<name>_phantom_intro.png     three-view ground-truth slabs
    ct2_<name>_iter_ladder_xy.png    mid-axial single vs two ladder
    ct2_<name>_iter_ladder_roi.png   same ladder cropped to an ROI
    ct2_<name>_convergence.png       image RMSE vs iter

Cache:  cache/ct2_<name>_recon.pkl
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

import victre_reconstruction as vr   # noqa: E402
from phantom3d import image3D        # noqa: E402


def _install_grid_cache():
    """Cache the meshgrid in image3D._grids at float32 -- speeds embed_in
    from ~6 min down to seconds when many sub-objects are present."""
    def _cached_grids(self):
        key = (self.nx, self.ny, self.nz,
               self.x0, self.y0, self.z0,
               self.dx, self.dy, self.dz)
        cache = getattr(self, "_grid_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        x = self.x0 + (np.arange(self.nx, dtype=np.float32) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny, dtype=np.float32) + 0.5) * self.dy
        z = self.z0 + (np.arange(self.nz, dtype=np.float32) + 0.5) * self.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        self._grid_cache = (
            key,
            (X.astype(np.float32, copy=False),
             Y.astype(np.float32, copy=False),
             Z.astype(np.float32, copy=False))
        )
        return self._grid_cache[1]
    image3D._grids = _cached_grids


_install_grid_cache()


# ---------------------------------------------------------------------------
# Per-phantom configuration
#   shape:   (NX, NY, NZ) voxel grid
#   dx_cm:   isotropic voxel size in cm
#   center:  (xc, yc, zc) of the volume box in phantom coords
#   det:     (det_row_count, det_col_count, det_spacing_cm) for CBCT detector
#   roi:     (r0, r1, c0, c1) crop on the xy slice for the ROI ladder
#   itermax, snapshot_iters, ladder_iters
# ---------------------------------------------------------------------------
PHANTOM_CONFIGS = {
    "breast": {
        "builder_module": "breast_phantom_demo",
        "builder_func":   "build_breast_phantom",
        "shape":  (144, 144, 32),
        "dx_cm":  0.15,
        "center": (3.0, 0.0, 0.0),
        "det":    (384, 384, 0.05),     # 19.2 cm
        "roi":    (36, 110, 50, 124),
        # Wide display window: recons overshoot well above the phantom
        # max (~0.53), so a GT-fitted window saturated them to white.
        "display": {"vmin": 0.0, "vmax": 0.6},
        "itermax": 500,
        "snapshot_iters": [10, 50, 100, 200, 300, 500],
        "ladder_iters":   [10, 50, 100, 200],
        "intro_title": "ct-2 analytic breast phantom (144x144x32 @ 1.5 mm)",
        "ladder_title": "ct-2 breast -- mid-axial slice across iterations "
                        "(25 views / 50 deg LAR)",
        "roi_title":   "Soft-tissue ROI (lateral fatty region with mass inserts)",
        "conv_title":  "ct-2 breast phantom: image RMSE vs iteration",
    },
    "head": {
        "builder_module": "head_phantom_demo",
        "builder_func":   "build_head_phantom",
        # head x,y extent = 26 cm (full skull); z extent = 1.625 cm thin slab.
        # 176x176x16 at 1.5 mm = 26.4x26.4x2.4 cm: head fits in plane, slab
        # captured by central 11 z-voxels.
        "shape":  (176, 176, 16),
        "dx_cm":  0.15,
        "center": (0.0, 0.0, 0.0),
        # Detector must span the full skull (26 cm) without truncation.
        # 544x544 at 0.5 mm = 27.2 cm. Larger nrays -> slower but no truncation.
        "det":    (544, 544, 0.05),
        # ROI: central brain region around (0, 0) -> rows/cols ~ (60-130).
        "roi":    (60, 130, 60, 130),
        # Wide display window: recons overshoot above the phantom max
        # (1.8); a GT-fitted window saturated them to white.
        "display": {"vmin": 0.0, "vmax": 2.6},
        "itermax": 500,
        "snapshot_iters": [10, 50, 100, 200, 300, 500],
        "ladder_iters":   [10, 50, 100, 200],
        "intro_title": "ct-2 head phantom (176x176x16 @ 1.5 mm, 26 cm FOV)",
        "ladder_title": "ct-2 head -- mid-axial slice across iterations",
        "roi_title":   "Central brain ROI (ventricle + spots)",
        "conv_title":  "ct-2 head phantom: image RMSE vs iteration",
    },
    "jaw": {
        "builder_module": "jaw_phantom_demo",
        "builder_func":   "build_jaw_phantom",
        # jaw extent x=[-10,10], y=[-6,14], z=[0,12].
        # Center y at +4 (mid-mouth) to span anatomy of interest.
        "shape":  (144, 144, 80),
        "dx_cm":  0.15,
        "center": (0.0, 4.0, 6.0),
        # 432 x 432 @ 0.5 mm = 21.6 cm detector covers the 21.6 cm volume.
        "det":    (432, 432, 0.05),
        # Teeth+crown region typically sits around the center of the mid-axial
        # slice; centered ROI captures it.
        "roi":    (50, 110, 40, 110),
        "itermax": 500,
        "snapshot_iters": [10, 50, 100, 200, 300, 500],
        "ladder_iters":   [10, 50, 100, 200],
        "intro_title": "ct-2 jaw phantom (144x144x80 @ 1.5 mm)",
        "ladder_title": "ct-2 jaw -- mid-axial slice across iterations",
        "roi_title":   "Mouth/teeth ROI (gold crowns + soft tissue)",
        "conv_title":  "ct-2 jaw phantom: image RMSE vs iteration",
        # Gold crowns at att=19.3 dominate percentile-based vmax; clamp the
        # display window so soft tissue (att=1.0) and bone (att=2.0) are
        # the visible contrast band.
        "display": {"vmin": 0.0, "vmax": 3.0},
    },
}


def build_phantom_volume(cfg):
    """Build the analytic phantom and embed in an isotropic image grid."""
    import importlib
    mod = importlib.import_module(cfg["builder_module"])
    builder = getattr(mod, cfg["builder_func"])

    print(f"Building ct-2 phantom via {cfg['builder_module']}.{cfg['builder_func']} ...")
    # All ct-2 builders return a phantom3D positioned at their natural origin.
    # The breast builder accepts (breast_xc, breast_yc, breast_zc); call without
    # args otherwise.
    try:
        phantom = builder()
    except TypeError:
        phantom = builder(breast_xc=0., breast_yc=0., breast_zc=0.)

    NX, NY, NZ = cfg["shape"]
    DX = cfg["dx_cm"]
    xc, yc, zc = cfg["center"]
    xlen, ylen, zlen = NX * DX, NY * DX, NZ * DX

    img = image3D(
        shape=(NX, NY, NZ),
        xlen=xlen, ylen=ylen, zlen=zlen,
        x0=xc - xlen / 2., y0=yc - ylen / 2., z0=zc - zlen / 2.,
    )
    t0 = time.time()
    phantom.embed_in(img)
    print(f"  embed_in: {time.time()-t0:.1f}s")

    vol = img.mat
    print(f"  volume range: [{vol.min():.4f}, {vol.max():.4f}]")
    print(f"  non-zero fraction: {(vol > 1e-6).mean():.3f}")

    vol = np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32)
    print(f"  astra-ordered shape: {vol.shape}")
    return vol


def run_recon(cfg, arc_deg=50.0):
    phantom = build_phantom_volume(cfg)
    det_row, det_col, det_sp = cfg["det"]
    vol_geom, proj_geom, geom_info = vr.build_geometry(
        phantom.shape, cfg["dx_cm"],
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        arc_deg=arc_deg,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)

    R_hi, R_lo = vr.build_sinogram_filters(
        geom_info["det_col_count"], geom_info["det_spacing"],
        vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
    )

    sino_shape = (geom_info["det_row_count"],
                  geom_info["nviews"],
                  geom_info["det_col_count"])
    vr.adjoint_test(A, At, phantom.shape, sino_shape)

    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )

    saved_itermax = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = cfg["itermax"]
    try:
        print("\n--- single-channel ---")
        rs, is_, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            geom_info["nrays"], snapshot_iters=cfg["snapshot_iters"],
        )
        print("\n--- two-channel ---")
        rt, it_, dt_, tt_, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            geom_info["nrays"], snapshot_iters=cfg["snapshot_iters"],
        )
    finally:
        vr.CONFIG["itermax"] = saved_itermax

    return {
        "phantom": phantom,
        "recon_single": rs, "recon_two": rt,
        "ierrs_single": is_, "ierrs_two": it_,
        "derrs_single": ds_, "derrs_two": dt_,
        "tvs_single": ts_, "tvs_two": tt_,
        "snapshots_single": snaps_s,
        "snapshots_two":    snaps_t,
        "dx_cm": cfg["dx_cm"],
        "geometry": geom_info,
    }


def load_or_run(cache_path, cfg, force, arc_deg=50.0):
    if cache_path.exists() and not force:
        print(f"Loading cached recon from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    res = run_recon(cfg, arc_deg=arc_deg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(res, f)
    print(f"Cached recon -> {cache_path}")
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    z0, z1 = max(z - 1, 0), min(z + 2, nz)
    return vol[z0:z1].mean(axis=0)


def fig_phantom_intro(result, out_path, title=None, display=None):
    phi = result["phantom"]
    nz, ny, nx = phi.shape
    axial    = phi[nz // 2]
    coronal  = phi[:, ny // 2, :]
    sagittal = phi[:, :, nx // 2]

    if display is not None:
        vmin, vmax = display["vmin"], display["vmax"]
    else:
        nonzero = phi[phi > 1e-4]
        vmax = float(np.percentile(nonzero, 99.0)) * 1.05 if nonzero.size else 1.0
        vmin = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, img, label in zip(
        axes,
        (axial, coronal, sagittal),
        ("axial (xy)", "coronal (xz)", "sagittal (yz)"),
    ):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="lower", aspect="equal")
        ax.set_title(label, fontsize=11)
        _strip(ax)
    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_iter_ladder(result, iters, out_path, title=None, crop=None,
                    display=None):
    phi = result["phantom"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]

    slc_gt = _xy_slice(phi)
    if display is not None:
        vmin, vmax = display["vmin"], display["vmax"]
    else:
        nonzero = slc_gt[slc_gt > 0]
        vmax = float(np.percentile(nonzero, 99.0)) * 1.05 if nonzero.size else 1.0
        vmin = 0.0

    def crop_slice(s):
        if crop is None: return s
        r0, r1, c0, c1 = crop
        return s[r0:r1, c0:c1]

    n = len(iters)
    fig, axes = plt.subplots(3, n + 1, figsize=(1.9 * (n + 1), 5.6))
    for ax in axes.flat: _strip(ax)

    gt = crop_slice(slc_gt)
    for r, label in enumerate(["ground truth", "single", "two-channel"]):
        axes[r, 0].imshow(gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[r, 0].set_ylabel(label, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)

    for i, it in enumerate(iters):
        col = i + 1
        s = crop_slice(_xy_slice(snaps_s[it]))
        t = crop_slice(_xy_slice(snaps_t[it]))
        axes[0, col].imshow(gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[1, col].imshow(s, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[2, col].imshow(t, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

    if title is not None: fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_convergence(result, out_path, title=None, xlim=None):
    s = result["ierrs_single"]
    t = result["ierrs_two"]
    iters = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(iters, s, "r-", lw=1.4, label="single-channel")
    ax.semilogy(iters, t, "b-", lw=1.4, label="two-channel")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    if xlim is not None: ax.set_xlim(xlim)
    if title is not None: ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_error_ladder(result, iters, out_path, title=None, crop=None,
                     err_range=None, style="signed"):
    """Difference maps (reconstruction - ground truth), single vs
    two-channel, across iterations, with a colorbar legend.

    style:
      "signed"    -- signed grayscale: black under-estimate, mid-gray 0,
                     white over-estimate (the original representation).
      "magnitude" -- absolute error in grayscale: black 0, white largest.
                     Drops the sign; shows only where error is large.
      "diverging" -- colourblind-safe diverging map (blue under, white 0,
                     red over); high-contrast for a projector.

    err_range (auto = 99th percentile of |difference| over all panels) is
    symmetric for signed/diverging, [0, err_range] for magnitude.
    """
    phi = result["phantom"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]
    gt = _xy_slice(phi)

    def crop_slice(s):
        if crop is None:
            return s
        r0, r1, c0, c1 = crop
        return s[r0:r1, c0:c1]

    gt_c = crop_slice(gt)
    diffs_s = [crop_slice(_xy_slice(snaps_s[it])) - gt_c for it in iters]
    diffs_t = [crop_slice(_xy_slice(snaps_t[it])) - gt_c for it in iters]

    if err_range is None:
        allabs = np.concatenate([np.abs(d).ravel()
                                 for d in diffs_s + diffs_t])
        err_range = float(np.percentile(allabs, 99.0))

    if style == "magnitude":
        data_s = [np.abs(d) for d in diffs_s]
        data_t = [np.abs(d) for d in diffs_t]
        cmap, vmin, vmax = "gray", 0.0, err_range
        ticks = [0.0, err_range]
        ticklabels = ["0", f"{err_range:.2f}"]
        cbar_label = ("absolute error |reconstruction - ground truth|   "
                      "(black: no error,  white: largest error)")
    elif style == "diverging":
        data_s, data_t = diffs_s, diffs_t
        cmap, vmin, vmax = "RdBu_r", -err_range, err_range
        ticks = [-err_range, 0.0, err_range]
        ticklabels = [f"-{err_range:.2f}", "0", f"+{err_range:.2f}"]
        cbar_label = ("reconstruction - ground truth   "
                      "(blue: under-estimate,  white: no error,  "
                      "red: over-estimate)")
    else:  # signed
        data_s, data_t = diffs_s, diffs_t
        cmap, vmin, vmax = "gray", -err_range, err_range
        ticks = [-err_range, 0.0, err_range]
        ticklabels = [f"-{err_range:.2f}", "0", f"+{err_range:.2f}"]
        cbar_label = ("reconstruction - ground truth   "
                      "(black: under-estimate,  mid-gray: no error,  "
                      "white: over-estimate)")

    n = len(iters)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6),
                             constrained_layout=True)
    for ax in axes.flat:
        _strip(ax)
    axes[0, 0].set_ylabel("single - truth", fontsize=11)
    axes[1, 0].set_ylabel("two - truth", fontsize=11)
    im = None
    for i, it in enumerate(iters):
        im = axes[0, i].imshow(data_s[i], cmap=cmap, vmin=vmin, vmax=vmax,
                               origin="lower")
        axes[0, i].set_title(f"iter {it}", fontsize=10)
        axes[1, i].imshow(data_t[i], cmap=cmap, vmin=vmin, vmax=vmax,
                          origin="lower")
    if title is not None:
        fig.suptitle(title, fontsize=11)
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                        fraction=0.05, pad=0.02, aspect=50, ticks=ticks)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(cbar_label, fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", required=True, choices=list(PHANTOM_CONFIGS))
    ap.add_argument("--arc", type=float, default=50.0,
                    help="LAR source-arc in degrees (default 50)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = PHANTOM_CONFIGS[args.phantom]
    # Arc tag: empty for the default 50 deg, "_arc<N>" otherwise, so the
    # wide-arc runs get their own cache and figure files.
    tag = "" if abs(args.arc - 50.0) < 0.5 else f"_arc{int(round(args.arc))}"
    cache_path = ROOT / "cache" / f"ct2_{args.phantom}_recon{tag}.pkl"
    fig_dir = ROOT / "presentation" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    result = load_or_run(cache_path, cfg, args.force, arc_deg=args.arc)

    # Figure titles are intentionally omitted -- the slide frame title
    # names the phantom and the slide caption carries the generation note.
    display = cfg.get("display")
    fig_phantom_intro(
        result,
        fig_dir / f"ct2_{args.phantom}_phantom_intro{tag}.png",
        title=None,
        display=display,
    )
    fig_iter_ladder(
        result, cfg["ladder_iters"],
        fig_dir / f"ct2_{args.phantom}_iter_ladder_xy{tag}.png",
        title=None,
        display=display,
    )
    fig_iter_ladder(
        result, cfg["ladder_iters"],
        fig_dir / f"ct2_{args.phantom}_iter_ladder_roi{tag}.png",
        title=None,
        crop=cfg["roi"],
        display=display,
    )
    # Full slice (no ROI crop), magnitude style -- per team decision.
    fig_error_ladder(
        result, [50, 100, 200, 500],
        fig_dir / f"ct2_{args.phantom}_error_ladder{tag}.png",
        crop=None, style="magnitude",
    )
    fig_convergence(
        result,
        fig_dir / f"ct2_{args.phantom}_convergence{tag}.png",
        title=(f"Analytic {args.phantom} phantom ({args.arc:.0f} deg arc): "
               f"image RMSE vs iteration"),
        xlim=(0, 500),
    )


if __name__ == "__main__":
    main()
