"""Three synthetic 3D phantoms to diagnose whether the lack of 2D-like
two-channel performance in 3D is a phantom-design issue or a
regime/geometry issue. All three use the same recon config so results
are directly comparable.

C  (run first -- diagnostic): smooth 3D Gaussian blobs only. No edges,
   no texture. If two-channel still doesn't help here, the issue is the
   regime, not the phantom.

A  3D power-law noise field generated natively in 3D (1/f^alpha,
   alpha=2.5). Thresholded to create binary fibroglandular regions,
   combined with adipose background and sparse microcalcs -- the
   Reiser-style 2D recipe done natively in 3D so y is non-trivial.

B  Same as A plus an explicit slow 3D intensity gradient (smooth
   trilinear bias field). Forces low-frequency content that two-channel
   should specifically correct.

All phantoms get a circular xy petri-dish mask. Full physical size
matches the analytic breast: 21.6 x 21.6 x 4.8 cm at 0.05 cm voxels.

Geometry: DBT arc=15 deg, 25 views, FDA-spec 240^2 detector. CP at
cutoffparm_lo=3 + eps_lo_ratio=0.5 (2D-paper-default otherwise),
itermax=500 to see the full convergence story.

Outputs (per phantom):
    cache/ct2_breast_synthetic_<label>.pkl    snapshots + recons
    presentation/figs/ct2_breast_synthetic_<label>_intro.png
    presentation/figs/ct2_breast_synthetic_<label>_iter_ladder.png
    presentation/figs/ct2_breast_synthetic_<label>_error_ladder.png
    presentation/figs/ct2_breast_synthetic_<label>_convergence.png
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import victre_reconstruction as vr  # noqa: E402

CACHE_DIR = ROOT / "cache"
FIG_DIR   = ROOT / "presentation" / "figs"

# -- Geometry --
SHAPE = (432, 432, 96)    # (NX, NY, NZ) in builder convention; transposed for ASTRA later
DX_CM = 0.05
DET = (240, 240, 0.10)
NVIEWS = 25
ARC_DEG = 15.0
SOD_CM = 65.0
ODD_CM = 5.0

ITERMAX = 500
SNAPSHOT_ITERS = [10, 50, 100, 200, 300, 500]
LADDER_ITERS   = [10, 50, 100, 200, 500]

CP_OVERRIDES = {"cutoffparm_lo": 3.0, "eps_lo_ratio": 0.5}

# -- Petri dish mask radius (cm) --
PETRI_RADIUS_CM = 9.5   # 19 cm diameter dish inscribed in 21.6 cm box

# -- Display window (mu in cm^-1) --
DISPLAY = {"vmin": 0.0, "vmax": 0.4}

DEFAULTS = {
    "sigma_lo_scale":  vr.CONFIG["sigma_lo_scale"],
    "norm_inflate_3d": vr.CONFIG["norm_inflate_3d"],
    "eps_hi_ratio":    vr.CONFIG["eps_hi_ratio"],
    "eps_lo_ratio":    vr.CONFIG["eps_lo_ratio"],
    "cutoffparm":      vr.CONFIG["cutoffparm"],
    "cutoffparm_lo":   vr.CONFIG["cutoffparm_lo"],
    "beta":            vr.CONFIG["beta"],
}


def reset_cp():
    for k, v in DEFAULTS.items():
        vr.CONFIG[k] = v


def petri_dish_mask(shape, dx_cm, radius_cm):
    """Cylindrical mask: circular in xy, full extent in z."""
    NZ, NY, NX = shape
    y = (np.arange(NY) - (NY - 1) / 2) * dx_cm
    x = (np.arange(NX) - (NX - 1) / 2) * dx_cm
    Y, X = np.meshgrid(y, x, indexing="ij")
    in_disk = (Y * Y + X * X) <= radius_cm * radius_cm
    mask = np.broadcast_to(in_disk[np.newaxis, :, :], shape)
    return np.ascontiguousarray(mask).astype(bool)


# ============================================================================
# Phantom builders
# ============================================================================

def build_blob_phantom(shape, dx_cm, rng):
    """C: a few smooth 3D Gaussian blobs, no edges, no texture."""
    NZ, NY, NX = shape
    z = (np.arange(NZ) - (NZ - 1) / 2) * dx_cm
    y = (np.arange(NY) - (NY - 1) / 2) * dx_cm
    x = (np.arange(NX) - (NX - 1) / 2) * dx_cm
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    vol = np.full(shape, 0.10, dtype=np.float32)  # uniform background mu
    # A few overlapping blobs (cm centres, cm sigma, mu amplitude)
    blobs = [
        ((0.0,  0.0,   0.0), 3.5, 0.20),
        ((0.0,  3.5,   2.5), 2.0, 0.15),
        ((0.0, -3.0,  -2.0), 2.2, 0.18),
        ((0.5,  0.0,   4.0), 1.5, 0.12),
        ((-0.5, -4.0,  3.0), 1.8, 0.14),
    ]
    for (cz, cy, cx), sigma, amp in blobs:
        r2 = ((Z - cz) ** 2 + (Y - cy) ** 2 + (X - cx) ** 2) / (2 * sigma ** 2)
        vol += (amp * np.exp(-r2)).astype(np.float32)
    return vol


def _powerlaw_field(shape, alpha, rng):
    """Generate a normalized 1/f^alpha noise field via FFT filter."""
    NZ, NY, NX = shape
    fz = np.fft.fftfreq(NZ); fy = np.fft.fftfreq(NY); fx = np.fft.fftfreq(NX)
    FZ, FY, FX = np.meshgrid(fz, fy, fx, indexing="ij")
    f_mag = np.sqrt(FZ * FZ + FY * FY + FX * FX)
    f_mag[0, 0, 0] = 1.0
    filt = 1.0 / (f_mag ** alpha)
    filt[0, 0, 0] = 0.0
    white = rng.standard_normal(shape).astype(np.float32)
    field = np.real(np.fft.ifftn(np.fft.fftn(white) * filt)).astype(np.float32)
    # Standardize
    field = (field - field.mean()) / (field.std() + 1e-12)
    return field


def build_powerlaw_phantom(shape, dx_cm, rng,
                            alpha_main=2.5, alpha_hf=1.8,
                            hf_weight=0.18,
                            n_islands=6,
                            island_sigma_cm=1.8,
                            island_strength=2.2,
                            island_max_offset_cm=5.5,
                            island_aspect_range=(0.4, 2.5),
                            glandular_fraction=0.38,
                            intra_tissue_variation=0.10,
                            n_calc_clusters=40,
                            calc_cluster_size_range=(4, 12),
                            calc_cluster_radius_range=(1, 3),
                            skin_thickness_cm=0.20, skin_radius_cm=9.0,
                            mu_adipose=0.10, mu_glandular=0.22,
                            mu_skin=0.18, mu_calc=0.50):
    """A: 3D power-law phantom (single band, alpha=2.5) with a Gaussian
    center bias so the glandular region forms a large central island
    surrounded by adipose -- matching the spatial layout of the 2D
    paper-breast (Reiser-style) phantom.

    Single-band alpha=2.5 gives smoother large-scale boundaries (less
    fine speckle than the multi-band v1 / v2). The radial Gaussian
    center bias is added to the noise field before thresholding so
    voxels near the centre are more likely to be flagged glandular.
    A skin ring is added at the outer breast edge. Calcifications are
    emplaced as small clusters (3-12 voxels each). Within-tissue
    intensity is modulated by the noise field for slow density drift.
    """
    NZ, NY, NX = shape
    field_main = _powerlaw_field(shape, alpha_main, rng)
    field_hf   = _powerlaw_field(shape, alpha_hf, rng)
    field = (1.0 - hf_weight) * field_main + hf_weight * field_hf

    # Multi-island Gaussian bias (xy only). N Gaussian bumps at random
    # offsets inside island_max_offset_cm of the petri centre, so the
    # threshold gets crossed in several places -> multiple glandular
    # islands instead of one central blob.
    y = (np.arange(NY) - (NY - 1) / 2) * dx_cm
    x = (np.arange(NX) - (NX - 1) / 2) * dx_cm
    Y, X = np.meshgrid(y, x, indexing="ij")
    island_bias_xy = np.zeros_like(Y, dtype=np.float32)
    asp_lo, asp_hi = island_aspect_range
    for _ in range(n_islands):
        # Uniform offset within a disc of radius island_max_offset_cm
        r_off = island_max_offset_cm * np.sqrt(float(rng.uniform()))
        th_pos = 2 * np.pi * float(rng.uniform())
        cy, cx = r_off * np.sin(th_pos), r_off * np.cos(th_pos)
        # Anisotropic Gaussian: random aspect ratio + random orientation
        aspect = float(rng.uniform(asp_lo, asp_hi))
        sigma_u = island_sigma_cm * np.sqrt(aspect)
        sigma_v = island_sigma_cm / np.sqrt(aspect)
        th_rot = 2 * np.pi * float(rng.uniform())
        cos_t, sin_t = np.cos(th_rot), np.sin(th_rot)
        u =  cos_t * (X - cx) + sin_t * (Y - cy)
        v = -sin_t * (X - cx) + cos_t * (Y - cy)
        island_bias_xy += np.exp(
            -(u * u / (2 * sigma_u ** 2) + v * v / (2 * sigma_v ** 2))
        ).astype(np.float32)
    # normalize so peak bias ~ island_strength regardless of overlap
    peak = max(float(island_bias_xy.max()), 1e-12)
    island_bias_xy = island_bias_xy / peak
    island_bias = np.broadcast_to(island_bias_xy[np.newaxis, :, :],
                                   shape).astype(np.float32)
    biased = field + island_strength * island_bias

    # Binary glandular mask (single threshold on biased field)
    threshold = float(np.quantile(biased, 1 - glandular_fraction))
    glandular = biased > threshold

    combined = field   # used below for intra-tissue modulation

    vol = np.where(glandular, mu_glandular, mu_adipose).astype(np.float32)

    # Intra-tissue density variation (+-intra_tissue_variation of nominal mu)
    if intra_tissue_variation > 0:
        mod = intra_tissue_variation * (combined / max(np.abs(combined).max(), 1e-12))
        vol = vol * (1.0 + mod).astype(np.float32)

    # Outer skin ring (mu_skin) and air outside skin
    y = (np.arange(NY) - (NY - 1) / 2) * dx_cm
    x = (np.arange(NX) - (NX - 1) / 2) * dx_cm
    Y, X = np.meshgrid(y, x, indexing="ij")
    r_xy = np.sqrt(Y * Y + X * X)
    skin_mask_xy = (r_xy >= skin_radius_cm - skin_thickness_cm) & \
                   (r_xy <  skin_radius_cm)
    skin_mask = np.broadcast_to(skin_mask_xy[np.newaxis, :, :], shape)
    vol = np.where(skin_mask, mu_skin, vol).astype(np.float32)

    # Calcification clusters (groups of 3-9 voxels within a small radius)
    n_clusters = n_calc_clusters
    cs_lo, cs_hi = calc_cluster_size_range
    cr_lo, cr_hi = calc_cluster_radius_range
    for _ in range(n_clusters):
        cz = int(rng.integers(2, max(NZ - 2, 3)))
        cy = int(rng.integers(8, max(NY - 8, 9)))
        cx = int(rng.integers(8, max(NX - 8, 9)))
        # Avoid placing calcs in skin/air -- only inside tissue
        if not (mu_adipose * 0.5 <= vol[cz, cy, cx] <= mu_glandular * 1.3):
            continue
        size = int(rng.integers(cs_lo, cs_hi + 1))
        rad  = int(rng.integers(cr_lo, cr_hi + 1))
        for _ in range(size):
            dy = int(rng.integers(-rad, rad + 1))
            dx = int(rng.integers(-rad, rad + 1))
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < NY and 0 <= xx < NX:
                vol[cz, yy, xx] = mu_calc

    return vol


def build_powerlaw_gradient_phantom(shape, dx_cm, rng):
    """B: powerlaw phantom * (1 + 3D linear bias field)."""
    vol = build_powerlaw_phantom(shape, dx_cm, rng)
    NZ, NY, NX = shape
    # Trilinear ramp: lower at one corner, higher at the opposite corner
    z = (np.arange(NZ) / (NZ - 1) - 0.5).astype(np.float32)
    y = (np.arange(NY) / (NY - 1) - 0.5).astype(np.float32)
    x = (np.arange(NX) / (NX - 1) - 0.5).astype(np.float32)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    bias = 1.0 + 0.4 * (0.3 * Z + 0.5 * Y + 0.2 * X)  # +/- 20% across volume
    vol = (vol * bias).astype(np.float32)
    return vol


PHANTOM_BUILDERS = {
    "C_blob":              build_blob_phantom,
    "A_powerlaw":          build_powerlaw_phantom,
    "B_powerlaw_gradient": build_powerlaw_gradient_phantom,
}


def build_phantom(label, shape, dx_cm, seed=42):
    rng = np.random.default_rng(seed)
    builder = PHANTOM_BUILDERS[label]
    print(f"  building {label} ({shape} @ {dx_cm} cm)")
    t0 = time.time()
    vol = builder(shape, dx_cm, rng)
    print(f"  built in {time.time()-t0:.1f}s, range "
          f"[{vol.min():.4f}, {vol.max():.4f}]")
    # Apply petri-dish mask
    mask = petri_dish_mask(shape, dx_cm, PETRI_RADIUS_CM)
    vol = np.where(mask, vol, 0.0).astype(np.float32)
    print(f"  after petri-dish mask "
          f"(r={PETRI_RADIUS_CM} cm): mu range "
          f"[{vol.min():.4f}, {vol.max():.4f}], "
          f"non-zero {100 * (vol > 1e-4).mean():.1f}%")
    return vol


# ============================================================================
# Figures
# ============================================================================

def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    z0, z1 = max(z - 1, 0), min(z + 2, nz)
    return vol[z0:z1].mean(axis=0)


def fig_phantom_intro(phantom, dx_cm, out_path):
    nz, ny, nx = phantom.shape
    axial = _xy_slice(phantom)
    coronal = phantom[:, ny // 2, :]
    sagittal = phantom[:, :, nx // 2]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, img, lab, extent in zip(
        axes, (axial, coronal, sagittal),
        ("axial (xy)", "coronal (xz)", "sagittal (yz)"),
        ((0, nx*dx_cm, 0, ny*dx_cm),
         (0, nx*dx_cm, 0, nz*dx_cm),
         (0, ny*dx_cm, 0, nz*dx_cm)),
    ):
        ax.imshow(img, cmap="gray", vmin=DISPLAY["vmin"], vmax=DISPLAY["vmax"],
                  origin="lower", aspect="equal", extent=extent)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("cm"); ax.set_ylabel("cm")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def fig_iter_ladder(phantom, snaps_s, snaps_t, iters, out_path):
    slc_gt = _xy_slice(phantom)
    n = len(iters)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 3.8))
    for ax in axes.flat:
        _strip(ax)
    for r, lab in enumerate(["single", "two-channel"]):
        axes[r, 0].imshow(slc_gt, cmap="gray", vmin=DISPLAY["vmin"],
                          vmax=DISPLAY["vmax"], origin="lower")
        axes[r, 0].set_ylabel(lab, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)
    for i, it in enumerate(iters):
        col = i + 1
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[0, col].imshow(_xy_slice(snaps_s[it]), cmap="gray",
                            vmin=DISPLAY["vmin"], vmax=DISPLAY["vmax"],
                            origin="lower")
        axes[1, col].imshow(_xy_slice(snaps_t[it]), cmap="gray",
                            vmin=DISPLAY["vmin"], vmax=DISPLAY["vmax"],
                            origin="lower")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def fig_error_ladder(phantom, snaps_s, snaps_t, iters, out_path,
                     err_max=0.05):
    slc_gt = _xy_slice(phantom)
    n = len(iters)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 3.8))
    for ax in axes.flat:
        _strip(ax)
    axes[0, 0].imshow(slc_gt, cmap="gray", vmin=DISPLAY["vmin"],
                      vmax=DISPLAY["vmax"], origin="lower")
    axes[0, 0].set_title("ground truth", fontsize=10)
    axes[1, 0].imshow(slc_gt, cmap="gray", vmin=DISPLAY["vmin"],
                      vmax=DISPLAY["vmax"], origin="lower")
    axes[0, 0].set_ylabel("single error", fontsize=10)
    axes[1, 0].set_ylabel("two error", fontsize=10)
    for i, it in enumerate(iters):
        col = i + 1
        err_s = np.abs(_xy_slice(snaps_s[it]) - slc_gt)
        err_t = np.abs(_xy_slice(snaps_t[it]) - slc_gt)
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[0, col].imshow(err_s, cmap="hot", vmin=0, vmax=err_max,
                            origin="lower")
        axes[1, col].imshow(err_t, cmap="hot", vmin=0, vmax=err_max,
                            origin="lower")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def fig_convergence(ierrs_single, ierrs_two, out_path, title):
    s = np.asarray(ierrs_single); t = np.asarray(ierrs_two)
    its = np.arange(1, len(s) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, xlim, title_ in [
        (ax1, (0, len(s) + 0.5), "linear-y, full"),
        (ax2, (5, min(100.5, len(s) + 0.5)), "zoom: iter 5-100"),
    ]:
        ax.plot(its, s, "r-", lw=1.7, label="single")
        ax.plot(its, t, "b-", lw=1.7, label="two-channel")
        ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc="upper right")
        ax.set_title(title_)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ============================================================================
# Recon driver
# ============================================================================

def run_one(label, phantom):
    print(f"\n##### {label} #####")
    det_row, det_col, det_sp = DET
    vol_geom, proj_geom, gi = vr.build_dbt_geometry(
        phantom.shape, DX_CM,
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        nviews=NVIEWS, arc_deg=ARC_DEG, sod=SOD_CM, odd=ODD_CM,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )

    saved = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX
    try:
        reset_cp()
        print("\n--- single ---")
        t0 = time.time()
        rs, is_, _, _, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        print(f"  single {time.time()-t0:.0f}s, "
              f"RMSE@iter{ITERMAX} = {is_[-1]:.5f}")

        reset_cp()
        for k, v in CP_OVERRIDES.items():
            vr.CONFIG[k] = v
        R_hi, R_lo = vr.build_sinogram_filters(
            gi["det_col_count"], gi["det_spacing"],
            vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
            axis="2d", det_row_count=gi["det_row_count"],
        )
        print("\n--- two ---")
        t0 = time.time()
        rt, it_, _, _, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        print(f"  two    {time.time()-t0:.0f}s, "
              f"RMSE@iter{ITERMAX} = {it_[-1]:.5f}")
    finally:
        vr.CONFIG["itermax"] = saved
        reset_cp()

    print("\n=== iter-by-iter ===")
    print(f"  {'iter':>5}  {'single':>9}  {'two':>9}  {'red %':>7}")
    for it in SNAPSHOT_ITERS:
        a = is_[it - 1]; b = it_[it - 1]
        print(f"  {it:5d}  {a:9.5f}  {b:9.5f}  {(a-b)/a*100:+7.2f}")

    # Save cache
    out_pkl = CACHE_DIR / f"ct2_breast_synthetic_{label}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "phantom":          phantom,
            "recon_single":     rs, "recon_two": rt,
            "ierrs_single":     is_, "ierrs_two": it_,
            "snapshots_single": snaps_s, "snapshots_two": snaps_t,
            "dx_cm":            DX_CM, "geometry": gi,
            "label":            label,
            "cp_overrides":     CP_OVERRIDES,
        }, f)
    print(f"  cached {out_pkl.name}")

    # Render figures
    fig_phantom_intro(phantom, DX_CM,
                      FIG_DIR / f"ct2_breast_synthetic_{label}_intro.png")
    fig_iter_ladder(phantom, snaps_s, snaps_t, LADDER_ITERS,
                    FIG_DIR / f"ct2_breast_synthetic_{label}_iter_ladder.png")
    fig_error_ladder(phantom, snaps_s, snaps_t, LADDER_ITERS,
                     FIG_DIR / f"ct2_breast_synthetic_{label}_error_ladder.png")
    fig_convergence(is_, it_,
                    FIG_DIR / f"ct2_breast_synthetic_{label}_convergence.png",
                    f"synthetic phantom {label} -- DBT arc=15, 25v, 432^3, "
                    f"cutLo=3 eps_lo=0.5 itermax={ITERMAX}")
    return rs, rt, is_, it_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", choices=list(PHANTOM_BUILDERS) + ["all"],
                    default="all")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = list(PHANTOM_BUILDERS) if args.phantom == "all" else [args.phantom]
    nz, ny, nx = SHAPE[2], SHAPE[1], SHAPE[0]  # (NX, NY, NZ) -> astra (NZ, NY, NX)
    astra_shape = (nz, ny, nx)

    for label in labels:
        # Note: astra wants (nz, ny, nx) ordering for the volume array.
        # build_phantom builds in that order directly (np shape NZ first).
        phantom = build_phantom(label, astra_shape, DX_CM)
        try:
            run_one(label, phantom)
        except Exception as exc:
            import traceback
            print(f"\n!!! {label} ERRORED: {exc}")
            traceback.print_exc()
            continue

    print("\nAll done.")


if __name__ == "__main__":
    main()
