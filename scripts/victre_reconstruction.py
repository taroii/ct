"""
Full-phantom VICTRE DTV-L1 CBCT reconstruction (LAR 50 deg), single and
two channel.

Loads the full VICTRE breast phantom from data/victre_phantom/p_-72912147.raw,
downsamples 4x (0.4 mm voxels), simulates a 25-view circular cone-beam scan
truncated to a 50 deg arc, and runs both the single-channel and two-channel
Chambolle-Pock DTV+L1 reconstructions from
scripts/compare_methods_multiresolution.py ported to 3D.

The two-channel variant splits the sinogram along the detector-column
(u) axis into high- and low-frequency bands via sqrt-Hanning filters,
and uses the diagonal-dual-preconditioning convergence condition
    tau * ||Sigma^(1/2) K||_2^2 < 1
with Sigma = diag(sig_hi, sig_lo, sig_tv, sig_tv, sig_tv, sig_l1).

Cone-beam forward/backprojection is delegated to astra-toolbox (CUDA).
"""

import sys
import time
import pickle
from pathlib import Path

import numpy as np
from numpy import sqrt, maximum
from numpy.random import randn
import matplotlib.pyplot as plt

try:
    import astra
except ImportError:
    sys.exit(
        "astra-toolbox is required. Install it manually, e.g.\n"
        "    conda install -c astra-toolbox astra-toolbox\n"
        "or see https://astra-toolbox.com/docs/install.html"
    )

np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # CP algorithm (same scalars as scripts/compare_methods_multiresolution.py)
    "beta":        5.0,     # L1 sparsity cap
    "rho":         1.75,    # over-relaxation
    "eps":         0.001,   # data-discrepancy RMSE
    "l1f":         1.0,     # L1 block weight
    "stepbalance": 100.0,   # tau <-> sigma split
    "nuxfact":     0.5,     # per-axis gradient prescale factor
    "nuyfact":     0.5,
    "nuzfact":     0.5,
    "tv_cap":      1.0,     # isotropic TV cap per axis (analogue of alpha/2-alpha)

    # Two-channel parameters
    "cutoffparm":       4.0,   # hi-pass Hanning cutoff (= cutoff for han_hi)
    "cutoffparm_lo":    8.0,   # lo-pass Hanning cutoff
    "eps_hi_ratio":     1.0,   # eps_hi = eps_hi_ratio * eps
    "eps_lo_ratio":     1.25,  # eps_lo = eps_lo_ratio * eps
    "sigma_lo_scale":   4.0,   # r_lo = sig_lo / sig_hi
    "sigma_tv_scale":   1.0,   # r_tv = sig_tv / sig_hi
    "sigma_l1_scale":   1.0,   # r_l1 = sig_l1 / sig_hi
    "cvg_slack":        1.e-3, # strict "< 1" slack on the CP convergence condition

    # Iterations
    "itermax":     500,
    "npower":      200,
}

DOWNSAMPLE = 4   # bin native 0.1 mm voxels to 0.4 mm

PHANTOM_MHD = Path("data/victre_phantom/p_-72912147.mhd")
PHANTOM_RAW = Path("data/victre_phantom/p_-72912147.raw")

CACHE_DIR = Path("cache")
FIGURE_DIR = Path("final_figures/victre")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TISSUE LABEL -> LINEAR ATTENUATION (cm^-1, approx 30 keV)
# ============================================================================
# Standard VICTRE tissue codes (Graff 2016). Values for codes not listed
# default to 0 (treated as air). Tune later if the simulation needs to be
# physically precise; for an RMSE-style test these are fine.

MU_TABLE = {
    0:   0.000,   # air / background
    1:   0.275,   # adipose / fat
    2:   0.375,   # skin
    29:  0.368,   # glandular
    33:  0.368,   # muscle / ligament
    40:  0.375,   # nipple
    88:  0.368,   # Cooper's ligament
    95:  0.368,   # TDLU (lobular)
    125: 0.368,   # duct
    150: 0.368,   # artery
    225: 0.368,   # vein
}

mu_lut = np.zeros(256, dtype=np.float32)
for k, v in MU_TABLE.items():
    mu_lut[k] = v

# ============================================================================
# PHANTOM LOADING + DOWNSAMPLING
# ============================================================================

def parse_mhd(path):
    meta = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = [s.strip() for s in line.split("=", 1)]
        meta[k] = v
    dim = [int(s) for s in meta["DimSize"].split()]
    spacing = [float(s) for s in meta["ElementSpacing"].split()]
    return dim, spacing

def load_and_downsample_phantom():
    dim_xyz, spacing_xyz = parse_mhd(PHANTOM_MHD)
    nx, ny, nz = dim_xyz          # .mhd DimSize is X Y Z
    dx_native = spacing_xyz[0]    # assume isotropic spacing (VICTRE = 0.1 mm)

    print(f"Native phantom: {nx}x{ny}x{nz} voxels at {dx_native} mm spacing")
    print(f"Physical extent: {nx*dx_native:.1f} x {ny*dx_native:.1f} x {nz*dx_native:.1f} mm")

    # astra/numpy convention: volume array shape (Z, Y, X)
    raw = np.memmap(PHANTOM_RAW, dtype=np.uint8, mode="r", shape=(nz, ny, nx))

    # Trim to a multiple of DOWNSAMPLE along each axis
    d = DOWNSAMPLE
    nz_use = (nz // d) * d
    ny_use = (ny // d) * d
    nx_use = (nx // d) * d
    nz_d, ny_d, nx_d = nz_use // d, ny_use // d, nx_use // d

    phantom = np.zeros((nz_d, ny_d, nx_d), dtype=np.float32)

    # Walk along Z in chunks to avoid materializing the whole float32 volume at once
    chunk_slices = 8 * d           # 32 native z-slices per pass
    for z0 in range(0, nz_use, chunk_slices):
        z1 = min(z0 + chunk_slices, nz_use)
        block = mu_lut[raw[z0:z1, :ny_use, :nx_use]]          # (dz, ny_use, nx_use) float32
        dz = z1 - z0
        binned = block.reshape(dz // d, d,
                               ny_d, d,
                               nx_d, d).mean(axis=(1, 3, 5))
        phantom[z0 // d : z1 // d] = binned

    dx_d = dx_native * d / 10.0    # mm -> cm
    print(f"Downsampled phantom: {nx_d}x{ny_d}x{nz_d} voxels at {dx_d*10:.2f} mm = {dx_d:.3f} cm")
    print(f"mu range: [{phantom.min():.4f}, {phantom.max():.4f}] cm^-1, "
          f"non-air fraction: {(phantom > 1e-6).mean():.3f}")
    return phantom, dx_d

# ============================================================================
# 3D FINITE-DIFFERENCE GRADIENT / NEG-ADJOINT (matches gmatx.T convention)
# ============================================================================
# gmatx in the 2D script has -1 on the diagonal (all rows) and +1 on the
# super-diagonal (all but the last row). Equivalently: forward difference
# with a Dirichlet "zero outside" boundary at the upper end. Each of the
# 3D operators below is the axis-wise analogue; divX/Y/Z are the literal
# matrix transpose, not a negated gradient.

def gradx(v):                    # last axis
    g = -v.copy()
    g[:, :, :-1] += v[:, :, 1:]
    return g

def mdivx(g):
    d = -g.copy()
    d[:, :, 1:] += g[:, :, :-1]
    return d

def grady(v):                    # middle axis
    g = -v.copy()
    g[:, :-1, :] += v[:, 1:, :]
    return g

def mdivy(g):
    d = -g.copy()
    d[:, 1:, :] += g[:, :-1, :]
    return d

def gradz(v):                    # first axis (rotation axis)
    g = -v.copy()
    g[:-1, :, :] += v[1:, :, :]
    return g

def mdivz(g):
    d = -g.copy()
    d[1:, :, :] += g[:-1, :, :]
    return d

# ============================================================================
# ASTRA CONE-BEAM GEOMETRY + FORWARD/ADJOINT WRAPPERS
# ============================================================================

def build_geometry(phantom_shape, dx_cm):
    nz_d, ny_d, nx_d = phantom_shape

    vol_geom = astra.create_vol_geom(
        ny_d, nx_d, nz_d,
        -nx_d * dx_cm / 2, nx_d * dx_cm / 2,
        -ny_d * dx_cm / 2, ny_d * dx_cm / 2,
        -nz_d * dx_cm / 2, nz_d * dx_cm / 2,
    )

    # LAR scan: 25 views across a 50 deg arc.  Circular cone-beam trajectory,
    # source and detector rotating together around the phantom Z axis.
    nviews = 25
    arc_deg = 50.0
    angles = np.deg2rad(np.linspace(-arc_deg / 2, arc_deg / 2, nviews))

    source_origin = 50.0     # cm  (matches 2D script: radius = 50)
    origin_det    = 50.0     # cm  (total SDD = 100, same as 2D)

    det_row_count = 384
    det_col_count = 384
    det_spacing   = 0.05     # cm per pixel -> 19.2 cm detector

    proj_geom = astra.create_proj_geom(
        "cone",
        det_spacing, det_spacing,
        det_row_count, det_col_count,
        angles,
        source_origin, origin_det,
    )

    nrays = nviews * det_row_count * det_col_count
    info = {
        "nviews": nviews, "arc_deg": arc_deg,
        "source_origin": source_origin, "origin_det": origin_det,
        "det_row_count": det_row_count, "det_col_count": det_col_count,
        "det_spacing": det_spacing,
        "nrays": nrays,
    }
    print(f"Cone-beam: {nviews} views x {det_row_count}x{det_col_count} detector "
          f"@ {det_spacing*10:.2f} mm, arc = {arc_deg} deg, SOD/ODD = {source_origin}/{origin_det} cm")
    return vol_geom, proj_geom, info

def make_projector(vol_geom, proj_geom):
    """Return (A, At) closures over astra GPU forward/backward projectors."""
    def A(vol):
        sid, sino = astra.create_sino3d_gpu(
            np.ascontiguousarray(vol, dtype=np.float32), proj_geom, vol_geom
        )
        astra.data3d.delete(sid)
        return sino
    def At(sino):
        vid, vol = astra.create_backprojection3d_gpu(
            np.ascontiguousarray(sino, dtype=np.float32), proj_geom, vol_geom
        )
        astra.data3d.delete(vid)
        return vol
    return A, At

# ============================================================================
# SINOGRAM HI/LO FREQUENCY SPLIT (R_hi, R_lo)
# ============================================================================
# 1D FFT-domain filters along the detector u-axis (det_col, last axis of the
# 3D sinogram).  Mirrors the 2D script's R_hi/R_lo/R_fft_weight, with the
# sqrt-ramp dropped because the 3D astra projector is unfiltered. The filters
# are pure sqrt-Hanning so R_hi^T R_hi, R_lo^T R_lo are diagonal in the
# detector-u Fourier domain -- exactly the structure the two-channel CP
# diagonal dual preconditioning assumes.

def build_sinogram_filters(det_col_count, det_spacing, cutoffparm, cutoffparm_lo):
    nb = det_col_count
    db = det_spacing
    blen = nb * db
    b00 = -blen / 2.0
    uar = np.arange(b00 + db / 2.0, b00 + blen, db)   # length nb, symmetric

    def hanning_window(u, c):
        uhanp = abs(b00) / c
        h = 0.5 * (1.0 + np.cos(np.pi * u / uhanp))
        h[np.abs(u) > uhanp] = 0.0
        return h

    han_lo = np.clip(hanning_window(uar, cutoffparm_lo), 0.0, 1.0)
    han_hi = np.clip(1.0 - hanning_window(uar, cutoffparm), 0.0, 1.0)

    F_lo = np.sqrt(han_lo)
    F_hi = np.sqrt(han_hi)

    # fftshifted so they line up with np.fft.fft output order (DC first)
    W_lo = np.fft.fftshift(F_lo).astype(np.float32)
    W_hi = np.fft.fftshift(F_hi).astype(np.float32)

    def R_filter(sino, W):
        # sino shape (det_row_count, nviews, det_col_count); filter last axis.
        imft = np.fft.fft(sino, axis=-1)
        imft *= W
        return np.fft.ifft(imft, axis=-1).real.astype(np.float32)

    def R_hi(sino): return R_filter(sino, W_hi)
    def R_lo(sino): return R_filter(sino, W_lo)

    print(f"Sinogram filters: cutoff_hi={cutoffparm}, cutoff_lo={cutoffparm_lo}, "
          f"||F_hi||_inf={float(F_hi.max()):.3f}, ||F_lo||_inf={float(F_lo.max()):.3f}")
    return R_hi, R_lo

# ============================================================================
# OPERATOR-NORM POWER ITERATION  (port of lines 430-474, 3D)
# ============================================================================

def operator_norms(phantom_shape, A, At, npower):
    nz_d, ny_d, nx_d = phantom_shape

    # ||A|| via At(A(.))
    xim = randn(nz_d, ny_d, nx_d).astype(np.float32)
    xnorm = 0.0
    for _ in range(npower):
        s = A(xim)
        xim = At(s)
        xnorm = sqrt((xim * xim).sum()) + 1e-12
        xim /= xnorm
    nusino = 1.0 / sqrt(xnorm)

    def axis_norm(grad, div):
        x = randn(nz_d, ny_d, nx_d).astype(np.float32)
        n = 0.0
        for _ in range(npower // 4):
            g = grad(x)
            x = div(g)
            n = sqrt((x * x).sum()) + 1e-12
            x /= n
        return sqrt(n)

    gxn = axis_norm(gradx, mdivx)
    gyn = axis_norm(grady, mdivy)
    gzn = axis_norm(gradz, mdivz)

    nuxgrad = CONFIG["nuxfact"] / gxn
    nuygrad = CONFIG["nuyfact"] / gyn
    nuzgrad = CONFIG["nuzfact"] / gzn
    print(f"Operator norms: ||A||={1/nusino:.4f}  "
          f"||dx||={gxn:.4f}  ||dy||={gyn:.4f}  ||dz||={gzn:.4f}")
    print(f"nusino={nusino:.6f}  nuxgrad={nuxgrad:.6f}  "
          f"nuygrad={nuygrad:.6f}  nuzgrad={nuzgrad:.6f}")
    return nusino, nuxgrad, nuygrad, nuzgrad

# ============================================================================
# SINGLE-CHANNEL CP LOOP (port of lines 476-607, 3D, no Hanning filter,
# no FOV mask, 3 gradient blocks instead of 2, isotropic TV cap)
# ============================================================================

def run_single_channel(phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad, nrays):
    nz_d, ny_d, nx_d = phantom.shape
    cfg = CONFIG

    # Total operator norm (stacked [A ; dx ; dy ; dz ; l1]) via a joint power iteration
    print("Joint power iteration for stacked operator norm...")
    xim = randn(nz_d, ny_d, nx_d).astype(np.float32)
    xnorm = sqrt((xim * xim).sum()) + 1e-12
    xim /= xnorm
    for _ in range(cfg["npower"]):
        s = A(xim) * nusino
        gx = gradx(xim) * nuxgrad
        gy = grady(xim) * nuygrad
        gz = gradz(xim) * nuzgrad
        yl = cfg["l1f"] * xim
        mag1 = sqrt((s*s).sum() + (gx*gx).sum() + (gy*gy).sum() +
                    (gz*gz).sum() + (yl*yl).sum())
        if mag1 > 0:
            s  /= mag1; gx /= mag1; gy /= mag1; gz /= mag1; yl /= mag1
        xim = (At(s) * nusino +
               mdivx(gx) * nuxgrad +
               mdivy(gy) * nuygrad +
               mdivz(gz) * nuzgrad +
               cfg["l1f"] * yl)
        mag2 = sqrt((xim * xim).sum()) + 1e-12
        xim /= mag2
    totalnorm = (mag1 + mag2) * 0.5
    sig = cfg["stepbalance"] / totalnorm
    tau = 1.0 / (totalnorm * cfg["stepbalance"])
    print(f"||K||={totalnorm:.4f}  sig={sig:.6f}  tau={tau:.6f}")

    epssc = cfg["eps"] * sqrt(nrays)

    # Ground-truth sinogram (no noise, matches 2D addnoise=0)
    print("Forward-projecting ground-truth phantom...")
    truesino = A(phantom)
    sinodatasc = nusino * truesino
    print(f"truesino range [{truesino.min():.4f}, {truesino.max():.4f}]")

    # Primal / duals
    xim     = np.zeros_like(phantom)
    xbarim  = np.zeros_like(phantom)
    ygradx_ = np.zeros_like(phantom)
    ygrady_ = np.zeros_like(phantom)
    ygradz_ = np.zeros_like(phantom)
    yim     = np.zeros_like(phantom)
    ysino   = np.zeros_like(truesino)

    rho   = cfg["rho"]
    theta = 1.0
    beta  = cfg["beta"]
    cap   = cfg["tv_cap"]
    l1f   = cfg["l1f"]
    itermax = cfg["itermax"]
    istops = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]

    ierrs, derrs, tvs = [], [], []
    n_voxels = phantom.size
    t0 = time.time()

    for itr in range(1, itermax + 1):
        ysino_old  = ysino.copy()
        ygradx_old = ygradx_.copy()
        ygrady_old = ygrady_.copy()
        ygradz_old = ygradz_.copy()
        yim_old    = yim.copy()
        ximold     = xim.copy()

        # Primal descent: x <- x - tau * K^T y
        wimp  = At(ysino) * nusino
        wimqx = mdivx(ygradx_) * nuxgrad
        wimqy = mdivy(ygrady_) * nuygrad
        wimqz = mdivz(ygradz_) * nuzgrad
        wiml1 = l1f * yim
        xim = xim - tau * (wimp + wimqx + wimqy + wimqz + wiml1)
        xim[xim < 0.0] = 0.0
        xbarim = xim + theta * (xim - ximold)

        # Dual: data fidelity (eps-ball projection on ysino)
        Ax = A(xbarim) * nusino
        resid = Ax - sinodatasc
        derr = sqrt(((resid / nusino) ** 2).sum() / float(nrays))
        derrs.append(derr)

        ysino = ysino + sig * resid
        ymag = sqrt((ysino * ysino).sum())
        slack = sig * nusino * epssc
        if ymag - slack > 0:
            ysino *= (ymag - slack) / ymag
        else:
            ysino *= 0.0

        # Dual: per-axis TV prox  (cap=1 each axis -> plain anisotropic TV)
        tgx = gradx(xbarim) * nuxgrad
        ptilx = ygradx_ + sig * tgx
        ygradx_ = cap * ptilx / maximum(np.abs(ptilx), cap)

        tgy = grady(xbarim) * nuygrad
        ptily = ygrady_ + sig * tgy
        ygrady_ = cap * ptily / maximum(np.abs(ptily), cap)

        tgz = gradz(xbarim) * nuzgrad
        ptilz = ygradz_ + sig * tgz
        ygradz_ = cap * ptilz / maximum(np.abs(ptilz), cap)

        # Dual: L1 prox (cap=beta)
        tl1 = l1f * xbarim
        ptil1 = yim + sig * tl1
        yim = beta * ptil1 / maximum(np.abs(ptil1), beta)

        # 3D TV for logging
        gx_log = gradx(xbarim); gy_log = grady(xbarim); gz_log = gradz(xbarim)
        tvs.append(float(sqrt(gx_log*gx_log + gy_log*gy_log + gz_log*gz_log).sum()))

        # Over-relaxation inertial step
        ygradx_ = ygradx_old - rho * (ygradx_old - ygradx_)
        ygrady_ = ygrady_old - rho * (ygrady_old - ygrady_)
        ygradz_ = ygradz_old - rho * (ygradz_old - ygradz_)
        ysino   = ysino_old  - rho * (ysino_old  - ysino)
        yim     = yim_old    - rho * (yim_old    - yim)
        xim     = ximold     - rho * (ximold     - xim)

        ierrs.append(float(sqrt(((xbarim - phantom) ** 2).sum() / n_voxels)))

        if itr in istops or itr == itermax:
            print(f"  iter {itr:4d}: data_err={derrs[-1]:.6f}  "
                  f"img_err={ierrs[-1]:.6f}  TV={tvs[-1]:.2e}  "
                  f"({time.time()-t0:.1f}s)")

    print(f"CP loop done in {time.time()-t0:.1f}s")
    return xbarim, ierrs, derrs, tvs

# ============================================================================
# TWO-CHANNEL CP LOOP
# ============================================================================
# Port of the two-channel block in scripts/compare_methods_multiresolution.py
# (lines ~613-790), 3D-ified and with the diagonal-dual-preconditioning
# convergence condition enforced at step-size selection, exactly as done in
# the 2D script after the convergence-theorem fix:
#
#     tau * || Sigma^(1/2) K ||_2^2  <  1
#
# with Sigma = diag(sig_hi, sig_lo, sig_tv, sig_tv, sig_tv, sig_l1) and
# K = [ R_hi A ; R_lo A ; grad_x ; grad_y ; grad_z ; l1f ].  The weighted
# power iteration below estimates ||Sigma_ratio^(1/2) K||_2 directly, and
# the assignments make  tau * sig_hi * norm_weighted^2 = 1/(1+cvg_slack).

def run_two_channel(phantom, A, At, R_hi, R_lo,
                    nusino, nuxgrad, nuygrad, nuzgrad, nrays):
    nz_d, ny_d, nx_d = phantom.shape
    cfg = CONFIG

    r_lo = cfg["sigma_lo_scale"]
    r_tv = cfg["sigma_tv_scale"]
    r_l1 = cfg["sigma_l1_scale"]
    sqrt_r_lo = float(np.sqrt(r_lo))
    sqrt_r_tv = float(np.sqrt(r_tv))
    sqrt_r_l1 = float(np.sqrt(r_l1))

    # Weighted joint power iteration estimating  ||Sigma_ratio^(1/2) K||_2
    print("Weighted joint power iteration (two-channel)...")
    xim = randn(nz_d, ny_d, nx_d).astype(np.float32)
    xim /= sqrt((xim * xim).sum()) + 1e-12
    mag1 = mag2 = 0.0
    for _ in range(cfg["npower"]):
        # Forward K_w x: apply K, then scale each block by sqrt(r_block).
        Ax = A(xim)
        s_hi = R_hi(Ax) * nusino
        s_lo = R_lo(Ax) * (nusino * sqrt_r_lo)
        gx = gradx(xim) * (nuxgrad * sqrt_r_tv)
        gy = grady(xim) * (nuygrad * sqrt_r_tv)
        gz = gradz(xim) * (nuzgrad * sqrt_r_tv)
        yl = (cfg["l1f"] * sqrt_r_l1) * xim
        mag1 = sqrt((s_hi*s_hi).sum() + (s_lo*s_lo).sum() +
                    (gx*gx).sum() + (gy*gy).sum() + (gz*gz).sum() +
                    (yl*yl).sum())
        if mag1 > 0:
            s_hi /= mag1; s_lo /= mag1
            gx /= mag1; gy /= mag1; gz /= mag1
            yl /= mag1
        # Adjoint K_w^T y: scale each block of y by sqrt(r_block) again
        # (adjoint of the diagonal weighting), then apply K^T.  Combined
        # across hi/lo data blocks before the shared backprojection.
        bp_in = R_hi(s_hi) + R_lo(s_lo * sqrt_r_lo)
        xim = (At(bp_in) * nusino +
               mdivx(gx * sqrt_r_tv) * nuxgrad +
               mdivy(gy * sqrt_r_tv) * nuygrad +
               mdivz(gz * sqrt_r_tv) * nuzgrad +
               (cfg["l1f"] * sqrt_r_l1) * yl)
        mag2 = sqrt((xim * xim).sum()) + 1e-12
        xim /= mag2
    norm_weighted = (mag1 + mag2) * 0.5

    # ----------------------------------------------------------------------
    # Convergence condition (product-space CP with diagonal dual preconditioning):
    #     tau * || Sigma^(1/2) K ||_2^2  <  1
    # with Sigma = diag(sig_hi, sig_lo, sig_tv, sig_tv, sig_tv, sig_l1).
    # norm_weighted estimates ||Sigma_ratio^(1/2) K||_2 directly (sig_hi
    # factored out).  cvg_slack > 0 shrinks tau so the condition holds
    # strictly rather than at equality.
    # ----------------------------------------------------------------------
    cvg_slack = cfg["cvg_slack"]
    sig_hi = cfg["stepbalance"] / norm_weighted
    sig_lo = r_lo * sig_hi
    sig_tv = r_tv * sig_hi           # used for ygradx, ygrady, ygradz
    sig_l1 = r_l1 * sig_hi           # used for yim
    tau = 1.0 / (norm_weighted * cfg["stepbalance"] * (1.0 + cvg_slack))

    lhs_cvg = tau * sig_hi * norm_weighted ** 2
    assert lhs_cvg < 1.0, f"CP convergence condition violated: {lhs_cvg:.6f} >= 1"

    epssc_hi = cfg["eps_hi_ratio"] * cfg["eps"] * sqrt(nrays)
    epssc_lo = cfg["eps_lo_ratio"] * cfg["eps"] * sqrt(nrays)

    print(f"||Sigma^(1/2) K||={norm_weighted:.4f}  "
          f"sig_hi={sig_hi:.6f}  sig_lo={sig_lo:.6f}  tau={tau:.6f}  "
          f"tau*sig_hi*||.||^2={lhs_cvg:.6f}  (< 1 required)")

    # Ground-truth sinogram split into hi/lo channels
    print("Forward-projecting phantom and splitting into hi/lo channels...")
    truesino = A(phantom)
    sinodatasc_hi = nusino * R_hi(truesino)
    sinodatasc_lo = nusino * R_lo(truesino)

    # Primal / duals
    xim     = np.zeros_like(phantom)
    xbarim  = np.zeros_like(phantom)
    ygradx_ = np.zeros_like(phantom)
    ygrady_ = np.zeros_like(phantom)
    ygradz_ = np.zeros_like(phantom)
    yim     = np.zeros_like(phantom)
    ysino_hi = np.zeros_like(truesino)
    ysino_lo = np.zeros_like(truesino)

    rho   = cfg["rho"]
    theta = 1.0
    beta  = cfg["beta"]
    cap   = cfg["tv_cap"]
    l1f   = cfg["l1f"]
    itermax = cfg["itermax"]
    istops = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]

    ierrs, derrs, tvs = [], [], []
    n_voxels = phantom.size
    t0 = time.time()

    for itr in range(1, itermax + 1):
        ysino_hi_old = ysino_hi.copy()
        ysino_lo_old = ysino_lo.copy()
        ygradx_old = ygradx_.copy()
        ygrady_old = ygrady_.copy()
        ygradz_old = ygradz_.copy()
        yim_old    = yim.copy()
        ximold     = xim.copy()

        # Primal descent: x <- x - tau * K^T y
        wimp  = At(R_hi(ysino_hi) + R_lo(ysino_lo)) * nusino
        wimqx = mdivx(ygradx_) * nuxgrad
        wimqy = mdivy(ygrady_) * nuygrad
        wimqz = mdivz(ygradz_) * nuzgrad
        wiml1 = l1f * yim
        xim = xim - tau * (wimp + wimqx + wimqy + wimqz + wiml1)
        xim[xim < 0.0] = 0.0
        xbarim = xim + theta * (xim - ximold)

        # Dual: hi/lo data fidelity (two eps-balls)
        Ax = A(xbarim) * nusino
        Ax_hi = R_hi(Ax)
        Ax_lo = R_lo(Ax)
        resid_hi = Ax_hi - sinodatasc_hi
        resid_lo = Ax_lo - sinodatasc_lo
        derr = sqrt(
            (((resid_hi / nusino) ** 2).sum() +
             ((resid_lo / nusino) ** 2).sum()) / float(nrays)
        )
        derrs.append(float(derr))

        ysino_hi = ysino_hi + sig_hi * resid_hi
        ymag_hi = sqrt((ysino_hi * ysino_hi).sum())
        slack_hi = sig_hi * nusino * epssc_hi
        if ymag_hi - slack_hi > 0:
            ysino_hi *= (ymag_hi - slack_hi) / ymag_hi
        else:
            ysino_hi *= 0.0

        ysino_lo = ysino_lo + sig_lo * resid_lo
        ymag_lo = sqrt((ysino_lo * ysino_lo).sum())
        slack_lo = sig_lo * nusino * epssc_lo
        if ymag_lo - slack_lo > 0:
            ysino_lo *= (ymag_lo - slack_lo) / ymag_lo
        else:
            ysino_lo *= 0.0

        # Dual: per-axis TV prox  (sig_tv, not sig_hi)
        tgx = gradx(xbarim) * nuxgrad
        ptilx = ygradx_ + sig_tv * tgx
        ygradx_ = cap * ptilx / maximum(np.abs(ptilx), cap)

        tgy = grady(xbarim) * nuygrad
        ptily = ygrady_ + sig_tv * tgy
        ygrady_ = cap * ptily / maximum(np.abs(ptily), cap)

        tgz = gradz(xbarim) * nuzgrad
        ptilz = ygradz_ + sig_tv * tgz
        ygradz_ = cap * ptilz / maximum(np.abs(ptilz), cap)

        # Dual: L1 prox  (sig_l1, not sig_hi)
        tl1 = l1f * xbarim
        ptil1 = yim + sig_l1 * tl1
        yim = beta * ptil1 / maximum(np.abs(ptil1), beta)

        # 3D TV for logging
        gx_log = gradx(xbarim); gy_log = grady(xbarim); gz_log = gradz(xbarim)
        tvs.append(float(sqrt(gx_log*gx_log + gy_log*gy_log + gz_log*gz_log).sum()))

        # Over-relaxation
        ygradx_ = ygradx_old - rho * (ygradx_old - ygradx_)
        ygrady_ = ygrady_old - rho * (ygrady_old - ygrady_)
        ygradz_ = ygradz_old - rho * (ygradz_old - ygradz_)
        ysino_hi = ysino_hi_old - rho * (ysino_hi_old - ysino_hi)
        ysino_lo = ysino_lo_old - rho * (ysino_lo_old - ysino_lo)
        yim     = yim_old    - rho * (yim_old    - yim)
        xim     = ximold     - rho * (ximold     - xim)

        ierrs.append(float(sqrt(((xbarim - phantom) ** 2).sum() / n_voxels)))

        if itr in istops or itr == itermax:
            print(f"  iter {itr:4d}: data_err={derrs[-1]:.6f}  "
                  f"img_err={ierrs[-1]:.6f}  TV={tvs[-1]:.2e}  "
                  f"({time.time()-t0:.1f}s)")

    print(f"Two-channel CP loop done in {time.time()-t0:.1f}s")
    return xbarim, ierrs, derrs, tvs

# ============================================================================
# ADJOINT TEST  (<A x, y> vs <x, At y>)
# ============================================================================

def adjoint_test(A, At, phantom_shape, truesino_shape):
    x = randn(*phantom_shape).astype(np.float32)
    y = randn(*truesino_shape).astype(np.float32)
    lhs = float((A(x) * y).sum())
    rhs = float((x * At(y)).sum())
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    print(f"Adjoint test: <Ax,y>={lhs:.6e}  <x,Aty>={rhs:.6e}  rel={rel:.2e}")
    return rel

# ============================================================================
# FIGURES
# ============================================================================

def save_figures(phantom, recon_single, recon_two, ierrs_single, ierrs_two):
    nz, ny, nx = phantom.shape
    z_mid, y_mid, x_mid = nz // 2, ny // 2, nx // 2
    vmin, vmax = 0.0, float(phantom.max()) * 1.05

    # Row 1: ground truth.  Row 2: single-channel.  Row 3: two-channel.
    # Diff row intentionally omitted (previous layout kept below, commented,
    # in case we want it back).
    rows = [
        ("ground truth", phantom),
        ("single-channel", recon_single),
        ("two-channel",    recon_two),
    ]
    views = [
        ("xy", lambda v: v[z_mid, :, :]),
        ("xz", lambda v: v[:, y_mid, :]),
        ("yz", lambda v: v[:, :, x_mid]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row, (row_label, vol) in enumerate(rows):
        for col, (view_name, slicer) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(slicer(vol), cmap="gray", origin="lower",
                      vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{row_label} ({view_name})")
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "victre_single_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- previous diff row (kept commented for quick restoration) ---
    # diff = recon_single - phantom
    # dmax = float(np.abs(diff).max())
    # axes[2, col].imshow(slicer(diff), cmap="seismic", origin="lower",
    #                     vmin=-dmax, vmax=dmax, aspect="auto")
    # axes[2, col].set_title(f"diff ({view_name})")

    fig2, ax = plt.subplots(figsize=(6, 4))
    iters_s = range(1, len(ierrs_single) + 1)
    iters_t = range(1, len(ierrs_two) + 1)
    ax.semilogy(iters_s, ierrs_single, "r-", linewidth=1.5, label="single-channel")
    ax.semilogy(iters_t, ierrs_two,    "b-", linewidth=1.5, label="two-channel")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.set_title("CBCT-LAR on VICTRE phantom: single vs two channel")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    fig2.savefig(FIGURE_DIR / "victre_single_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Figures written under {FIGURE_DIR}/")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("VICTRE single-channel + two-channel CBCT-LAR reconstruction")
    print("=" * 70)

    phantom, dx_cm = load_and_downsample_phantom()
    vol_geom, proj_geom, geom_info = build_geometry(phantom.shape, dx_cm)
    A, At = make_projector(vol_geom, proj_geom)

    R_hi, R_lo = build_sinogram_filters(
        geom_info["det_col_count"], geom_info["det_spacing"],
        CONFIG["cutoffparm"], CONFIG["cutoffparm_lo"],
    )

    # Quick adjoint sanity check before we invest in the main loop
    sino_shape = (geom_info["det_row_count"],
                  geom_info["nviews"],
                  geom_info["det_col_count"])
    adjoint_test(A, At, phantom.shape, sino_shape)

    nusino, nuxgrad, nuygrad, nuzgrad = operator_norms(
        phantom.shape, A, At, CONFIG["npower"]
    )

    recon_single, ierrs_single, derrs_single, tvs_single = run_single_channel(
        phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad, geom_info["nrays"]
    )

    recon_two, ierrs_two, derrs_two, tvs_two = run_two_channel(
        phantom, A, At, R_hi, R_lo,
        nusino, nuxgrad, nuygrad, nuzgrad, geom_info["nrays"]
    )

    save_figures(phantom, recon_single, recon_two, ierrs_single, ierrs_two)

    out = {
        "phantom": phantom,
        "recon_single": recon_single,
        "recon_two":    recon_two,
        "ierrs_single": ierrs_single,
        "ierrs_two":    ierrs_two,
        "derrs_single": derrs_single,
        "derrs_two":    derrs_two,
        "tvs_single":   tvs_single,
        "tvs_two":      tvs_two,
        "config":   CONFIG,
        "geometry": geom_info,
        "dx_cm":    dx_cm,
    }
    with open(CACHE_DIR / "victre_single_channel.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"Saved: {CACHE_DIR / 'victre_single_channel.pkl'}")

if __name__ == "__main__":
    main()
