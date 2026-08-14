"""Self-contained 3D cone-beam (CBCT) reconstruction: single- vs two-channel
DTV Chambolle-Pock with CERTIFIED step sizes. 3D analog of paper/reconstruction.py.

The two-channel step sizes are set from the sharp condition tau*lambda_max(M) < 1,
where M = sum_j sigma_j K_j^T K_j is the weighted normal operator (band filters
entering as F^2, with sigma_lo baked in). This replaces the conservative
sqrt(sigma_lo_scale) norm inflation the earlier 3D code used, which throttled tau
by a fixed 2x regardless of how disjoint the low band actually is.

Cone-beam projector via ASTRA (GPU). Sinogram filters are pure sqrt-Hanning along
the detector u-axis (no ramp; the ASTRA projector is unfiltered). Geometry and CP
iteration are ported from presentation/src/victre_reconstruction.py.
"""
import time

import numpy as np
from numpy import maximum, sqrt
import astra

DEFAULTS = dict(
    beta=5.0, rho=1.75, eps=0.001, l1f=1.0, stepbalance=100.0,
    nuxfact=0.5, nuyfact=0.5, nuzfact=0.5, tv_cap=1.0,
    cutoffparm=4.0, cutoffparm_lo=8.0,
    sigma_lo_scale=4.0, eps_hi_ratio=1.0, eps_lo_ratio=1.25,
    itermax=300, npower=100, cvg_slack=1e-3,
    # Power-iteration inits. Left unseeded, every run drew different start
    # vectors, converged to slightly different operator norms, and therefore ran
    # with different tau/sigma -- which made 3D results irreproducible
    # run-to-run. Seeded per solver call via _rng below.
    seed=42,
    # Poisson transmission noise. i0 = incident photons per ray; 0 disables
    # noise entirely (the historical behaviour). noise_seed indexes the noise
    # REALIZATION and is deliberately independent of `seed`: statistics are
    # gathered by varying noise_seed at fixed seed, so that spread across runs
    # reflects measurement noise and not step-size jitter. Conflating the two
    # would make every error bar uninterpretable.
    i0=0.0,
    noise_seed=0,
    # With noisy data the truth is no longer exactly consistent, so the data
    # tolerance must admit it. "noise" sets eps from an independent calibration
    # draw; "fixed" keeps cfg["eps"] as given. See calibrate_eps.
    eps_mode="fixed",
    eps_factor=1.1,
)


def _rng(cfg, stream):
    """Independent generator per power iteration.

    Deliberately NOT one shared generator: a shared stream would make each
    norm estimate depend on how many draws happened before it, so adding or
    reordering a power iteration would silently change every later result.
    Keying on a stream name makes each estimate independent of the others.
    """
    seed = cfg.get("seed", 42)
    if seed is None:                       # explicit opt-out, non-reproducible
        return np.random.default_rng()
    mix = int.from_bytes(stream.encode(), "little") % (2 ** 31)
    return np.random.default_rng((int(seed) * 1_000_003 + mix) % (2 ** 63))
ISTOPS = (1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500)


# --------------------------------------------------------------------------
# Poisson transmission noise
# --------------------------------------------------------------------------
def poisson_sinogram(clean, i0, rng):
    """Monoenergetic transmission noise on line integrals.

        N ~ Poisson(i0 * exp(-p)),   p_hat = -log(N / i0)

    `clean` holds the ideal line integrals p; the return is p_hat. Zero counts
    are clamped to 1 photon before the log -- exact zeros would give p_hat=inf
    and poison the whole sinogram. The clamp biases the very few most-attenuated
    rays slightly low; at sane i0 for breast DBT it fires on a negligible
    fraction, and `reconstruct` reports that fraction so the choice of i0 can be
    sanity-checked rather than assumed.
    """
    expected = i0 * np.exp(-np.asarray(clean, dtype=np.float64))
    counts = rng.poisson(expected)
    n_zero = int((counts == 0).sum())
    counts = np.maximum(counts, 1)
    noisy = -np.log(counts / i0)
    return noisy.astype(np.float32), n_zero


def calibrate_eps(clean, i0, R_single, nrays, rng, factor=1.1):
    """Data tolerance matched to the noise level, from an INDEPENDENT draw.

    With noisy data the true image no longer satisfies ||F(Af - g)|| = 0; it
    satisfies it only to about the filtered noise norm. If eps stays at its
    noiseless value the constraint set excludes the truth, the solver is pushed
    somewhere else entirely, and the single-vs-two comparison stops meaning
    anything. So eps has to track the noise.

    The calibration draw is independent of the data draw on purpose. Setting eps
    from the realized noise would leak knowledge of the actual realization into
    the reconstruction -- a small inverse crime, and one that would make the
    constraint suspiciously well-matched on every run. `factor` > 1 buys margin
    so the truth is comfortably inside the set rather than on its boundary.
    """
    cal, _ = poisson_sinogram(clean, i0, rng)
    resid = R_single(cal - clean)
    return factor * float(sqrt((resid * resid).sum())) / sqrt(nrays)


# --------------------------------------------------------------------------
# Geometry and projector (circular cone-beam)
# --------------------------------------------------------------------------
def cone_geometry(shape, dx_cm, det_rows=256, det_cols=256, det_spacing=0.05,
                  nviews=25, arc_deg=50.0, sod=50.0, odd=50.0):
    nz, ny, nx = shape
    vol_geom = astra.create_vol_geom(
        ny, nx, nz,
        -nx * dx_cm / 2, nx * dx_cm / 2, -ny * dx_cm / 2, ny * dx_cm / 2,
        -nz * dx_cm / 2, nz * dx_cm / 2)
    angles = np.deg2rad(np.linspace(-arc_deg / 2, arc_deg / 2, nviews))
    proj_geom = astra.create_proj_geom(
        "cone", det_spacing, det_spacing, det_rows, det_cols, angles, sod, odd)
    info = dict(nviews=nviews, arc_deg=arc_deg, det_rows=det_rows,
                det_cols=det_cols, det_spacing=det_spacing,
                nrays=nviews * det_rows * det_cols, sod=sod, odd=odd)
    print(f"Cone-beam: {nviews} views x {det_rows}x{det_cols} det @ "
          f"{det_spacing*10:.2f} mm, arc {arc_deg} deg, SOD/ODD {sod}/{odd} cm")
    return vol_geom, proj_geom, info


def dbt_geometry(shape, dx_cm, det_rows=384, det_cols=384, det_spacing=0.05,
                 nviews=25, arc_deg=50.0, sod=65.0, odd=5.0):
    """DBT-style limited-angle cone-beam: source arcs in the y-z plane, flat
    detector fixed below the volume in the x-y plane. z is the compression
    (depth) direction -- the one limited-angle DBT resolves poorly."""
    nz, ny, nx = shape
    vol_geom = astra.create_vol_geom(
        ny, nx, nz,
        -nx * dx_cm / 2, nx * dx_cm / 2, -ny * dx_cm / 2, ny * dx_cm / 2,
        -nz * dx_cm / 2, nz * dx_cm / 2)
    angles = np.deg2rad(np.linspace(-arc_deg / 2, arc_deg / 2, nviews))
    vectors = np.zeros((nviews, 12), dtype=np.float64)
    for i, th in enumerate(angles):
        vectors[i, 0:3] = (0.0, sod * np.sin(th), sod * np.cos(th))   # source
        vectors[i, 3:6] = (0.0, 0.0, -odd)                            # det centre
        vectors[i, 6:9] = (det_spacing, 0.0, 0.0)                     # u along x
        vectors[i, 9:12] = (0.0, det_spacing, 0.0)                    # v along y
    proj_geom = astra.create_proj_geom("cone_vec", det_rows, det_cols, vectors)
    info = dict(nviews=nviews, arc_deg=arc_deg, det_rows=det_rows,
                det_cols=det_cols, det_spacing=det_spacing,
                nrays=nviews * det_rows * det_cols, sod=sod, odd=odd, orbit="dbt")
    print(f"DBT cone-beam: {nviews} views x {det_rows}x{det_cols} det @ "
          f"{det_spacing*10:.2f} mm, arc {arc_deg} deg, SOD/ODD {sod}/{odd} cm")
    return vol_geom, proj_geom, info


def make_projector(vol_geom, proj_geom):
    def A(vol):
        sid, sino = astra.create_sino3d_gpu(
            np.ascontiguousarray(vol, np.float32), proj_geom, vol_geom)
        astra.data3d.delete(sid)
        return sino

    def At(sino):
        vid, vol = astra.create_backprojection3d_gpu(
            np.ascontiguousarray(sino, np.float32), proj_geom, vol_geom)
        astra.data3d.delete(vid)
        return vol
    return A, At


# --------------------------------------------------------------------------
# Finite-difference gradients (forward diff, Dirichlet upper boundary)
# --------------------------------------------------------------------------
def gradx(v): g = -v.copy(); g[:, :, :-1] += v[:, :, 1:]; return g
def mdivx(g): d = -g.copy(); d[:, :, 1:] += g[:, :, :-1]; return d
def grady(v): g = -v.copy(); g[:, :-1, :] += v[:, 1:, :]; return g
def mdivy(g): d = -g.copy(); d[:, 1:, :] += g[:, :-1, :]; return d
def gradz(v): g = -v.copy(); g[:-1, :, :] += v[1:, :, :]; return g
def mdivz(g): d = -g.copy(); d[1:, :, :] += g[:-1, :, :]; return d


# --------------------------------------------------------------------------
# Sinogram band filters (sqrt-Hanning along detector u-axis = last axis)
# --------------------------------------------------------------------------
def _hann_axis(n, ds, c):
    blen = n * ds
    b00 = -blen / 2.0
    uar = np.arange(b00 + ds / 2.0, b00 + blen, ds)
    uhanp = abs(b00) / c
    h = 0.5 * (1.0 + np.cos(np.pi * uar / uhanp))
    h[np.abs(uar) > uhanp] = 0.0
    return np.clip(h, 0.0, 1.0)


def _uar(n, ds):
    blen = n * ds
    b00 = -blen / 2.0
    return np.arange(b00 + ds / 2.0, b00 + blen, ds)


def band_filters(det_cols, det_spacing, cutoffparm, cutoffparm_lo,
                 axis="u", det_rows=None, ramp=True):
    """Data-fidelity filters on the 3D sinogram (det_row, view, det_col).

    Each filter is  sqrt(ramp) * Hann^{1/2}, matching the 2D solver
    (paper/reconstruction.py): F_single = sqrt(ramp), F_hi = sqrt(ramp)*sqrt(han_hi),
    F_lo = sqrt(ramp)*sqrt(han_lo). The Hann^{1/2} splits the band; the sqrt-ramp
    suppresses the near-DC singular values (what makes the low band disjoint).
    ramp=False drops it (pure Hann^{1/2}, the previous 3D behaviour).

    axis='u': 1D along the detector u-axis (det_col); ramp = sqrt(|u|).
    axis='2d': separable 2D over (v, u); ramp = (u^2+v^2)^{1/4} (radial sqrt-ramp).
    Returns R_single, R_hi, R_lo, R_single_sq, R_data_sq.
    """
    if axis == "u":
        uu = _uar(det_cols, det_spacing)
        Wr = np.sqrt(np.abs(uu)) if ramp else np.ones_like(uu)
        han_lo = _hann_axis(det_cols, det_spacing, cutoffparm_lo)
        han_hi = np.clip(1.0 - _hann_axis(det_cols, det_spacing, cutoffparm), 0.0, 1.0)
        F_s, F_lo, F_hi = Wr, Wr * np.sqrt(han_lo), Wr * np.sqrt(han_hi)
        W_s = np.fft.fftshift(F_s)
        W_lo = np.fft.fftshift(F_lo)
        W_hi = np.fft.fftshift(F_hi)
        axes = (-1,)
    elif axis == "2d":
        if det_rows is None:
            raise ValueError("axis='2d' requires det_rows")
        uu = _uar(det_cols, det_spacing)
        vv = _uar(det_rows, det_spacing)
        Uv, Uu = np.meshgrid(vv, uu, indexing="ij")
        Wr = (Uv ** 2 + Uu ** 2) ** 0.25 if ramp else np.ones((det_rows, det_cols))
        H_lo = np.outer(_hann_axis(det_rows, det_spacing, cutoffparm_lo),
                        _hann_axis(det_cols, det_spacing, cutoffparm_lo))
        H_hi = np.clip(1.0 - np.outer(_hann_axis(det_rows, det_spacing, cutoffparm),
                                      _hann_axis(det_cols, det_spacing, cutoffparm)),
                       0.0, 1.0)
        F_s = Wr
        F_lo = Wr * np.sqrt(np.clip(H_lo, 0.0, 1.0))
        F_hi = Wr * np.sqrt(H_hi)
        sh = (det_rows, 1, det_cols)
        W_s = np.fft.fftshift(F_s).reshape(sh)
        W_lo = np.fft.fftshift(F_lo).reshape(sh)
        W_hi = np.fft.fftshift(F_hi).reshape(sh)
        axes = (0, 2)
    else:
        raise ValueError(f"axis must be 'u' or '2d', got {axis!r}")
    W_s2, W_lo2, W_hi2 = W_s ** 2, W_lo ** 2, W_hi ** 2

    def _apply(s, W):
        return np.fft.ifftn(np.fft.fftn(s, axes=axes) * W, axes=axes).real.astype(np.float32)

    def R_single(s): return _apply(s, W_s)
    def R_hi(s): return _apply(s, W_hi)
    def R_lo(s): return _apply(s, W_lo)
    def R_single_sq(s): return _apply(s, W_s2)              # for nusino: A^T R_s^2 A

    def R_data_sq(s, r_lo):
        """F_hi^2 + r_lo F_lo^2 in one transform (for the normal operator)."""
        return _apply(s, W_hi2 + r_lo * W_lo2)
    print(f"Sinogram filters (axis={axis}, ramp={ramp}): cutoff_hi={cutoffparm}, "
          f"cutoff_lo={cutoffparm_lo}")
    return R_single, R_hi, R_lo, R_single_sq, R_data_sq


# --------------------------------------------------------------------------
# Operator norms / certified step sizes
# --------------------------------------------------------------------------
def block_norms(shape, A, At, R_single_sq, cfg):
    npw = cfg["npower"]
    # nusino normalizes the ramp-filtered projector: 1/||R_single A|| via
    # power iteration on A^T R_single^2 A.
    x = _rng(cfg, "nusino").standard_normal(shape, dtype=np.float32)
    xn = 1.0
    for _ in range(npw):
        x = At(R_single_sq(A(x))); xn = sqrt((x * x).sum()) + 1e-12; x /= xn
    nusino = 1.0 / sqrt(xn)

    def axis_norm(grad, div, name):
        x = _rng(cfg, f"axis_{name}").standard_normal(shape, dtype=np.float32)
        n = 1.0
        for _ in range(max(npw // 4, 10)):
            x = div(grad(x)); n = sqrt((x * x).sum()) + 1e-12; x /= n
        return sqrt(n)
    nux = cfg["nuxfact"] / axis_norm(gradx, mdivx, "x")
    nuy = cfg["nuyfact"] / axis_norm(grady, mdivy, "y")
    nuz = cfg["nuzfact"] / axis_norm(gradz, mdivz, "z")
    print(f"||R_single A||={1/nusino:.4f}  nusino={nusino:.6f}  "
          f"nux={nux:.6f} nuy={nuy:.6f} nuz={nuz:.6f}")
    return nusino, nux, nuy, nuz


def single_norm(shape, A, At, R_single, nusino, nux, nuy, nuz, cfg):
    """||K|| for the single-channel stacked operator [R_single A; dx; dy; dz; l1]."""
    l1f = cfg["l1f"]
    x = _rng(cfg, "single_norm").standard_normal(shape, dtype=np.float32)
    x /= sqrt((x * x).sum()) + 1e-12
    mag1 = mag2 = 0.0
    for _ in range(cfg["npower"]):
        s = R_single(A(x)) * nusino
        gx, gy, gz = gradx(x) * nux, grady(x) * nuy, gradz(x) * nuz
        yl = l1f * x
        mag1 = sqrt((s*s).sum() + (gx*gx).sum() + (gy*gy).sum()
                    + (gz*gz).sum() + (yl*yl).sum())
        if mag1 > 0:
            s /= mag1; gx /= mag1; gy /= mag1; gz /= mag1; yl /= mag1
        x = (At(R_single(s)) * nusino + mdivx(gx) * nux + mdivy(gy) * nuy
             + mdivz(gz) * nuz + l1f * yl)
        mag2 = sqrt((x * x).sum()) + 1e-12; x /= mag2
    return 0.5 * (mag1 + mag2)


def certified_norm_two(shape, A, At, R_data_sq, nusino, nux, nuy, nuz, cfg):
    """sqrt(lambda_max(Mtilde)) for the sig_hi-factored weighted normal operator
        Mtilde = nusino^2 A^T (F_hi^2 + r_lo F_lo^2) A
                 + nux^2 dx^Tdx + nuy^2 dy^Tdy + nuz^2 dz^Tdz + l1^2 I,
    with r_lo = sigma_lo_scale. Symmetric power iteration (Rayleigh quotient);
    the band filters enter squared (via R_data_sq), i.e. the true X^T R^2 X."""
    r_lo = cfg["sigma_lo_scale"]
    l1sq = cfg["l1f"] ** 2
    nus2, nux2, nuy2, nuz2 = nusino**2, nux**2, nuy**2, nuz**2
    v = _rng(cfg, "certified_norm_two").standard_normal(shape, dtype=np.float32)
    v /= sqrt((v * v).sum()) + 1e-12
    # Run to a tolerance rather than a fixed count. This estimate is what the
    # certificate rests on, and the Rayleigh quotient converges from BELOW --
    # it underestimates lambda_max, which is the unsafe direction: too few
    # iterations and tau*lambda_max(M) silently exceeds 1. Since lambda = norm^2,
    # a relative error d in the returned norm is 2d in lambda, so the tolerance
    # has to be comfortably tighter than cvg_slack/2 for the margin to hold.
    # CAUTION: this tolerance is on the per-iteration CHANGE, which is not the
    # error. Power iteration converges linearly at rate (lam2/lam1)^k, so with a
    # small spectral gap the increment can look tiny while the estimate is still
    # far off -- measured here at 1e-5 increment but 0.3% disagreement between
    # seeds. The tolerance is therefore set tight and acts as a plateau guard,
    # not as a convergence proof. The authoritative check is the seed-spread
    # diagnostic in check_power_convergence.py; use it to pick npower per
    # geometry, and treat that npower as the real setting.
    tol = cfg.get("npower_tol", 1e-7)
    nmax = max(int(cfg["npower"]), 40)
    lam = lam_prev = 0.0
    rel = float("inf")            # tracked INSIDE the loop: a run that exhausts
    it = 0                        # nmax must report its last real change, not 0
    converged = False
    for it in range(1, nmax + 1):
        Mv = nus2 * At(R_data_sq(A(v), r_lo))
        Mv = Mv + nux2 * mdivx(gradx(v)) + nuy2 * mdivy(grady(v)) \
            + nuz2 * mdivz(gradz(v)) + l1sq * v
        lam = float((v * Mv).sum())
        nv = sqrt((Mv * Mv).sum()) + 1e-12
        v = (Mv / nv).astype(np.float32)
        if lam > 0:
            rel = abs(lam - lam_prev) / lam
        lam_prev = lam
        if it > 10 and rel < tol:
            converged = True
            break
    slack = cfg.get("cvg_slack", 1e-3)
    print(f"[two] power iteration: {it} its, lam={lam:.6f}, rel. change={rel:.2e} "
          f"(tol {tol:.1e}, {'converged' if converged else 'ran to cap'})")
    # lambda enters the certificate directly, and a relative error d in lambda
    # eats d of the cvg_slack safety margin. Rayleigh quotients converge from
    # below, so an unconverged estimate UNDERstates lambda_max -- the unsafe
    # direction, since tau is then set too large.
    #
    # Key on `rel`, NOT on whether the early exit fired. Running to the cap with
    # a tiny per-iteration change is fine (it plateaued); it is a LARGE change
    # that is dangerous. Keying on the flag made this warn spuriously whenever a
    # caller disabled the early exit by passing npower_tol=0.
    if rel > slack / 2:
        print(f"  WARNING: lambda_max uncertain to {rel:.2e} against "
              f"cvg_slack={slack:.1e}. tau*lambda_max(M)<1 may not actually "
              f"hold -- power iteration underestimates lambda_max, so tau is "
              f"set too large. Raise npower (cap is {nmax}) or cvg_slack.")
    return sqrt(max(lam, 0.0))


# --------------------------------------------------------------------------
# CP solvers
# --------------------------------------------------------------------------
def _tv_prox(yg, grad_xbar, sig, cap):
    p = yg + sig * grad_xbar
    return cap * p / maximum(np.abs(p), cap)


def solve_single(phantom, A, At, R_single, norms, cfg, sinodata=None,
                 snapshot_iters=None):
    nusino, nux, nuy, nuz = norms
    nrays = cfg["_nrays"]
    print("\n[single] joint operator norm...")
    norm = single_norm(phantom.shape, A, At, R_single, nusino, nux, nuy, nuz, cfg)
    sig = cfg["stepbalance"] / norm
    tau = 1.0 / (norm * cfg["stepbalance"] * (1.0 + cfg["cvg_slack"]))
    print(f"[single] ||K||={norm:.4f}  sig={sig:.5f}  tau={tau:.6f}  "
          f"tau*sig*||K||^2={tau*sig*norm**2:.4f}")
    epssc = cfg["eps"] * sqrt(nrays)
    truesino = A(phantom) if sinodata is None else sinodata
    sinodatasc = nusino * R_single(truesino)
    return _cp_single(phantom, A, At, R_single, nusino, nux, nuy, nuz, sig, tau,
                      epssc, sinodatasc, nrays, cfg, snapshot_iters)


def solve_two(phantom, A, At, R_hi, R_lo, R_data_sq, norms, cfg,
              sinodata=None, snapshot_iters=None):
    nusino, nux, nuy, nuz = norms
    nrays = cfg["_nrays"]
    print("\n[two] certified weighted normal-operator norm...")
    normw = certified_norm_two(phantom.shape, A, At, R_data_sq,
                               nusino, nux, nuy, nuz, cfg)
    sig_hi = cfg["stepbalance"] / normw
    sig_lo = cfg["sigma_lo_scale"] * sig_hi
    tau = 1.0 / (normw * cfg["stepbalance"] * (1.0 + cfg["cvg_slack"]))
    cp = tau * sig_hi * normw ** 2
    print(f"[two] sqrt(lam_max(M))={normw:.4f}  sig_hi={sig_hi:.5f}  "
          f"sig_lo={sig_lo:.5f}  tau={tau:.6f}  tau*lam_max(M)={cp:.4f}")
    epssc_hi = cfg["eps_hi_ratio"] * cfg["eps"] * sqrt(nrays)
    epssc_lo = cfg["eps_lo_ratio"] * cfg["eps"] * sqrt(nrays)
    truesino = A(phantom) if sinodata is None else sinodata
    return _cp_two(phantom, A, At, R_hi, R_lo, nusino, nux, nuy, nuz,
                   sig_hi, sig_lo, tau, epssc_hi, epssc_lo,
                   nusino * R_hi(truesino), nusino * R_lo(truesino),
                   nrays, cfg, snapshot_iters)


def _cp_single(phantom, A, At, R_single, nusino, nux, nuy, nuz, sig, tau, epssc,
               sinodatasc, nrays, cfg, snapshot_iters):
    rho, beta, cap, l1f = cfg["rho"], cfg["beta"], cfg["tv_cap"], cfg["l1f"]
    xim = np.zeros_like(phantom); ysino = np.zeros_like(sinodatasc)
    ygx = np.zeros_like(phantom); ygy = np.zeros_like(phantom); ygz = np.zeros_like(phantom)
    yim = np.zeros_like(phantom)
    ierrs, snaps = [], {}
    snap_set = set(snapshot_iters or [])
    nv = phantom.size; t0 = time.time()
    for itr in range(1, cfg["itermax"] + 1):
        ys_o, gx_o, gy_o, gz_o, yi_o, x_o = (ysino.copy(), ygx.copy(), ygy.copy(),
                                             ygz.copy(), yim.copy(), xim.copy())
        xim = xim - tau * (At(R_single(ysino)) * nusino + mdivx(ygx) * nux
                           + mdivy(ygy) * nuy + mdivz(ygz) * nuz + l1f * yim)
        xim[xim < 0.0] = 0.0
        xbar = 2.0 * xim - x_o
        resid = R_single(A(xbar)) * nusino - sinodatasc
        ysino = ysino + sig * resid
        ymag = sqrt((ysino * ysino).sum()); slack = sig * nusino * epssc
        ysino = ysino * ((ymag - slack) / ymag) if ymag - slack > 0 else ysino * 0.0
        ygx = _tv_prox(ygx, gradx(xbar) * nux, sig, cap)
        ygy = _tv_prox(ygy, grady(xbar) * nuy, sig, cap)
        ygz = _tv_prox(ygz, gradz(xbar) * nuz, sig, cap)
        p1 = yim + sig * (l1f * xbar); yim = beta * p1 / maximum(np.abs(p1), beta)
        ygx = gx_o - rho * (gx_o - ygx); ygy = gy_o - rho * (gy_o - ygy)
        ygz = gz_o - rho * (gz_o - ygz); ysino = ys_o - rho * (ys_o - ysino)
        yim = yi_o - rho * (yi_o - yim); xim = x_o - rho * (x_o - xim)
        ierrs.append(float(sqrt(((xbar - phantom) ** 2).sum() / nv)))
        if itr in snap_set:
            snaps[itr] = xbar.copy()
        if itr in ISTOPS or itr == cfg["itermax"]:
            print(f"  [single] iter {itr:4d}: img_err={ierrs[-1]:.6f} "
                  f"({time.time()-t0:.1f}s)")
    return dict(recon=xbar.copy(), ierrs=ierrs, snaps=snaps)


def _cp_two(phantom, A, At, R_hi, R_lo, nusino, nux, nuy, nuz,
            sig_hi, sig_lo, tau, epssc_hi, epssc_lo, sdsc_hi, sdsc_lo,
            nrays, cfg, snapshot_iters):
    rho, beta, cap, l1f = cfg["rho"], cfg["beta"], cfg["tv_cap"], cfg["l1f"]
    sig_b = sig_hi
    xim = np.zeros_like(phantom)
    yhi = np.zeros_like(sdsc_hi); ylo = np.zeros_like(sdsc_lo)
    ygx = np.zeros_like(phantom); ygy = np.zeros_like(phantom); ygz = np.zeros_like(phantom)
    yim = np.zeros_like(phantom)
    ierrs, snaps = [], {}
    snap_set = set(snapshot_iters or [])
    nv = phantom.size; t0 = time.time()
    for itr in range(1, cfg["itermax"] + 1):
        yh_o, yl_o, gx_o, gy_o, gz_o, yi_o, x_o = (
            yhi.copy(), ylo.copy(), ygx.copy(), ygy.copy(), ygz.copy(),
            yim.copy(), xim.copy())
        xim = xim - tau * (At(R_hi(yhi) + R_lo(ylo)) * nusino
                           + mdivx(ygx) * nux + mdivy(ygy) * nuy
                           + mdivz(ygz) * nuz + l1f * yim)
        xim[xim < 0.0] = 0.0
        xbar = 2.0 * xim - x_o
        Ax = A(xbar) * nusino
        rhi = R_hi(Ax) - sdsc_hi; rlo = R_lo(Ax) - sdsc_lo
        yhi = yhi + sig_hi * rhi
        ymh = sqrt((yhi * yhi).sum()); sh = sig_hi * nusino * epssc_hi
        yhi = yhi * ((ymh - sh) / ymh) if ymh - sh > 0 else yhi * 0.0
        ylo = ylo + sig_lo * rlo
        yml = sqrt((ylo * ylo).sum()); sl = sig_lo * nusino * epssc_lo
        ylo = ylo * ((yml - sl) / yml) if yml - sl > 0 else ylo * 0.0
        ygx = _tv_prox(ygx, gradx(xbar) * nux, sig_b, cap)
        ygy = _tv_prox(ygy, grady(xbar) * nuy, sig_b, cap)
        ygz = _tv_prox(ygz, gradz(xbar) * nuz, sig_b, cap)
        p1 = yim + sig_b * (l1f * xbar); yim = beta * p1 / maximum(np.abs(p1), beta)
        ygx = gx_o - rho * (gx_o - ygx); ygy = gy_o - rho * (gy_o - ygy)
        ygz = gz_o - rho * (gz_o - ygz)
        yhi = yh_o - rho * (yh_o - yhi); ylo = yl_o - rho * (yl_o - ylo)
        yim = yi_o - rho * (yi_o - yim); xim = x_o - rho * (x_o - xim)
        ierrs.append(float(sqrt(((xbar - phantom) ** 2).sum() / nv)))
        if itr in snap_set:
            snaps[itr] = xbar.copy()
        if itr in ISTOPS or itr == cfg["itermax"]:
            print(f"  [two] iter {itr:4d}: img_err={ierrs[-1]:.6f} "
                  f"({time.time()-t0:.1f}s)")
    return dict(recon=xbar.copy(), ierrs=ierrs, snaps=snaps)


def reconstruct(phantom, dx_cm, cfg=None, geom=None, snapshot_iters=None):
    """Run single + certified two-channel on a 3D phantom. Returns both results."""
    cfg = {**DEFAULTS, **(cfg or {})}
    geom = dict(geom or {})
    orbit = geom.pop("orbit", "cone")
    geom_fn = dbt_geometry if orbit == "dbt" else cone_geometry
    vol_geom, proj_geom, info = geom_fn(phantom.shape, dx_cm, **geom)
    cfg["_nrays"] = info["nrays"]
    A, At = make_projector(vol_geom, proj_geom)
    filter_axis = cfg.get("filter_axis", "2d" if orbit == "dbt" else "u")
    R_single, R_hi, R_lo, R_single_sq, R_data_sq = band_filters(
        info["det_cols"], info["det_spacing"], cfg["cutoffparm"],
        cfg["cutoffparm_lo"], axis=filter_axis, det_rows=info["det_rows"],
        ramp=cfg.get("ramp", True))
    norms = block_norms(phantom.shape, A, At, R_single_sq, cfg)

    # Build the measured sinogram ONCE and hand the same array to both arms.
    # Previously each arm called A(phantom) for itself, which is harmless when
    # the data is noiseless but would give the two arms DIFFERENT noise
    # realizations -- destroying the paired comparison that makes the statistics
    # tight, and adding a spurious between-arm difference that has nothing to do
    # with the method.
    clean = A(phantom)
    i0 = float(cfg.get("i0", 0.0) or 0.0)
    noise_info = dict(i0=i0, noise_seed=cfg.get("noise_seed", 0))
    if i0 > 0:
        nrng = np.random.default_rng(
            (int(cfg.get("noise_seed", 0)) * 2_654_435_761 + 12345) % (2 ** 63))
        sinodata, n_zero = poisson_sinogram(clean, i0, nrng)
        frac_zero = n_zero / clean.size
        resid = R_single(sinodata - clean)
        noise_info.update(zero_count_frac=frac_zero,
                          filtered_noise_norm=float(sqrt((resid * resid).sum())))
        print(f"Poisson noise: i0={i0:.3g}, realization seed="
              f"{cfg.get('noise_seed', 0)}, zero-count rays {frac_zero:.2e}")
        if frac_zero > 1e-3:
            print(f"  WARNING: {frac_zero:.2%} of rays recorded zero photons and "
                  f"were clamped to 1. i0 is too low for this object; the "
                  f"clamp biases those line integrals low.")
        if cfg.get("eps_mode", "fixed") == "noise":
            # Independent stream from the data draw -- see calibrate_eps.
            crng = np.random.default_rng(
                (int(cfg.get("noise_seed", 0)) * 40_503 + 777) % (2 ** 63))
            cfg["eps"] = calibrate_eps(clean, i0, R_single, info["nrays"], crng,
                                       factor=cfg.get("eps_factor", 1.1))
            print(f"  eps calibrated to noise: {cfg['eps']:.6g}")
        noise_info["eps"] = cfg["eps"]
    else:
        sinodata = clean

    single = solve_single(phantom, A, At, R_single, norms, cfg,
                          sinodata=sinodata, snapshot_iters=snapshot_iters)
    two = solve_two(phantom, A, At, R_hi, R_lo, R_data_sq, norms, cfg,
                    sinodata=sinodata, snapshot_iters=snapshot_iters)
    return dict(single=single, two=two, info=info, noise=noise_info)
