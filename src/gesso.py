"""GESSO: background-image estimation on top of the two-channel fidelity.

Codename only, for internal use. "Gesso" is the smooth ground layer a painter
lays down before the image goes on, which is what the extra image variable is:
a smooth underlayer that absorbs low-frequency content so the layer above it
does not have to carry it.

The method is Eq. (2) of Sidky et al., "Improving depth-resolution, in-plane
contrast, and reducing non-uniformity artifacts for wide-angle DBT"
(arXiv:2605.29133v2), adapted from their 3D DBT model to our 2D fan-beam one.
Three images are reconstructed at once: the raw image f1, a "background" image
f2 meant to collect low-frequency artifacts, and their difference f3 = f1 - f2,
which carries an l1 penalty. f2 is held to a data constraint built from heavily
blurred data through a heavily blurred object model, so the only thing it can
explain is large-scale structure. The l1 penalty on f3 pulls f1 toward f2 where
nothing is in focus, while being cheap enough at large values that real
in-focus features survive.

Two adaptations are forced by the geometry and are flagged here because they
are departures from the paper, not choices:

  * The oblique DTV directions d_a, d_b of their Eq. (2) are perpendicular to
    the first and last source positions in the xz-plane. Our 2D fan-beam model
    has no z, so there is no oblique pair to form; f1 and f2 carry the same
    (d_x, d_y) DTV that the manuscript's own model uses, with the same weights.
  * Their f2 constraint uses a single sqrt-ramp channel. It is left
    single-channel in every arm here, including the two-channel ones, because
    the band split is our contribution to the f1 residual and splitting f2's as
    well would confound the two effects. The arms differ only in how f1's
    residual is treated.

What this module is for is the 2x2 matrix {single, two} x {plain, gesso}. The
two axes are independent: the fidelity axis splits the *residual* into bands
with separate dual step sizes, the gesso axis adds a second *image* variable.
Sidky's own text treats them as separate ideas, citing the two-band fidelity as
a route to better convergence alongside, not as part of, the background
estimation.

READ THIS BEFORE INTERPRETING RESULTS. Our simulated data are generated with
the same X-ray transform used to reconstruct them, perturbed only by Poisson
noise. The artifacts the background image exists to absorb -- imperfect
knowledge of the illumination, scatter, dense tissue outside the viewing plane
-- are model mismatch, and none of them are present here. So a null result on
our phantoms is the expected outcome and is not evidence against the method;
it is evidence that our simulation does not contain its target. The honest test
is `--mismatch`, which injects a smooth multiplicative gain error into the
transmission data before the log, i.e. an illumination error of the kind the
method names. Run both.

Usage:
    python src/gesso.py --phantom breast --itermax 500
    python src/gesso.py --phantom breast --mismatch 0.15 --arms all
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numpy import cos, exp, log, maximum, pi, sqrt, zeros, zeros_like
from scipy import fft as _fft
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry2d import make_fanbeam                       # noqa: E402
from paths import TAB                                     # noqa: E402
from store import RunStore                                # noqa: E402

# Matches reconstruction.py. He-Yuan relaxation and the step balance are the
# manuscript's values and are not tuned per arm.
RHO = 1.75
STEPBALANCE = 100.0
CUTOFF_HI = 4.0
CUTOFF_LO = 8.0
SIGMA_LO_SCALE = 4.0
NPOWER = 200

ARMS = ("single_plain", "two_plain", "single_gesso", "two_gesso")


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
def make_filters(nbins, detectorlength, cutoff_hi=CUTOFF_HI,
                 cutoff_lo=CUTOFF_LO):
    """The sqrt-ramp channel filters of the manuscript.

    Each weight is real and even in detector frequency, so every one of these
    operators is self-adjoint and R^T R = R^2. The PDHG below relies on that.
    """
    db = detectorlength/nbins
    b00 = -detectorlength/2.
    uar = np.arange(b00 + db/2., b00 + detectorlength, db)*1.

    def hann(cut):
        uhanp = abs(b00)/cut
        h = 0.5*(1.0 + cos(pi*uar/uhanp))
        h[abs(uar) > uhanp] = 0.0
        return np.clip(h, 0.0, 1.0)

    w_ramp = sqrt(abs(uar) + 1e-12)
    f_single = w_ramp
    f_lo = w_ramp*sqrt(hann(cutoff_lo))
    f_hi = w_ramp*sqrt(1.0 - hann(cutoff_hi))

    def weight(sino, w):
        ft = _fft.fft(sino, axis=1)
        return _fft.ifft((np.ones([nbins])*_fft.fftshift(w))*ft, axis=1).real

    return (lambda s: weight(s, f_single),
            lambda s: weight(s, f_hi),
            lambda s: weight(s, f_lo))


def blur(arr, sigma, axes=None):
    """Self-adjoint Gaussian blur.

    mode='constant' (zero padding) is required, not cosmetic: a symmetric
    kernel with zero boundary extension is its own adjoint, which the PDHG
    needs. mode='nearest' and mode='reflect' are not self-adjoint and would
    silently break the saddle-point problem being solved.
    """
    if sigma is None or np.all(np.asarray(sigma) <= 0):
        return arr*1.0
    if axes is None:
        return gaussian_filter(arr, sigma, mode="constant", cval=0.0)
    s = [0.0]*arr.ndim
    for a, v in zip(axes, np.atleast_1d(sigma)):
        s[a] = v
    return gaussian_filter(arr, s, mode="constant", cval=0.0)


def grad_mats(n):
    g = zeros([n, n])
    for i in range(n):
        g[i, i] = -1.0
    for i in range(n-1):
        g[i, i+1] = 1.0
    return g


# --------------------------------------------------------------------------
# Problem assembly
# --------------------------------------------------------------------------
class Problem:
    """Operators, data and tolerances shared by all four arms.

    Built once per (phantom, realization) so that the arms are paired on
    exactly the same noise draw, the same calibration draws and the same
    operator norms -- the pairing the manuscript's statistics depend on.
    """

    def __init__(self, phantom, i0=None, noise_seed=0, mismatch=0.0,
                 bg_blur=1.0, bg_blur_z=None, det_blur_mult=1.2,
                 eps_cal_draws=20, eps_cal_k=4.0, alpha=1.9, beta=10.0,
                 nuxfact=0.5, nuyfact=0.5, seed=42):
        self.phantom = phantom = np.ascontiguousarray(phantom, dtype=np.float64)
        nx, ny = phantom.shape
        self.geo = geo = make_fanbeam(nx, ny)
        self.nx, self.ny = nx, ny
        self.mask = geo.mask
        self.alpha, self.beta = alpha, beta
        self.l1f = 1.0

        self.R_s, self.R_hi, self.R_lo = make_filters(
            geo.nbins, _detector_length(geo))

        self.gmatx, self.gmaty = grad_mats(nx), grad_mats(ny)

        # Background blur widths. Sidky asks for in-plane widths "on the order
        # of centimetres" and a projection blur "slightly larger" than those.
        # Both are specified here in field-of-view units and converted, so the
        # setting does not silently change meaning with the image grid.
        self.bg_sigma_pix = bg_blur*nx/geo.ximageside
        mag = 100.0/50.0                       # source_to_detector / radius
        self.det_sigma_bins = (det_blur_mult*bg_blur*mag
                               * geo.nbins/_detector_length(geo))

        rng = np.random.default_rng(seed)
        self._power_seed = seed

        # ---- data ----
        self.truesino = geo.A(phantom)
        # auto_i0 is imported rather than reimplemented: it is the rule the
        # published runs use, and a gesso arm sitting at a different dose than
        # the paper's arms would make the comparison meaningless.
        import noise2d
        self.i0 = float(i0) if i0 is not None else float(
            noise2d.auto_i0(phantom))
        self.noise_seed = noise_seed
        self.mismatch = float(mismatch)

        nrng = np.random.default_rng(
            (int(noise_seed)*2_654_435_761 + 12345) % (2**63))
        counts = nrng.poisson(self.i0*exp(-self.truesino))
        self.zero_frac = float((counts == 0).sum())/counts.size
        trans = maximum(counts, 1)/self.i0
        if self.mismatch > 0:
            trans = trans*self._gain_field(noise_seed)
        self.sinodata = -log(trans)

        # ---- tolerances ----
        self._calibrate(eps_cal_draws, eps_cal_k, noise_seed)

        # ---- operator norms ----
        self.nusino, self.nuxgrad, self.nuygrad = self._op_norms(
            nuxfact, nuyfact, rng)

    # -- data model helpers -------------------------------------------------
    def _gain_field(self, seed):
        """A smooth multiplicative gain error across the detector.

        This is the illumination-error term, the simplest of the three sources
        Sidky names for the background artifact, and the only one expressible
        without a scatter model. It multiplies transmission before the log, so
        it enters the line integrals as a smooth additive offset -- exactly the
        low-frequency, non-uniform background the f2 channel is built to catch.
        It is applied to the data only; the reconstruction operator knows
        nothing about it, so it is genuine model mismatch rather than a
        perturbation both arms could undo.
        """
        g = np.random.default_rng((int(seed)*7919 + 31) % (2**63))
        nv, nb = self.truesino.shape
        field = g.standard_normal((nv, nb))
        field = blur(field, (max(nv/8.0, 1.0), nb/8.0))
        field /= (np.abs(field).max() + 1e-12)
        return 1.0 + self.mismatch*field

    def _calibrate(self, draws, k, noise_seed):
        """Per-band tolerance balls, by the manuscript's mean + k*sd rule.

        The f2 ball is calibrated on the SAME rule applied to the operator f2
        actually sees -- the ramp filter composed with the detector blur. It is
        not a free parameter and it is not inherited from the f1 ball: the blur
        removes most of the noise, so reusing the f1 radius would leave f2 free
        to fit nearly anything smooth.
        """
        if draws < 2:
            raise ValueError("eps_cal_draws must be >= 2; the margin is a "
                             "sample sd, undefined for one draw")
        crng = np.random.default_rng((int(noise_seed)*40_503 + 777) % (2**63))
        ops = {"single": self.R_s, "hi": self.R_hi, "lo": self.R_lo,
               "bg": lambda d: self.R_s(blur(d, self.det_sigma_bins, axes=(1,)))}
        acc = {b: [] for b in ops}
        for _ in range(int(draws)):
            c = -log(maximum(crng.poisson(self.i0*exp(-self.truesino)), 1)
                     / self.i0)
            d = c - self.truesino
            for b, op in ops.items():
                acc[b].append(sqrt((op(d)**2).sum()))
        nrays = self.geo.nrays
        ball = {b: (np.mean(v) + k*np.std(v, ddof=1))/sqrt(nrays)
                for b, v in acc.items()}
        self.eps = ball["single"]
        self.eps_hi, self.eps_lo = ball["hi"], ball["lo"]
        self.eps_bg = ball["bg"]
        self.eps_cal_draws, self.eps_cal_k = draws, k

    def _op_norms(self, nuxfact, nuyfact, rng):
        geo, mask = self.geo, self.mask

        def pnorm(fwd, adj):
            x = rng.standard_normal((self.nx, self.ny))*mask
            n = 0.0
            for _ in range(50):
                x = adj(fwd(x))*mask
                n = sqrt((x**2).sum())
                x /= (n + 1e-12)
            return sqrt(n + 1e-12)

        nsino = 1./pnorm(lambda im: self.R_s(geo.A(im)),
                         lambda s: geo.AT(self.R_s(s)))
        nxg = nuxfact/pnorm(self.gradx, self.mdivx)
        nyg = nuyfact/pnorm(self.grady, self.mdivy)
        return nsino, nxg, nyg

    # -- differential operators --------------------------------------------
    def gradx(self, im):
        return self.gmatx @ im

    def grady(self, im):
        return im @ self.gmaty.T

    def mdivx(self, g):
        return self.gmatx.T @ g

    def mdivy(self, g):
        return g @ self.gmaty


def _detector_length(geo):
    return 2.*np.tan(np.arcsin((geo.ximageside/2.)/50.0))*100.0


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
def solve(prob, arm, itermax=500, alpha3=1.0, sigma_lo_scale=SIGMA_LO_SCALE,
          bg_nonneg=True, snapshot_iters=(), verbose=True):
    """Run one arm of the matrix.

    arm is one of ARMS. 'single'/'two' selects how the f1 residual is treated;
    'plain'/'gesso' selects whether the background image is estimated.

    Returns a dict of traces and, for the gesso arms, the background and
    display images alongside the raw one.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    two = arm.startswith("two")
    bg = arm.endswith("gesso")

    geo, mask = prob.geo, prob.mask
    nx, ny, nrays = prob.nx, prob.ny, prob.geo.nrays
    nu, nxg, nyg = prob.nusino, prob.nuxgrad, prob.nuygrad
    l1f = prob.l1f
    wx, wy, wl1 = 2.0 - prob.alpha, prob.alpha, prob.beta

    # Blurred data for the background channel, and the two residual channels.
    d_bg = prob.R_s(blur(prob.sinodata, prob.det_sigma_bins, axes=(1,)))*nu
    d_hi = prob.R_hi(prob.sinodata)*nu
    d_lo = prob.R_lo(prob.sinodata)*nu
    d_s = prob.R_s(prob.sinodata)*nu

    def A1(im):
        """f1's data operator: projection through the channel filters."""
        s = geo.A(im)
        return ((prob.R_hi(s)*nu, prob.R_lo(s)*nu) if two
                else (prob.R_s(s)*nu,))

    def A1T(ys):
        if two:
            return geo.AT(prob.R_hi(ys[0]) + prob.R_lo(ys[1]))*nu
        return geo.AT(prob.R_s(ys[0]))*nu

    def A2(im):
        """f2's data operator: ramp filter after projecting a blurred image."""
        return prob.R_s(geo.A(blur(im, prob.bg_sigma_pix)))*nu

    def A2T(y):
        return blur(geo.AT(prob.R_s(y)), prob.bg_sigma_pix)*nu

    # ---- step sizes -------------------------------------------------------
    # The manuscript's sharp preconditioned rule: tau * lambda_max(sum_j
    # sigma_j K_j^T K_j) < 1, evaluated by power iteration on the FULL stacked
    # operator, including the background blocks when they are present. Every
    # arm is certified by the same criterion, so a difference between arms is
    # never a difference in how generously each was allowed to step.
    sig_rel = {"hi": 1.0, "lo": sigma_lo_scale if two else 1.0, "bg": 1.0,
               "reg": 1.0, "diff": 1.0}
    lam, lam_spread = _power_stacked(prob, A1, A1T, A2, A2T, two, bg, sig_rel,
                                     wx, wy, wl1, alpha3, l1f)
    tau = 1.0/(lam*STEPBALANCE)
    sig = STEPBALANCE/lam
    sig_hi, sig_lo = sig, sig*sig_rel["lo"]
    # The step sizes sit exactly ON the certified bound: with sigma_j =
    # sig*sig_rel_j and lam the norm of the sqrt(sig_rel)-weighted stacked
    # operator, tau*lambda_max(sum_j sigma_j K_j^T K_j) = tau*sig*lam^2 = 1
    # identically. reconstruction.py does the same thing, so reporting that
    # product would be reporting the number 1. What is worth reporting is
    # whether lam itself is trustworthy, since power iteration approaches the
    # norm FROM BELOW and an underestimate puts the iteration over the true
    # bound. lam_spread is the relative gap between the two magnitude
    # estimates; it going to zero is the evidence that lam has converged.

    eps_sc = {"single": prob.eps*sqrt(nrays), "hi": prob.eps_hi*sqrt(nrays),
              "lo": prob.eps_lo*sqrt(nrays), "bg": prob.eps_bg*sqrt(nrays)}

    if verbose:
        print(f"  [{arm}] lambda_max={lam:.4f} (power spread "
              f"{lam_spread:.1e}) tau={tau:.3e} sig={sig:.3e} "
              f"sig_lo={sig_lo:.3e}")

    # ---- state ------------------------------------------------------------
    f1 = zeros([nx, ny]);  f1bar = zeros([nx, ny])
    f2 = zeros([nx, ny]);  f2bar = zeros([nx, ny])
    yd = [zeros([geo.nviews, geo.nbins]) for _ in range(2 if two else 1)]
    ybg = zeros([geo.nviews, geo.nbins])
    ygx1, ygy1, yl1 = zeros([nx, ny]), zeros([nx, ny]), zeros([nx, ny])
    ygx2, ygy2, yl2 = zeros([nx, ny]), zeros([nx, ny]), zeros([nx, ny])
    ydiff = zeros([nx, ny])
    theta = 1.0

    ierrs, ierrs_disp, derrs, bgmag = [], [], [], []
    snaps = {}
    want = set(int(i) for i in snapshot_iters)
    t0 = time.time()

    for itr in range(1, itermax + 1):
        old = (f1*1., f2*1., [y*1. for y in yd], ybg*1., ygx1*1., ygy1*1.,
               yl1*1., ygx2*1., ygy2*1., yl2*1., ydiff*1.)

        # ---- primal ----
        g1 = A1T(yd)*1.0
        g1 += prob.mdivx(ygx1)*nxg
        g1 += prob.mdivy(ygy1)*nyg
        g1 += l1f*yl1
        if bg:
            g1 += ydiff
        g1 *= mask
        f1old = f1*1.
        f1 = f1 - tau*g1
        f1[f1 < 0.0] = 0.
        f1 *= mask
        f1bar = f1 + theta*(f1 - f1old)

        if bg:
            g2 = A2T(ybg)
            g2 += prob.mdivx(ygx2)*nxg
            g2 += prob.mdivy(ygy2)*nyg
            g2 += l1f*yl2
            g2 -= ydiff
            g2 *= mask
            f2old = f2*1.
            f2 = f2 - tau*g2
            if bg_nonneg:
                f2[f2 < 0.0] = 0.
            f2 *= mask
            f2bar = f2 + theta*(f2 - f2old)

        # ---- dual: f1 data constraint(s) ----
        Ax = A1(f1bar)
        if two:
            res = (Ax[0] - d_hi, Ax[1] - d_lo)
            derr = sqrt(((res[0]/nu)**2).sum() + ((res[1]/nu)**2).sum())
            yd[0] = _ball(yd[0] + sig_hi*res[0], sig_hi*nu*eps_sc["hi"])
            yd[1] = _ball(yd[1] + sig_lo*res[1], sig_lo*nu*eps_sc["lo"])
        else:
            res = (Ax[0] - d_s,)
            derr = sqrt(((res[0]/nu)**2).sum())
            yd[0] = _ball(yd[0] + sig*res[0], sig*nu*eps_sc["single"])
        derrs.append(derr/sqrt(nrays))

        # ---- dual: f2 data constraint ----
        if bg:
            rbg = A2(f2bar) - d_bg
            ybg = _ball(ybg + sig*rbg, sig*nu*eps_sc["bg"])

        # ---- dual: regularizers ----
        ygx1 = _linf(ygx1 + sig*prob.gradx(f1bar)*nxg, wx)
        ygy1 = _linf(ygy1 + sig*prob.grady(f1bar)*nyg, wy)
        yl1 = _linf(yl1 + sig*l1f*f1bar, wl1)
        if bg:
            ygx2 = _linf(ygx2 + sig*prob.gradx(f2bar)*nxg, wx)
            ygy2 = _linf(ygy2 + sig*prob.grady(f2bar)*nyg, wy)
            yl2 = _linf(yl2 + sig*l1f*f2bar, wl1)
            ydiff = _linf(ydiff + sig*(f1bar - f2bar), alpha3)

        # ---- He-Yuan relaxation ----
        (f1o, f2o, ydo, ybgo, gx1o, gy1o, l1o, gx2o, gy2o, l2o, dfo) = old
        for j in range(len(yd)):
            yd[j] = ydo[j] - RHO*(ydo[j] - yd[j])
        ygx1 = gx1o - RHO*(gx1o - ygx1)
        ygy1 = gy1o - RHO*(gy1o - ygy1)
        yl1 = l1o - RHO*(l1o - yl1)
        f1 = f1o - RHO*(f1o - f1)
        if bg:
            ybg = ybgo - RHO*(ybgo - ybg)
            ygx2 = gx2o - RHO*(gx2o - ygx2)
            ygy2 = gy2o - RHO*(gy2o - ygy2)
            yl2 = l2o - RHO*(l2o - yl2)
            ydiff = dfo - RHO*(dfo - ydiff)
            f2 = f2o - RHO*(f2o - f2)

        # ---- scoring ----
        ierrs.append(float(sqrt(((f1bar - prob.phantom)**2).sum()/(nx*ny))))
        if bg:
            disp = _display(f1bar, f2bar, mask)
            ierrs_disp.append(
                float(sqrt(((disp - prob.phantom)**2).sum()/(nx*ny))))
            bgmag.append(float(sqrt((f2bar**2).sum()/(nx*ny))))
        if itr in want:
            snaps[itr] = f1bar*1.

        if verbose and (itr in (1, 10, 50, 100) or itr % 250 == 0):
            tail = f" disp={ierrs_disp[-1]:.4f}" if bg else ""
            print(f"    it {itr:5d}  rmse={ierrs[-1]:.4f}{tail}  "
                  f"derr={derrs[-1]:.3e}")

    out = {"ierrs": np.array(ierrs), "derrs": np.array(derrs),
           "image": f1bar, "lambda_max": lam,
           "lam_spread": lam_spread,
           "elapsed_s": time.time() - t0}
    if bg:
        out["ierrs_disp"] = np.array(ierrs_disp)
        out["bg_rms"] = np.array(bgmag)
        out["background"] = f2bar
        out["display"] = _display(f1bar, f2bar, mask)
    for k, v in snaps.items():
        out.setdefault("snaps", {})[k] = v
    return out


def _display(f1, f2, mask, thresh_frac=0.1):
    """Sidky Eq. (4) without the depth blur, which has no 2D counterpart.

    The background is subtracted and its mean level inside the support added
    back, so the image keeps an absolute level instead of being flattened to
    zero. The support threshold is relative to the background's own maximum
    rather than the paper's absolute 0.1 /cm, because our phantoms are in
    arbitrary attenuation units and an absolute cut would mean something
    different on each one.
    """
    sup = (f2 > thresh_frac*max(f2.max(), 1e-12)).astype(float)*mask
    b = (f2*sup).sum()/max(sup.sum(), 1.0)
    return ((f1 - f2) + b*sup)*mask


def _ball(y, radius):
    """Prox of the conjugate of an l2-ball indicator."""
    m = sqrt((y**2).sum())
    if m - radius > 0:
        return y*((m - radius)/m)
    return y*0.0


def _linf(p, w):
    """Prox of the conjugate of w*||.||_1: project onto the l-inf ball."""
    return w*p/maximum(np.abs(p), w)


def _power_stacked(prob, A1, A1T, A2, A2T, two, bg, sig_rel,
                   wx, wy, wl1, alpha3, l1f, npower=NPOWER):
    """lambda_max of the stacked operator, with the per-block sigma weights.

    This is the quantity the manuscript's stability section certifies. Running
    it over the stacked (f1, f2) operator is what keeps the gesso arms from
    being handed a step size their larger operator does not support.
    """
    rng = np.random.default_rng(prob._power_seed)
    mask = prob.mask
    x1 = rng.standard_normal((prob.nx, prob.ny))*mask
    x2 = rng.standard_normal((prob.nx, prob.ny))*mask if bg else None
    mag1 = mag2 = 1.0
    for _ in range(npower):
        ys = A1(x1)
        sd = [sqrt(sig_rel["hi"]), sqrt(sig_rel["lo"])] if two else [1.0]
        ys = [y*s for y, s in zip(ys, sd)]
        gx1, gy1 = prob.gradx(x1)*prob.nuxgrad, prob.grady(x1)*prob.nuygrad
        li1 = l1f*x1
        parts = [*ys, gx1, gy1, li1]
        if bg:
            parts += [A2(x2), prob.gradx(x2)*prob.nuxgrad,
                      prob.grady(x2)*prob.nuygrad, l1f*x2, x1 - x2]
        mag1 = sqrt(sum((p**2).sum() for p in parts))
        if mag1 > 0:
            parts = [p/mag1 for p in parts]
        k = len(ys)
        ys, rest = parts[:k], parts[k:]
        ys = [y*s for y, s in zip(ys, sd)]
        n1 = A1T(ys) + prob.mdivx(rest[0])*prob.nuxgrad \
            + prob.mdivy(rest[1])*prob.nuygrad + l1f*rest[2]
        if bg:
            bgp = rest[3:]
            n1 = n1 + bgp[4]
            n2 = (A2T(bgp[0]) + prob.mdivx(bgp[1])*prob.nuxgrad
                  + prob.mdivy(bgp[2])*prob.nuygrad + l1f*bgp[3] - bgp[4])
            n2 *= mask
        n1 *= mask
        mag2 = sqrt((n1**2).sum() + ((n2**2).sum() if bg else 0.0))
        if mag2 > 0:
            x1 = n1/mag2
            if bg:
                x2 = n2/mag2
    lam = (mag1 + mag2)*0.5
    return lam, abs(mag1 - mag2)/max(lam, 1e-12)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def _phantom(name):
    """The manuscript's phantom array for `name`.

    Routed through the same registry the published runners use so that a gesso
    run and a paper run see byte-identical input. PHANTOMS["breast"] is a None
    sentinel there -- the solver builds that one internally -- so it comes from
    noise2d's mirror of that construction instead.
    """
    from convergence2d import PHANTOMS
    import noise2d
    if name == "breast":
        return noise2d._breast_phantom()
    ph = PHANTOMS[name]()
    if ph is None:
        raise ValueError(f"no array available for phantom {name!r}")
    return ph


def run_matrix(name, itermax=500, noise_seed=1000, mismatch=0.0, alpha3=1.0,
               bg_blur=1.0, arms=ARMS, force=False, mfact=2):
    """Run the chosen arms on one realization, through the runstore."""
    store = RunStore("2d_gesso")
    out = {}
    for arm in arms:
        cfg = {"phantom": name, "arm": arm, "itermax": itermax,
               "noise_seed": noise_seed, "mismatch": mismatch,
               "alpha3": alpha3, "bg_blur": bg_blur, "mfact": mfact,
               "eps_mode": "noise_per_band", "sigma_lo_scale": SIGMA_LO_SCALE}

        def compute(c, _arm=arm):
            ph = _phantom(c["phantom"])
            prob = Problem(ph, noise_seed=c["noise_seed"],
                           mismatch=c["mismatch"], bg_blur=c["bg_blur"])
            r = solve(prob, _arm, itermax=c["itermax"], alpha3=c["alpha3"])
            r["eps"] = prob.eps
            r["eps_bg"] = prob.eps_bg
            r["zero_frac"] = prob.zero_frac
            r.pop("snaps", None)
            return r

        out[arm] = store.load_or_run(cfg, compute, force=force,
                                     label=f"{name}_{arm}_s{noise_seed}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phantom", default="breast")
    p.add_argument("--itermax", type=int, default=500)
    p.add_argument("--seed", type=int, default=1000, help="noise realization")
    p.add_argument("--mismatch", type=float, default=0.0,
                   help="amplitude of the smooth gain error; 0 disables it")
    p.add_argument("--alpha3", type=float, default=10.0,
                   help="weight on ||f1 - f2||_1; swept on the breast phantom "
                        "under a 0.5 gain error, best-iterate RMSE went "
                        "0.2934 / 0.2676 / 0.2421 / 0.2400 for single_gesso "
                        "and 0.1813 / 0.1770 / 0.1754 / 0.1816 for two_gesso "
                        "at alpha3 = 0.1 / 1 / 10 / 100, so 10 is the "
                        "two-channel optimum and the single-channel arm is "
                        "still improving at 100")
    p.add_argument("--bg-blur", type=float, default=1.0,
                   help="background in-plane blur, in field-of-view units")
    p.add_argument("--arms", default="all")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    arms = ARMS if a.arms == "all" else tuple(a.arms.split(","))
    res = run_matrix(a.phantom, itermax=a.itermax, noise_seed=a.seed,
                     mismatch=a.mismatch, alpha3=a.alpha3,
                     bg_blur=a.bg_blur, arms=arms, force=a.force)

    print(f"\n{'arm':<14}{'RMSE@end':>11}{'best':>10}{'@it':>7}"
          f"{'display':>11}{'lam':>9}")
    for arm in arms:
        r = res[arm]
        ie = r["ierrs"]
        disp = (f"{r['ierrs_disp'][-1]:.4f}" if "ierrs_disp" in r else "--")
        print(f"{arm:<14}{ie[-1]:>11.4f}{ie.min():>10.4f}"
              f"{int(ie.argmin())+1:>7}{disp:>11}"
              f"{float(r['lambda_max']):>9.4f}")

    base = res.get("single_plain")
    if base is not None:
        print()
        for arm in arms:
            if arm == "single_plain":
                continue
            g = 100.0*(base["ierrs"][-1] - res[arm]["ierrs"][-1]) \
                / base["ierrs"][-1]
            print(f"  {arm:<14} {g:+6.1f}% vs single_plain at "
                  f"iteration {a.itermax}")

    TAB.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
