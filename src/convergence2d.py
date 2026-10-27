"""2D convergence experiments for the certified-acceleration story.

For each phantom (breast = original abstract phantom, then Shepp-Logan) runs
the real DTV/PDHG solver (compare_methods_multiresolution) single-channel vs
two-channel at sigma_lo_scale in {1, 4, 8}:

    s=1  plain two-channel (no LF emphasis)      ~ ties single-channel
    s=4  certified         (tau*lambda_max(M) ~ 1.0001, same step budget)
    s=8  uncertified       (tau*lambda_max(M) ~ 1.18)

Outputs (experiments/figs, experiments/tables):
    {phantom}_convergence.png   image RMSE vs iteration, all curves
    {phantom}_iter_ladder.png   single vs two(s=4) reconstructions at iters
    {phantom}_error_ladder.png  |recon - truth| for the same
    {phantom}_rmse.md           RMSE + % reduction table

Geometry/filters are identical across phantoms (256, 25 views / 50 deg,
cutoff_hi=4, cutoff_lo=8), so the operator-level stability numbers from
stability.py apply to both.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT                                     # noqa: E402
os.chdir(ROOT)
import reconstruction as cm  # noqa: E402
from store import RunStore                                  # noqa: E402

from paths import FIG, TAB, ensure                         # noqa: E402
ensure()

MFACT = 2
SCALES = [1.0, 4.0, 8.0]
HERO_S = 4.0                       # the certified setting shown in the ladders
SNAP_ITERS = [10, 50, 100, 200, 500]
REPORT_ITERS = [10, 20, 50, 100, 200, 300, 500]
ITERMAX = 500
VMAX = 1.6
SCALE_COLOR = {1.0: "#7f7f7f", 4.0: "#1f77b4", 8.0: "#d62728"}


def shepp_logan_2d(n):
    table = [
        (1.0, .69, .92, 0, 0, 0), (-0.8, .6624, .8740, 0, -.0184, 0),
        (-0.2, .11, .31, .22, 0, -18), (-0.2, .16, .41, -.22, 0, 18),
        (0.1, .21, .25, 0, .35, 0), (0.1, .046, .046, 0, .1, 0),
        (0.1, .046, .046, 0, -.1, 0), (0.1, .046, .023, -.08, -.605, 0),
        (0.1, .023, .023, 0, -.606, 0), (0.1, .023, .046, .06, -.605, 0),
    ]
    img = np.zeros((n, n))
    coord = np.linspace(-1.0, 1.0, n)
    Xc, Yc = np.meshgrid(coord, coord, indexing="ij")
    for (A, a, b, x0, y0, phi) in table:
        ph = np.deg2rad(phi)
        xt = (Xc - x0) * np.cos(ph) + (Yc - y0) * np.sin(ph)
        yt = -(Xc - x0) * np.sin(ph) + (Yc - y0) * np.cos(ph)
        img[(xt / a) ** 2 + (yt / b) ** 2 <= 1.0] += A
    sl = np.maximum(img, 0.0)
    return sl * (VMAX / max(sl.max(), 1e-12))


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def configure_solver():
    cm.cutoffparm = 4.0
    cm.cutoffparm_lo = 8.0
    cm.eps = 0.001
    cm.eps_hi = cm.eps
    cm.eps_lo = 1.25 * cm.eps
    cm.RESOLUTION_PARAMS[256]["itermax"] = ITERMAX


def _compute(cfg):
    """One (phantom, s) pair, as the store calls it.

    Flattened to plain arrays because the store round-trips through npz: the
    snapshots go in alongside the traces so the ladder figures redraw without
    a solve, which is the whole point -- these runs are noiseless and
    deterministic, so recomputing them yields the same numbers at the cost of
    several minutes.
    """
    configure_solver()
    cm.sigma_lo_scale = cfg["s"]
    r = cm.run_reconstruction_for_mfact(
        MFACT, snapshot_iters=SNAP_ITERS,
        phantom_override=_PHANTOM_CACHE[cfg["phantom"]])
    out = {"ierrs_single": np.asarray(r["ierrs_single"], dtype=np.float64),
           "ierrs_two": np.asarray(r["ierrs_two"], dtype=np.float64),
           # the ladder figures draw the phantom beside the iterates
           "phimage": np.asarray(r["phimage"], dtype=np.float32)}
    for arm in ("single", "two"):
        for i, img in sorted(r[f"snapshots_{arm}"].items()):
            out[f"snap_{arm}_{i}"] = np.asarray(img, dtype=np.float32)
    return out


# run_phantom hands the built phantom to the store's compute callback without
# putting a 256x256 array in the config, which is hashed and written to the
# manifest as JSON.
_PHANTOM_CACHE = {}


def run_phantom(name, phantom_override, force=False):
    """Traces and snapshots for every s, from the store where possible."""
    _PHANTOM_CACHE[name] = phantom_override
    store = RunStore("2d_convergence")
    res = {}
    for s in SCALES:
        cfg = dict(phantom=name, s=s, itermax=ITERMAX, mfact=MFACT,
                   snap_iters=list(SNAP_ITERS), noisy=False, noise_seed=None,
                   cutoff_hi=4.0, cutoff_lo=8.0, eps=0.001, eps_lo_mult=1.25,
                   solver="reconstruction.py")
        d = store.load_or_run(cfg, _compute, force=force,
                              label=f"{name}_s{s:g}")
        res[s] = {
            "ierrs_single": np.asarray(d["ierrs_single"]),
            "ierrs_two": np.asarray(d["ierrs_two"]),
            "phimage": np.asarray(d["phimage"]),
            "snapshots_single": {i: np.asarray(d[f"snap_single_{i}"])
                                 for i in SNAP_ITERS
                                 if f"snap_single_{i}" in d},
            "snapshots_two": {i: np.asarray(d[f"snap_two_{i}"])
                              for i in SNAP_ITERS if f"snap_two_{i}" in d},
        }
    return res


def fig_convergence(name, res):
    single = np.asarray(res[SCALES[0]]["ierrs_single"])
    it = np.arange(1, len(single) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(it, single, "k-", lw=2.0, label="single-channel")
    for s in SCALES:
        tag = {1.0: "two-channel  s=1", 4.0: "two-channel  s=4 (certified)",
               8.0: "two-channel  s=8 (uncertified)"}[s]
        ax.semilogy(it, np.asarray(res[s]["ierrs_two"]), "-",
                    color=SCALE_COLOR[s], lw=1.8, label=tag)
    ax.set_xlabel("iteration")
    ax.set_ylabel("image RMSE")
    ax.set_xlim(0, len(single))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = FIG / f"{name}_convergence.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def fig_iter_ladder(name, res):
    phi = res[HERO_S]["phimage"]
    snaps_s = res[HERO_S]["snapshots_single"]
    snaps_t = res[HERO_S]["snapshots_two"]
    n = len(SNAP_ITERS)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 4.4))
    for ax in axes.flat:
        _strip(ax)
    for r in (0, 1):
        axes[r, 0].imshow(phi.T, cmap="gray", vmin=0, vmax=VMAX, origin="lower")
    axes[0, 0].set_title("ground truth", fontsize=10)
    axes[0, 0].set_ylabel("single", fontsize=11)
    axes[1, 0].set_ylabel(f"two-channel s={HERO_S:.0f}", fontsize=11)
    for i, itr in enumerate(SNAP_ITERS):
        c = i + 1
        s_im, t_im = snaps_s[itr], snaps_t[itr]
        axes[0, c].imshow(s_im.T, cmap="gray", vmin=0, vmax=VMAX, origin="lower")
        axes[0, c].set_title(f"iter {itr}\nRMSE {rmse(s_im, phi):.3f}", fontsize=10)
        axes[1, c].imshow(t_im.T, cmap="gray", vmin=0, vmax=VMAX, origin="lower")
        axes[1, c].set_xlabel(f"RMSE {rmse(t_im, phi):.3f}", fontsize=10)
    fig.tight_layout()
    p = FIG / f"{name}_iter_ladder.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def fig_error_ladder(name, res, err_range=0.5):
    phi = res[HERO_S]["phimage"]
    snaps_s = res[HERO_S]["snapshots_single"]
    snaps_t = res[HERO_S]["snapshots_two"]
    n = len(SNAP_ITERS)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4))
    for ax in axes.flat:
        _strip(ax)
    axes[0, 0].set_ylabel("|single - truth|", fontsize=11)
    axes[1, 0].set_ylabel(f"|two s={HERO_S:.0f} - truth|", fontsize=11)
    for i, itr in enumerate(SNAP_ITERS):
        axes[0, i].imshow(np.abs(snaps_s[itr] - phi).T, cmap="gray",
                          vmin=0, vmax=err_range, origin="lower")
        axes[0, i].set_title(f"iter {itr}", fontsize=10)
        axes[1, i].imshow(np.abs(snaps_t[itr] - phi).T, cmap="gray",
                          vmin=0, vmax=err_range, origin="lower")
    fig.tight_layout()
    p = FIG / f"{name}_error_ladder.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def write_table(name, res):
    single = res[SCALES[0]]["ierrs_single"]
    lines = [f"# {name}: image RMSE vs iteration  (256, 25 views / 50 deg)\n"]
    head = "| iter | single | " + " | ".join(
        f"s={s:.0f}" + {1.0: "", 4.0: " (cert.)", 8.0: " (uncert.)"}[s]
        for s in SCALES) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (2 + len(SCALES)))
    for itr in REPORT_ITERS:
        cells = [f"{itr}", f"{single[itr-1]:.5f}"]
        for s in SCALES:
            b = res[s]["ierrs_two"][itr - 1]
            red = (single[itr - 1] - b) / single[itr - 1] * 100.0
            cells.append(f"{b:.5f} ({red:+.1f}%)")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n%-reduction is vs single-channel at the same iteration "
                 "(+ = lower RMSE).")
    p = TAB / f"{name}_rmse.md"
    p.write_text("\n".join(lines) + "\n")
    print(f"  saved {p}")


def _ct2_phantom(name):
    from phantoms2d import load_2d_phantom
    return load_2d_phantom(name)


PHANTOMS = {
    "breast": lambda: None,                                  # cm default phantom
    "shepp_logan": lambda: shepp_logan_2d(int(512 / MFACT)),
    "head": lambda: _ct2_phantom("head"),
    "jaw": lambda: _ct2_phantom("jaw"),
    "defrise": lambda: _ct2_phantom("defrise"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", choices=list(PHANTOMS) + ["all"], default="all")
    args = ap.parse_args()
    names = list(PHANTOMS) if args.phantom == "all" else [args.phantom]
    for name in names:
        print(f"\n########## {name} ##########")
        res = run_phantom(name, PHANTOMS[name]())
        fig_convergence(name, res)
        fig_iter_ladder(name, res)
        fig_error_ladder(name, res)
        write_table(name, res)


if __name__ == "__main__":
    main()
