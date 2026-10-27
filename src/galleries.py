"""Candidate image-panel figures: what the two methods actually reconstruct.

The RMSE tables say the two-channel arm is better; these show it. Both figures
are image panels only -- no curves -- in the style of the left half of the 3D
DBT figure.

  gallery   one row per phantom: truth, both reconstructions, and the two
            error images on a shared per-row scale. Under Poisson noise at the
            iteration count the noise table reports.
  ladder    one phantom across iterations, single on top and two below, which
            is where the qualitative difference lives: the single-channel
            iterate degrades after its turnaround while the two-channel one
            keeps improving.

Config is taken from run_2d_noise_stats, so these are the same runs as
Table~\\ref{tab:noise} rather than lookalikes; as a guard the RMSE traces are
compared against the stored run for the same realization and the mismatch is
printed. Slices are cached to .npz so the figures can be restyled without
recomputing anything.

    python src/galleries.py                  # both
    python src/galleries.py --from-npz       # restyle only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
from paths import ROOT                                     # noqa: E402
# reconstruction.py lives in paper/, and the solver resolves its phantom data
# relative to the repo root, exactly as run_2d_noise_stats sets things up.
import os                                                       # noqa: E402
os.chdir(ROOT)
sys.path.insert(0, str(HERE))

from paths import FIG, RESULTS                             # noqa: E402

OUT = RESULTS / "galleries"
STORE2D = RESULTS / "2d_noise_stats"

ORDER = ["breast", "head", "defrise", "jaw", "shepp_logan"]
LABEL = {"breast": "Breast", "head": "Head", "defrise": "Defrise",
         "jaw": "Jaw", "shepp_logan": "Shepp-Logan"}
LADDER_ITERS = [10, 50, 100, 200, 500]
NOISE_SEED = 1000
EPS_MODE = "noise_per_band"


def stored_trace(phantom, noise_seed, itermax, eps_mode, s):
    """The run_2d_noise_stats traces for this realization, for the guard."""
    for jp in sorted(STORE2D.glob("*.json")):
        if jp.name.startswith("._"):
            continue
        try:
            c = json.loads(jp.read_text())["config"]
        except Exception:
            continue
        # itermax is deliberately NOT matched: PDHG's trajectory does not
        # depend on where it is stopped, so a 2000-iteration run must
        # reproduce the stored 500-iteration one over the first 500. Comparing
        # the common prefix therefore still validates a longer run.
        if (c.get("phantom") == phantom and c.get("noise_seed") == noise_seed
                and c.get("eps_mode") == eps_mode and c.get("s") == s):
            with np.load(jp.with_suffix(".npz")) as d:
                return (np.asarray(d["ierrs_single"], float),
                        np.asarray(d["ierrs_two"], float))
    return None


def run_case(name, itermax, noise_seed, eps_mode, snapshot_iters,
             noiseless=False):
    """One reconstruction of both arms, with the published configuration.

    `noiseless` reproduces run_2d_convergence instead: no Poisson sampling and
    the flat eps_lo = 1.25 eps, since per-band calibration is defined by the
    noise and there is none to calibrate against.
    """
    import reconstruction as cm                                  # noqa: E402
    from noise2d import (PHANTOMS, MFACT, auto_i0,    # noqa: E402
                                    HERO_S, SOLVER_SEED,
                                    EPS_CAL_DRAWS, EPS_CAL_K)
    cm.cutoffparm, cm.cutoffparm_lo = 4.0, 8.0
    cm.eps = cm.eps_hi = 0.001
    cm.eps_lo = 1.25 * 0.001          # overwritten by the per-band calibration
    cm.sigma_lo_scale = HERO_S
    phantom = PHANTOMS[name]()
    if noiseless:
        i0 = None
        r = cm.run_reconstruction_for_mfact(
            MFACT, phantom_override=phantom, seed=SOLVER_SEED,
            itermax_override=itermax, snapshot_iters=snapshot_iters)
    else:
        i0 = auto_i0(phantom)
        r = cm.run_reconstruction_for_mfact(
            MFACT, phantom_override=phantom, seed=SOLVER_SEED,
            itermax_override=itermax, i0=i0, noise_seed=noise_seed,
            eps_mode=eps_mode, eps_factor=1.1, eps_cal_draws=EPS_CAL_DRAWS,
            eps_cal_k=EPS_CAL_K, snapshot_iters=snapshot_iters)

    si = np.asarray(r["ierrs_single"], float)
    tw = np.asarray(r["ierrs_two"], float)
    if noiseless:
        print(f"  {name}: noiseless, no stored run to guard against")
        return r, si, tw, i0
    st = stored_trace(name, noise_seed, itermax, eps_mode, HERO_S)
    if st is None:
        print(f"  GUARD {name}: no stored run to compare against")
    else:
        m = min(len(si), len(st[0]))
        d1 = float(np.abs(si[:m] - st[0][:m]).max())
        d2 = float(np.abs(tw[:m] - st[1][:m]).max())
        ok = "OK" if max(d1, d2) <= 1e-6 else "*** MISMATCH ***"
        print(f"  GUARD {name}: first {m} iterations vs stored run -- "
              f"max|diff| single {d1:.1e} two {d2:.1e}  {ok}")
    return r, si, tw, i0


def collect(itermax, noise_seed, eps_mode, names, want_ladder,
            noiseless=False):
    OUT.mkdir(parents=True, exist_ok=True)
    snaps = LADDER_ITERS if want_ladder else None
    for name in names:
        p = OUT / f"{name}_seed{noise_seed}_{eps_mode}_it{itermax}.npz"
        if p.exists():
            print(f"  {name}: cached"); continue
        print(f"  {name}: running both arms, {itermax} iterations ...")
        r, si, tw, i0 = run_case(name, itermax, noise_seed, eps_mode, snaps,
                                 noiseless=noiseless)
        d = dict(gt=np.asarray(r["phimage"], float),
                 single=np.asarray(r["xbarim_single"], float),
                 two=np.asarray(r["xbarim_two"], float),
                 ierrs_single=si, ierrs_two=tw,
                 i0=(np.nan if i0 is None else i0), itermax=itermax)
        for it in (snaps or []):
            if it in r["snapshots_single"]:
                d[f"snap_single_{it}"] = np.asarray(r["snapshots_single"][it], float)
                d[f"snap_two_{it}"] = np.asarray(r["snapshots_two"][it], float)
        np.savez_compressed(p, **d)
        print(f"    saved {p}")


def _load(name, noise_seed, eps_mode, itermax):
    p = OUT / f"{name}_seed{noise_seed}_{eps_mode}_it{itermax}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def oracle_iters(name, noise_seed, eps_mode, itermax):
    """Each arm's own best iterate, read from a cached run's RMSE trace.

    Semiconvergence makes a single shared iteration count a choice that can
    flatter either arm: late favours the two-channel arm because the
    single-channel one has degraded, early favours the single-channel one. The
    oracle comparison gives BOTH arms a best-iterate stopping rule that neither
    has in practice, which is the most conservative framing available.
    """
    d = _load(name, noise_seed, eps_mode, itermax)
    if d is None:
        return None
    return (int(np.argmin(d["ierrs_single"])) + 1,
            int(np.argmin(d["ierrs_two"])) + 1)


def fig_oracle(names, noise_seed, eps_mode, itermax):
    """Gallery with each arm shown at its own best iterate."""
    rows = []
    for name in names:
        d = _load(name, noise_seed, eps_mode, itermax)
        it = oracle_iters(name, noise_seed, eps_mode, itermax)
        if d is None or it is None:
            continue
        ks, kt = it
        if f"snap_single_{ks}" not in d or f"snap_two_{kt}" not in d:
            print(f"  oracle: {name} missing snapshots at {ks}/{kt}"); continue
        rows.append((name, d, ks, kt))
    if not rows:
        print("  oracle: nothing to draw"); return
    fig, axes = plt.subplots(len(rows), 5, figsize=(11.0, 2.22 * len(rows)))
    axes = np.atleast_2d(axes)
    heads = ["Ground truth", "Single-channel", "Two-channel ($s=4$)",
             "Error, single-channel", "Error, two-channel"]
    for i, (name, d, ks, kt) in enumerate(rows):
        gt = d["gt"]
        rs, rt = d[f"snap_single_{ks}"], d[f"snap_two_{kt}"]
        rmse = lambda a: float(np.sqrt(((a - gt) ** 2).mean()))   # noqa: E731
        vmax = float(np.percentile(gt, 99.5))
        emax = float(np.percentile(np.abs(np.concatenate(
            [(rs - gt).ravel(), (rt - gt).ravel()])), 99.5))
        panels = [(gt, "gray", vmax, None, None),
                  (rs, "gray", vmax, rmse(rs), ks),
                  (rt, "gray", vmax, rmse(rt), kt),
                  (np.abs(rs - gt), "magma", emax, None, None),
                  (np.abs(rt - gt), "magma", emax, None, None)]
        for j, (img, cmap, vm, sc, k) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(heads[j], fontsize=9)
            if sc is not None:
                ax.text(0.03, 0.97, f"{sc:.4f}\niter {k}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7.5, color="w",
                        bbox=dict(boxstyle="square,pad=0.18", fc="k",
                                  ec="none", alpha=0.55))
        gain = 100 * (rmse(rs) - rmse(rt)) / rmse(rs)
        axes[i, 0].set_ylabel(f"{LABEL[name]}\n$+{gain:.1f}\\%$", fontsize=9)
    # Titles name the figure; the stopping rule and its justification go in
    # the caption, where they can be stated carefully.
    fig.suptitle("Reconstructions at each arm's best iterate", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = FIG / "gallery_oracle.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def fig_gallery(names, noise_seed, eps_mode, itermax):
    data = [(n, _load(n, noise_seed, eps_mode, itermax)) for n in names]
    data = [(n, d) for n, d in data if d is not None]
    if not data:
        print("  gallery: nothing cached"); return
    rows = len(data)
    fig, axes = plt.subplots(rows, 5, figsize=(11.0, 2.22 * rows))
    axes = np.atleast_2d(axes)
    heads = ["Ground truth", "Single-channel", "Two-channel ($s=4$)",
             "Error, single-channel", "Error, two-channel"]
    for i, (name, d) in enumerate(data):
        gt, rs, rt = d["gt"], d["single"], d["two"]
        rmse = lambda a: float(np.sqrt(((a - gt) ** 2).mean()))   # noqa: E731
        # One grey window and one error scale per row: phantoms are imaged at
        # their own dose, so absolute levels are not comparable across rows.
        vmax = float(np.percentile(gt, 99.5))
        emax = float(np.percentile(np.abs(np.concatenate(
            [(rs - gt).ravel(), (rt - gt).ravel()])), 99.5))
        panels = [(gt, "gray", vmax, None), (rs, "gray", vmax, rmse(rs)),
                  (rt, "gray", vmax, rmse(rt)),
                  (np.abs(rs - gt), "magma", emax, None),
                  (np.abs(rt - gt), "magma", emax, None)]
        for j, (img, cmap, vm, sc) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(heads[j], fontsize=9)
            if sc is not None:
                ax.text(0.03, 0.97, f"{sc:.4f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=8, color="w",
                        bbox=dict(boxstyle="square,pad=0.18", fc="k",
                                  ec="none", alpha=0.55))
        gain = 100 * (rmse(rs) - rmse(rt)) / rmse(rs)
        axes[i, 0].set_ylabel(f"{LABEL[name]}\n$+{gain:.1f}\\%$", fontsize=9)
    regime = "noiseless" if eps_mode == "noiseless" else "Poisson noise"
    fig.suptitle(f"Single- vs two-channel reconstructions, {regime}, "
                 f"iteration {itermax}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = FIG / "gallery.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def fig_ladder(name, noise_seed, eps_mode, itermax):
    d = _load(name, noise_seed, eps_mode, itermax)
    its = [i for i in LADDER_ITERS if f"snap_single_{i}" in (d or {})]
    if not its:
        print(f"  ladder: no snapshots cached for {name}"); return
    # The final iterate is stored as the run's result rather than as a
    # snapshot, so add it as the last rung when the budget runs past the
    # snapshot list. Without it a 2000-iteration ladder would stop at 500 and
    # disagree with the gallery drawn from the same cache.
    img = {(arm, i): d[f"snap_{arm}_{i}"]
           for arm in ("single", "two") for i in its}
    if itermax not in its:
        img[("single", itermax)] = d["single"]
        img[("two", itermax)] = d["two"]
        its = its + [itermax]
    gt = d["gt"]
    rmse = lambda a: float(np.sqrt(((a - gt) ** 2).mean()))       # noqa: E731
    vmax = float(np.percentile(gt, 99.5))
    fig, axes = plt.subplots(2, len(its) + 1, figsize=(2.0 * (len(its) + 1), 4.6))
    for r, arm, lab in ((0, "single", "Single-channel"),
                        (1, "two", "Two-channel ($s=4$)")):
        for c, it in enumerate(its):
            a = img[(arm, it)]
            ax = axes[r, c]
            ax.imshow(a, cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Iteration {it}", fontsize=9)
            # Inside the panel: text under row 0 is drawn behind row 1's axes.
            ax.text(0.03, 0.97, f"{rmse(a):.4f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color="w",
                    bbox=dict(boxstyle="square,pad=0.18", fc="k", ec="none",
                              alpha=0.55))
        axes[r, 0].set_ylabel(lab, fontsize=9)
        ax = axes[r, len(its)]
        ax.imshow(gt, cmap="gray", vmin=0.0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("Ground truth", fontsize=9)
    fig.suptitle(f"{LABEL[name]} reconstruction vs iteration, Poisson noise",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = FIG / f"ladder_{name}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--itermax", type=int, default=500)
    ap.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    ap.add_argument("--eps-mode", default=EPS_MODE)
    ap.add_argument("--phantoms", nargs="+", default=ORDER)
    ap.add_argument("--ladder", default="breast")
    ap.add_argument("--noiseless", action="store_true",
                    help="reproduce run_2d_convergence instead: no Poisson "
                         "sampling, flat eps_lo = 1.25 eps")
    ap.add_argument("--oracle", action="store_true",
                    help="re-run with snapshots at each arm's own best iterate")
    ap.add_argument("--from-npz", action="store_true",
                    help="skip the solver and restyle from the cache")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    tag = "noiseless" if args.noiseless else args.eps_mode
    if not args.from_npz:
        collect(args.itermax, args.noise_seed, tag, args.phantoms,
                want_ladder=True, noiseless=args.noiseless)
    if args.oracle:
        for name in args.phantoms:
            it = oracle_iters(name, args.noise_seed, tag, args.itermax)
            if it is None:
                print(f"  oracle: no cached run for {name}"); continue
            d = _load(name, args.noise_seed, tag, args.itermax)
            if all(f"snap_{a}_{k}" in d for a, k in (("single", it[0]),
                                                     ("two", it[1]))):
                print(f"  oracle {name}: snapshots already cached"); continue
            print(f"  oracle {name}: re-running for snapshots at "
                  f"single={it[0]}, two={it[1]} ...")
            r, si, tw, i0 = run_case(name, args.itermax, args.noise_seed, tag,
                                     sorted(set(it)), noiseless=args.noiseless)
            d = dict(d)
            for a, k in (("single", it[0]), ("two", it[1])):
                d[f"snap_{a}_{k}"] = np.asarray(r[f"snapshots_{a}"][k], float)
            np.savez_compressed(
                OUT / f"{name}_seed{args.noise_seed}_{tag}_it{args.itermax}.npz", **d)
        fig_oracle(args.phantoms, args.noise_seed, tag, args.itermax)
        return
    fig_gallery(args.phantoms, args.noise_seed, tag, args.itermax)
    fig_ladder(args.ladder, args.noise_seed, tag, args.itermax)


if __name__ == "__main__":
    main()
