"""Emit the experiment tables as standalone LaTeX fragments from the runstore.

Every number is produced here, from the same files the figures are drawn
from, so a table and a figure cannot disagree. Each fragment is written to
out/tables/tex/ and also printed; nothing is inserted into any document, and
where the fragments end up is not this repo's concern.

Sources, one per table and no mixing:
  tab:stability  results/stability_curve/margin_vs_s.json  (+ the band
                 separation block of tables/stability.md, which is the only
                 place the two channel eigenvalues are computed)
  tab:rmse       tables/<phantom>_rmse.md, written by convergence2d.py,
                 which caches nothing -- that markdown is its single source
  tab:noise      results/2d_noise_stats/ and results/3d_noise_stats/, through
                 the same paired_stats the per-phantom tables use

    python src/tables.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import paired_stats, oracle_stats         # noqa: E402

from paths import TAB, TEX as OUT, RESULTS                 # noqa: E402

STORE2D = RESULTS / "2d_noise_stats"
STORE3D = RESULTS / "3d_noise_stats"

EPS_MODE = "noise_per_band"
HERO_S = 4.0
# The control: the same two-channel split with no amplification. It isolates
# the step-up from the split, because s=1 and s=4 solve the identical problem
# and differ only in the trajectory.
CTRL_S = 1.0
# Tables report RMSE x 10^2 so the entries are one- or two-digit numbers
# instead of a column of leading zeros. The caption states the factor.
RSCALE = 100.0
STORE3D_CLEAN = RESULTS / "3d_convergence"
# Pin the iteration budget as well as the tolerance mode. Runs at a different
# itermax are a different experiment, and _load matches on config alone, so
# without this a longer rerun would be pooled into the same n and silently
# widen (or narrow) every confidence interval in the table. --itermax-2d and
# --itermax-3d move these without editing the file, so switching the
# manuscript to a longer budget is one flag rather than one more constant to
# forget.
ITERMAX_2D = 500
ITERMAX_3D = 600
# No manuscript number may rest on fewer realizations than this.
MIN_REPS = 20
ORDER = ["breast", "head", "defrise", "jaw", "shepp_logan"]
LABEL = {"breast": "Breast", "head": "Head", "defrise": "Defrise",
         "jaw": "Jaw", "shepp_logan": "Shepp--Logan"}


def _md_rows(path):
    """Rows of the first pipe table in a generated markdown file."""
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln.startswith("|") or set(ln) <= set("|-: "):
            continue
        # Split on unescaped pipes only: these tables carry cells like
        # `\|<v_hi,v_lo>\|`, and a naive split tears them in half.
        cells = [c.strip().replace("\\|", "|")
                 for c in re.split(r"(?<!\\)\|", ln.strip("|"))]
        rows.append(cells)
    return rows[1:] if rows else rows          # drop the header row


def _pct(cell):
    """'0.07678 (+22.0%)' -> 22.0"""
    m = re.search(r"\(([-+][\d.]+)%\)", cell)
    return float(m.group(1)) if m else float("nan")


def _rm(x):
    """An RMSE as it is printed in the tables: scaled, fixed width."""
    return "--" if x is None or not np.isfinite(x) else f"{RSCALE * x:.2f}"


def _val(cell):
    return float(cell.split()[0])


# ---------------------------------------------------------------- stability
def tab_stability():
    data = json.loads((RESULTS / "stability_curve" /
                       "margin_vs_s.json").read_text())
    s = np.array(data["s"]); m = np.array(data["margin"])
    lo = np.array(data["lo_share"])
    show = [1.0, 2.0, 4.0, 5.5, 6.0, 8.0]
    idx = [int(np.argmin(np.abs(s - v))) for v in show]

    q = TAB / "stability.json"
    if not q.exists():
        raise SystemExit(f"{q} is missing -- run src/stability.py first")
    band = json.loads(q.read_text())
    lam_hi = f"{band['lam_hi']:.4f}"
    lam_lo = f"{band['lam_lo']:.4f}"
    ov = f"{band['overlap']:.4f}"

    cross = s[np.argmax(lo > 0.5)] if (lo > 0.5).any() else float("nan")
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Operator-level stability diagnostic, from geometry and "
         r"filters only. Left: leading eigenvalues of the two data-channel "
         r"normal operators and the overlap of their dominant eigenvectors. "
         r"Right: the margin $\tau\lambda_{\max}(M)=\lambda_{\max}"
         r"(\widetilde M(s))/\lambda_{\max}(\widetilde M(1))$, normalized "
         r"so the single-channel step-size boundary is $1$, and the low "
         r"channel's share of the limiting eigenmode. Values are "
         r"power-iteration estimates reported as computed; a margin within "
         r"the iteration's accuracy of $1$ lies at the single-channel "
         r"boundary.}",
         r"\label{tab:stability}",
         r"\begin{tabular}{lc}", r"\toprule",
         rf"$\lambda_{{\max}}(X^\top F_{{\mathrm{{hi}}}}^2 X)$ & ${lam_hi}$\\",
         rf"$\lambda_{{\max}}(X^\top F_{{\mathrm{{lo}}}}^2 X)$ & ${lam_lo}$\\",
         rf"$|\langle v_{{\mathrm{{hi}}}},v_{{\mathrm{{lo}}}}\rangle|$ & ${ov}$\\",
         r"\bottomrule", r"\end{tabular}", r"\qquad",
         r"\begin{tabular}{l" + "c" * len(idx) + "}", r"\toprule",
         "$s$ & " + " & ".join(f"${s[i]:g}$" for i in idx) + r"\\", r"\midrule",
         r"$\tau\lambda_{\max}(M)$ & "
         + " & ".join(f"${m[i]:.4f}$" for i in idx) + r"\\",
         r"low-band share & "
         + " & ".join(f"${lo[i]:.3f}$" for i in idx) + r"\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ------------------------------------------------------------- noiseless RMSE
def _clean3d(it, s):
    """Image RMSE of both arms at iteration `it` from the noiseless 3D run."""
    if not STORE3D_CLEAN.exists():
        return None, None
    for jp in sorted(STORE3D_CLEAN.glob("*.json")):
        if jp.name.startswith("._"):
            continue
        try:
            c = json.loads(jp.read_text())["config"]
        except Exception:
            continue
        if c.get("s") != s or c.get("phantom") != "victre":
            continue
        with np.load(jp.with_suffix(".npz")) as d:
            sg = np.asarray(d["ierrs_single"], float)
            tw = np.asarray(d["ierrs_two"], float)
        if len(sg) < it:
            print(f"  (noiseless 3D s={s:g} ran only {len(sg)} iterations, "
                  f"need {it})")
            return None, None
        return float(sg[it - 1]), float(tw[it - 1])
    return None, None


def tab_rmse(it=500):
    """Noiseless RMSE for the baseline and the two certified step scales.

    s=8 is not reported. It sits above the step-size boundary
    (tau*lambda_max(M)=1.18 in Table~\ref{tab:stability}), so it is outside
    the regime the paper certifies; quoting its RMSE invites the reader to
    use a setting the stability analysis does not cover.
    """
    miss = []
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Noiseless data: image RMSE ($\times10^{{-2}}$) at "
         rf"iteration ${it}$ for the single-channel baseline and for the "
         rf"two-channel method at low-frequency step scales $s=1$ and $s=4$. "
         rf"$s=1$ is the same split with no amplification, so the difference "
         rf"between the $s=1$ and $s=4$ columns is the effect of the "
         rf"step-up alone, and the difference between the single-channel and "
         rf"$s=1$ columns is the effect of the split alone. Both scales hold "
         rf"$\tau\lambda_{{\max}}(M)$ at the single-channel boundary "
         rf"(Table~\ref{{tab:stability}}). CLEAN3D}}",
         r"\label{tab:rmse}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Phantom & Single & $s{=}1$ & $s{=}4$\\", r"\midrule"]
    for ph in ORDER:
        rows = {int(r[0]): r for r in _md_rows(TAB / f"{ph}_rmse.md")}
        r = rows[it]
        L.append(f"{LABEL[ph]:<13}& ${_rm(_val(r[1]))}$ & "
                 f"${_rm(_val(r[2]))}$ & ${_rm(_val(r[3]))}$" + r"\\")
    sg1, tw1 = _clean3d(it, CTRL_S)
    sg4, tw4 = _clean3d(it, HERO_S)
    single3 = sg4 if sg4 is not None else sg1
    if single3 is None:
        miss.append("the 3D row")
        print("  (tab_rmse: no noiseless 3D run -- "
              "python src/noise3d.py --noiseless --s 1 / --s 4)")
    else:
        L += [r"\midrule",
              f"3D DBT       & ${_rm(single3)}$ & ${_rm(tw1)}$ & "
              f"${_rm(tw4)}$" + r"\\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(L)
    # Built as one string rather than squeezed afterwards: a blanket
    # whitespace collapse over the whole table also eats the column padding.
    note = (r"The 3D row is the DBT geometry of Section~\ref{sec:experiments}."
            if single3 is not None else
            r"\textbf{Provisional:} the noiseless 3D run has not been "
            r"completed, so that row is absent.")
    return tex.replace("CLEAN3D", note)


# ------------------------------------------------------------- noise summary
def _load(store, **match):
    runs = []
    for jp in sorted(store.glob("*.json")):
        if jp.name.startswith("._"):
            continue
        try:
            c = json.loads(jp.read_text())["config"]
        except Exception:
            continue
        if any(c.get(k) != v for k, v in match.items()):
            continue
        with np.load(jp.with_suffix(".npz")) as d:
            runs.append(dict(ierrs_single=np.asarray(d["ierrs_single"], float),
                             ierrs_two=np.asarray(d["ierrs_two"], float)))
    return runs


def _load_seeded(store, **match):
    """Same as _load, but keyed by noise realization.

    The s=1 control and the s=4 arm are separate runs, so a column-by-column
    mean over whatever each one happens to have would compare them on
    different noise draws and quietly stop being paired. Keying on the seed
    lets the caller intersect first and compare like with like.
    """
    out = {}
    for jp in sorted(store.glob("*.json")):
        if jp.name.startswith("._"):
            continue
        try:
            c = json.loads(jp.read_text())["config"]
        except Exception:
            continue
        if any(c.get(k) != v for k, v in match.items()):
            continue
        with np.load(jp.with_suffix(".npz")) as d:
            out[c.get("noise_seed")] = dict(
                ierrs_single=np.asarray(d["ierrs_single"], float),
                ierrs_two=np.asarray(d["ierrs_two"], float))
    return out


def _arm_cells(store, itermax, iters, **match):
    """Mean RMSE per arm at each checkpoint, paired across the two s values.

    Returns (cells, n, stats4) where cells maps iteration -> (single, s1, s4).
    """
    r4 = _load_seeded(store, s=HERO_S, itermax=itermax, **match)
    r1 = _load_seeded(store, s=CTRL_S, itermax=itermax, **match)
    if not r4:
        return None, 0, None
    seeds = sorted(set(r4) & set(r1)) if r1 else sorted(r4)
    runs4 = [r4[k] for k in sorted(r4)]
    cells, stats4 = {}, {}
    for it in iters:
        a = paired_stats(runs4, it)
        stats4[it] = a
        s1 = None
        if r1 and seeds:
            # Paired on the shared draws only, and the baseline is taken from
            # the same runs, so every number in the row sees one noise set.
            sg = np.array([r4[k]["ierrs_single"][it - 1] for k in seeds])
            t1 = np.array([r1[k]["ierrs_two"][it - 1] for k in seeds])
            chk = np.array([r1[k]["ierrs_single"][it - 1] for k in seeds])
            if np.max(np.abs(sg - chk)) > 1e-9:
                print(f"  WARNING: the single-channel trace differs between "
                      f"the s=1 and s=4 runs at iteration {it} "
                      f"(max |diff| {np.max(np.abs(sg - chk)):.3g}); the two "
                      f"sets are not the same realizations")
            s1 = float(t1.mean())
        cells[it] = (a["single"], s1, a["two"])
    return cells, len(runs4), stats4


def tab_noise(early=500, late=None):
    """Paired RMSE at an early checkpoint and at the published budget.

    `late` defaults to the budget rather than a literal: the two were set
    independently before, so raising the budget read the longer runs but went
    on reporting iteration 500 of them -- the right data at the wrong column.
    """
    late = ITERMAX_2D if late is None else late
    iters = (early, late)
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Poisson noise: image RMSE ($\times10^{{-2}}$) at "
         rf"iterations ${early}$ and ${late}$, averaged over independent "
         rf"realizations. The comparison is paired: every arm reconstructs "
         rf"the same realizations, and the solver seed is fixed so the "
         rf"spread is measurement noise and not step-size jitter. $s=1$ is "
         rf"the two-channel split with no amplification. WINLINE D3LINE"
         + (rf"All runs use a ${ITERMAX_2D}$-iteration budget.}}"
            if ITERMAX_2D == ITERMAX_3D else
            rf"2D runs use a ${ITERMAX_2D}$-iteration budget and the 3D run "
            rf"a ${ITERMAX_3D}$-iteration one.}}"),
         r"\label{tab:noise}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         rf" & \multicolumn{{3}}{{c}}{{Iteration ${early}$}} & "
         rf"\multicolumn{{3}}{{c}}{{Iteration ${late}$}}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Phantom & Single & $s{=}1$ & $s{=}4$ & Single & $s{=}1$ & "
         r"$s{=}4$\\", r"\midrule"]
    wins, thin, noctrl, ns = [], [], [], []
    for ph in ORDER:
        cells, nrep, st = _arm_cells(STORE2D, ITERMAX_2D, iters,
                                     phantom=ph, eps_mode=EPS_MODE)
        if cells is None:
            print(f"  (no runs for {ph})"); continue
        ns.append(nrep)
        wins += [st[i]["win"] for i in iters]
        if nrep < MIN_REPS:
            thin.append(rf"{LABEL[ph]} ($n={nrep}$)")
        if cells[late][1] is None:
            noctrl.append(LABEL[ph])
        L.append(f"{LABEL[ph]:<13}& " + " & ".join(
            f"${_rm(v)}$" for i in iters for v in cells[i]) + r"\\")
        print(f"  {LABEL[ph]:<13} n={nrep}  " + "  ".join(
            f"@{i}: {st[i]['rel']:+.1f}%" for i in iters))
    c3, n3, st3 = _arm_cells(STORE3D, ITERMAX_3D, iters,
                             eps_mode=EPS_MODE)
    if c3 is None:
        norow3d = True
        print("  (tab_noise: no 3D runs at all -- row omitted; "
              "python src/noise3d.py --down 4 ...)")
    else:
        norow3d = False
        ns.append(n3)
        wins += [st3[i]["win"] for i in iters]
        if n3 < MIN_REPS:
            thin.append(rf"3D DBT ($n={n3}$)")
        if c3[late][1] is None:
            noctrl.append("3D DBT")
        L += [r"\midrule", "3D DBT       & " + " & ".join(
            f"${_rm(v)}$" for i in iters for v in c3[i]) + r"\\"]
        print(f"  {'3D DBT':<13} n={n3}  " + "  ".join(
            f"@{i}: {st3[i]['rel']:+.1f}%" for i in iters))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    flags = []
    if thin:
        verb = "rests" if len(thin) == 1 else "rest"
        flags.append(", ".join(thin)
                     + rf" {verb} on fewer than ${MIN_REPS}$ realizations.")
    if norow3d:
        flags.append("the 3D run has not been completed, so that row is "
                     "absent.")
    if noctrl:
        flags.append("the $s=1$ control has not been run for "
                     + ", ".join(noctrl) + ".")
        print(f"  (tab_noise: no s=1 control for {', '.join(noctrl)} -- "
              f"python src/noise2d.py --s 1 ... / noise3d.py --s 1 ...)")
    if flags:
        L[2] = L[2].rstrip("}") + r" \textbf{Provisional:} " \
            + " ".join(flags) + "}"
    tex = "\n".join(L)
    if wins and min(wins) == 1.0:
        note = ("The two-channel arm at $s=4$ had the lower RMSE in every one "
                "of these paired realizations, at both iteration counts.")
    else:
        note = (rf"Win rates for $s=4$ over both iteration counts range from "
                rf"{100 * min(wins):.0f}\% to {100 * max(wins):.0f}\%.")
    if ns:
        print(f"  realizations per row: {ns}")
    d3 = "" if norow3d else (r"The 3D row is the DBT geometry of "
                             r"Section~\ref{sec:experiments}. ")
    return tex.replace("WINLINE", note).replace("D3LINE", d3)


# -------------------------------------------------------------- oracle table
def tab_oracle():
    """Both arms scored at their own best iterate, over the same realizations.

    This is the table that answers the stopping-rule objection: a shared
    iteration count flatters whichever arm has not yet degraded, so the
    conservative comparison gives both arms a best-iterate rule that neither
    has in practice. A censored minimum -- one sitting on the final iterate --
    means the budget ended the descent, so the reduction is a lower bound, and
    the caption says so rather than implying an optimum that was never seen.
    """
    rows = []
    for p in ORDER:
        runs = _load(STORE2D, phantom=p, eps_mode=EPS_MODE, s=HERO_S,
                     itermax=ITERMAX_2D)
        if not runs:
            print(f"  (oracle: no runs for {p})")
            continue
        rows.append((LABEL[p], oracle_stats(runs)))
    runs3 = _load(STORE3D, eps_mode=EPS_MODE, s=HERO_S, itermax=ITERMAX_3D)
    if runs3:
        rows.append(("3D DBT", oracle_stats(runs3)))
    if not rows:
        return ""

    censored = [lab for lab, o in rows if o["censored_two"]]
    thin = [rf"{lab} ($n={o['n']}$)" for lab, o in rows if o["n"] < MIN_REPS]
    note = ""
    if censored:
        where = ("every row" if len(censored) == len(rows)
                 else ", ".join(censored))
        note = (r" The two-channel minimum sits on the final iterate for "
                + where + r", so there the budget rather than the method "
                r"ended the descent and the reduction is a lower bound.")
    if thin:
        verb = "rests" if len(thin) == 1 else "rest"
        note += (r" \textbf{Provisional:} " + ", ".join(thin)
                 + rf" {verb} on fewer than ${MIN_REPS}$ realizations.")

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Oracle stopping: each arm scored at its own best iterate, "
         r"over the same paired realizations as Table~\ref{tab:noise}. Both "
         r"arms semiconverge, so a shared iteration count is not neutral; "
         r"this gives both a best-iterate rule that neither supplies in "
         r"practice, which is the most conservative comparison available."
         + note + r"}",
         r"\label{tab:oracle}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Phantom & $n$ & Single best & Two best & Reduction (\%) & "
         r"Best iterate (single / two)\\", r"\midrule"]
    for lab, o in rows:
        ks = (f"{o['ks_lo']}" if o["ks_lo"] == o["ks_hi"]
              else rf"{o['ks_lo']}\text{{--}}{o['ks_hi']}")
        kt = (f"{o['kt_lo']}" if o["kt_lo"] == o["kt_hi"]
              else rf"{o['kt_lo']}\text{{--}}{o['kt_hi']}")
        if lab == "3D DBT":
            L.append(r"\midrule")
        L.append(f"{lab:<13}& ${o['n']}$ & ${o['single']:.3f}$ & "
                 f"${o['two']:.3f}$ & ${o['rel']:+.1f}$ & "
                 f"${ks}$ / ${kt}$" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    for lab, o in rows:
        print(f"  oracle {lab:<13} n={o['n']:<3} {o['rel']:+.1f}%  "
              f"single@{o['ks_lo']}-{o['ks_hi']} two@{o['kt_lo']}-{o['kt_hi']}"
              + ("  [two censored by the budget]" if o["censored_two"] else ""))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--itermax-2d", type=int, default=ITERMAX_2D)
    ap.add_argument("--itermax-3d", type=int, default=ITERMAX_3D)
    args = ap.parse_args()
    globals()["ITERMAX_2D"] = args.itermax_2d
    globals()["ITERMAX_3D"] = args.itermax_3d
    print(f"budgets: 2D {ITERMAX_2D}, 3D {ITERMAX_3D}; "
          f"minimum realizations {MIN_REPS}")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("stability", tab_stability), ("rmse", tab_rmse),
                     ("noise", tab_noise), ("oracle", tab_oracle)):
        tex = fn()
        if not tex:
            print(f"  (tab_{name}: nothing to emit -- skipped)")
            continue
        (OUT / f"tab_{name}.tex").write_text(tex + "\n")
        print(f"\n%% ---- tables/tex/tab_{name}.tex ----\n{tex}")


if __name__ == "__main__":
    main()
