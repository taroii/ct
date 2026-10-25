"""Emit the Experiments section's LaTeX tables straight from the runstore.

Every number in the manuscript's tables is produced here, from the same files
the figures are drawn from, so a table and a figure cannot disagree and a
rerun cannot leave a stale digit behind in the .tex. The fragments are written
to tables/tex/ and also printed, so they can be pasted into a self-contained
Overleaf document; the generator stays in the repo as the provenance.

Sources, one per table and no mixing:
  tab:stability  results/stability_curve/margin_vs_s.json  (+ the band
                 separation block of tables/stability.md, which is the only
                 place the two channel eigenvalues are computed)
  tab:rmse       tables/<phantom>_rmse.md, written by run_2d_convergence.py,
                 which caches nothing -- that markdown is its single source
  tab:noise      results/2d_noise_stats/ and results/3d_noise_stats/, through
                 the same paired_stats the per-phantom tables use

    python paper/experiments/make_tex_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pairedstats import paired_stats                       # noqa: E402

TAB = HERE / "tables"
OUT = TAB / "tex"
STORE2D = HERE / "results" / "2d_noise_stats"
STORE3D = HERE / "results" / "3d_noise_stats"

EPS_MODE = "noise_per_band"
HERO_S = 4.0
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


def _val(cell):
    return float(cell.split()[0])


# ---------------------------------------------------------------- stability
def tab_stability():
    data = json.loads((HERE / "results" / "stability_curve" /
                       "margin_vs_s.json").read_text())
    s = np.array(data["s"]); m = np.array(data["margin"])
    lo = np.array(data["lo_share"])
    show = [1.0, 2.0, 4.0, 5.5, 6.0, 8.0]
    idx = [int(np.argmin(np.abs(s - v))) for v in show]

    band = {}
    for cells in _md_rows(TAB / "stability.md"):
        if len(cells) == 2:
            band[cells[0]] = cells[1]
    lam_hi = band.get("lambda_max(X^T F_hi^2 X)", "?")
    lam_lo = band.get("lambda_max(X^T F_lo^2 X)", "?")
    ov = next((v for k, v in band.items() if "overlap" in k), "?")

    cross = s[np.argmax(lo > 0.5)] if (lo > 0.5).any() else float("nan")
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Operator-level stability diagnostic: geometry and filters "
         r"only, with no phantom and no reconstruction. \textbf{Left:} leading "
         r"eigenvalues of the two data-channel normal operators and the "
         r"overlap of their dominant eigenvectors. \textbf{Right:} the sharp "
         r"margin $\tau\lambda_{\max}(M)=\lambda_{\max}(\widetilde M(s))/"
         r"\lambda_{\max}(\widetilde M(1))$, normalised so the single-channel "
         r"step-size boundary is exactly $1$, and the low channel's share of "
         r"the limiting eigenmode. Values are power-iteration estimates and "
         r"are reported as computed; a margin within the iteration's own "
         r"accuracy of $1$ sits \emph{at} the single-channel boundary rather "
         rf"than strictly below it. The low band first owns the limiting "
         rf"eigenmode at $s={cross:g}$.}}",
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
def tab_rmse(it=500):
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Noiseless data: image RMSE at iteration ${it}$ for the "
         r"single-channel baseline, and the relative RMSE reduction of the "
         r"two-channel method at low-frequency step scales $s\in\{1,4,8\}$ "
         r"(positive means lower RMSE than single-channel). $s=4$ holds "
         r"$\tau\lambda_{\max}(M)$ at the single-channel boundary; $s=8$ "
         r"exceeds it.}",
         r"\label{tab:rmse}", r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{RMSE reduction (\%)}\\",
         r"\cmidrule(l){3-5}",
         r"Phantom & Single RMSE & $s{=}1$ & $s{=}4$ & $s{=}8$\\", r"\midrule"]
    for p in ORDER:
        rows = {int(r[0]): r for r in _md_rows(TAB / f"{p}_rmse.md")}
        r = rows[it]
        L.append(f"{LABEL[p]:<13}& ${_val(r[1]):.3f}$ & "
                 f"${_pct(r[2]):+.1f}$ & ${_pct(r[3]):+.1f}$ & "
                 f"${_pct(r[4]):+.1f}$" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


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


def tab_noise(early=100, late=500):
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Poisson noise, paired over independent realizations: "
         rf"both arms reconstruct the same realization, and the solver seed is "
         rf"fixed so the spread is measurement noise and not step-size jitter. "
         rf"Image RMSE at iterations ${early}$ and ${late}$, with the "
         rf"reduction relative to single-channel at the same iteration, the "
         rf"$95\%$ confidence interval of the mean paired difference at "
         rf"iteration ${late}$. WINLINE The 3D row is the DBT geometry of "
         rf"Section~\ref{{sec:experiments}}; GPU time caps $n$ there.}}",
         r"\label{tab:noise}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{Single RMSE} & "
         r"\multicolumn{2}{c}{Reduction (\%)} & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         rf"Phantom & $n$ & @${early}$ & @${late}$ & @${early}$ & @${late}$ & "
         rf"$95\%$ CI @${late}$ \\", r"\midrule"]
    wins = []
    for p in ORDER:
        runs = _load(STORE2D, phantom=p, eps_mode=EPS_MODE, s=HERO_S)
        if not runs:
            print(f"  (no runs for {p})"); continue
        a, b = paired_stats(runs, early), paired_stats(runs, late)
        wins += [a["win"], b["win"]]
        L.append(f"{LABEL[p]:<13}& ${b['n']}$ & ${a['single']:.3f}$ & "
                 f"${b['single']:.3f}$ & ${a['rel']:+.1f}$ & "
                 f"${b['rel']:+.1f}$ & $[{b['lo']:+.3f},{b['hi']:+.3f}]$" + r"\\")
    runs3 = _load(STORE3D, eps_mode=EPS_MODE, s=HERO_S)
    if runs3:
        a, b = paired_stats(runs3, early), paired_stats(runs3, late)
        wins += [a["win"], b["win"]]
        L += [r"\midrule",
              f"3D DBT & ${b['n']}$ & ${a['single']:.3f}$ & "
              f"${b['single']:.3f}$ & ${a['rel']:+.1f}$ & ${b['rel']:+.1f}$ & "
              f"$[{b['lo']:+.3f},{b['hi']:+.3f}]$" + r"\\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(L)
    # Only claim a clean sweep if every row actually swept.
    if wins and min(wins) == 1.0:
        note = (rf"The two-channel arm had the lower RMSE in every one of "
                rf"these paired realizations, at both iteration counts.")
    else:
        note = (rf"Win rates over both iteration counts range from "
                rf"{100 * min(wins):.0f}\% to {100 * max(wins):.0f}\%.")
    print(f"  win rates over both iteration counts: "
          f"{', '.join(f'{100 * w:.0f}%' for w in wins)}")
    return tex.replace("WINLINE", note)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("stability", tab_stability), ("rmse", tab_rmse),
                     ("noise", tab_noise)):
        tex = fn()
        (OUT / f"tab_{name}.tex").write_text(tex + "\n")
        print(f"\n%% ---- tables/tex/tab_{name}.tex ----\n{tex}")


if __name__ == "__main__":
    main()
