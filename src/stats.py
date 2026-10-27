"""Paired single-vs-two statistics, shared by the 2D and 3D noise runs.

Lives here rather than in either runner so both use ONE definition of the
paired t-interval and the Wilcoxon guard. The 3D runner cannot simply import
the 2D one: run_2d_noise_stats does os.chdir(ROOT) at import time and pulls in
the 2D solver, so importing it from a GPU script would move the working
directory out from under the caller as a side effect.

Takes any list of run dicts carrying `ierrs_single` / `ierrs_two` traces, so it
is indifferent to which dimensionality produced them.
"""
import numpy as np


def _summarise(s, t):
    """The paired summary of two arrays of per-realization scores.

    Factored out so paired_stats and oracle_stats cannot drift apart: the
    interval, the Wilcoxon guard and the win rate are defined once.
    """
    d = s - t
    n = len(d)
    mean = d.mean()
    # t interval on the paired difference; ddof=1 since sigma is estimated.
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    # Separate try blocks on purpose. Sharing one meant a Wilcoxon failure
    # (it raises when all differences are zero, and warns for tiny n) silently
    # downgraded tcrit to the normal-approximation 1.96, quietly narrowing every
    # confidence interval in the table.
    tcrit = 1.96
    if n > 1:
        try:
            from scipy import stats
            tcrit = float(stats.t.ppf(0.975, n - 1))
        except Exception:
            pass                      # normal approximation; conservative enough
    p = np.nan
    if n >= 6:
        try:
            from scipy import stats
            p = float(stats.wilcoxon(s, t).pvalue)
        except Exception:
            pass
    lo, hi = mean - tcrit * se, mean + tcrit * se
    return dict(n=n, single=s.mean(), two=t.mean(), mean=mean, lo=lo, hi=hi,
                median=np.median(d), q1=np.percentile(d, 25),
                q3=np.percentile(d, 75), p=p, win=(d > 0).mean(),
                rel=100 * mean / s.mean() if s.mean() else np.nan)


def paired_stats(runs, it):
    """Paired difference at iteration `it`, positive = two-channel better."""
    s = np.array([r["ierrs_single"][it - 1] for r in runs])
    t = np.array([r["ierrs_two"][it - 1] for r in runs])
    return _summarise(s, t)


def oracle_stats(runs, itermax=None):
    """Paired difference with EACH arm scored at its own best iterate.

    A shared iteration count is not neutral when both arms semiconverge: a
    late one flatters the two-channel arm because the single-channel arm has
    degraded past its best, an early one flatters the single-channel arm. This
    gives both arms a best-iterate stopping rule that neither has in practice,
    which is the most conservative framing available and the one a referee
    cannot object to on stopping-rule grounds.

    `censored` reports whether any arm's minimum sits on the final iterate. A
    censored two-channel minimum means the budget, not the method, ended the
    descent, so the reduction is a LOWER bound on what the rule would give --
    reporting it without that caveat would understate nothing but would claim
    an optimum that was never observed.
    """
    k = itermax or min(len(r["ierrs_single"]) for r in runs)
    si = [np.asarray(r["ierrs_single"], float)[:k] for r in runs]
    tw = [np.asarray(r["ierrs_two"], float)[:k] for r in runs]
    s = np.array([a.min() for a in si])
    t = np.array([a.min() for a in tw])
    ks = np.array([int(a.argmin()) + 1 for a in si])
    kt = np.array([int(a.argmin()) + 1 for a in tw])
    out = _summarise(s, t)
    out.update(itermax=k, ks=ks, kt=kt,
               ks_lo=int(ks.min()), ks_hi=int(ks.max()),
               kt_lo=int(kt.min()), kt_hi=int(kt.max()),
               censored_single=bool((ks == k).any()),
               censored_two=bool((kt == k).any()))
    return out
