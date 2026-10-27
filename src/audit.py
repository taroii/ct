"""How many noise realizations does each manuscript result actually rest on?

The rule is that no reported number may rest on fewer than MIN_REPS
independent Poisson realizations. That is easy to state and easy to lose
track of, because the realization count lives in a run store, the number that
depends on it lives in a .tex fragment, and nothing connects them. This walks
the stores, reports n per configuration, and exits non-zero if anything a
table is built from falls short -- so the check can be run before a submission
instead of remembered.

Only the budgets the manuscript actually cites are enforced. The stores
accumulate every budget ever run -- a 600-iteration 3D pilot does not stop
existing when the reported run moves to 2000 -- and failing on those would make
the check cry wolf, which is how a check stops being read. Uncited
configurations are listed, so they stay visible, but they do not fail the run.

Deterministic results are listed separately rather than silently passed: the
stability diagnostic and the noiseless RMSE table have no noise to average
over, so a realization count does not apply to them and claiming one would be
worse than reporting none.

Results computed without a run store cannot be audited from here at all, so
they are named explicitly rather than omitted: a passing audit that quietly
skipped a figure would be worse than one that says what it did not check.
Check those by hand, in the header of the table they write.

    python src/audit.py
    python src/audit.py --itermax-2d 2000 --itermax-3d 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from paths import RESULTS, TAB                             # noqa: E402

# store -> the manuscript objects built from it
BACKS = {
    "2d_noise_stats": "tab:noise (2D rows), tab:oracle (2D rows), "
                      "fig:noisegrid, fig:mechanism(b)",
    "3d_noise_stats": "tab:noise (3D row), tab:oracle (3D row), fig:dbt",
    "2d_tolerance_sensitivity": "fig:mechanism(a)",
    "2d_band_resolved": "fig:bandresolved",
    "2d_convergence": "tab:rmse (noiseless; no realization applies)",
}
DETERMINISTIC = [
    ("tab:stability, fig:bandsplit", "operator eigenvalues; no noise model"),
    ("tab:rmse", "noiseless data; no realization to average over"),
]
# Stochastic, but recomputed rather than stored, so this script cannot count
# their realizations. Empty is the goal: every stochastic result should be
# store-backed, both so a rerun is free and so it can be counted here.
UNSTORED = []


def counts(store, deterministic, variants):
    """n per (phantom, eps_mode, itermax, extra), from the manifests."""
    d = RESULTS / store
    out = defaultdict(set)
    if not d.exists():
        return out
    for jp in d.glob("*.json"):
        # macOS AppleDouble sidecars are not manifests.
        if jp.name.startswith("._"):
            continue
        try:
            c = json.loads(jp.read_text())["config"]
        except Exception:
            continue
        # A run with no noise seed has no realization to count: the noiseless
        # arm of the band-resolved figure is deterministic, and reporting it
        # as "n=1" would read as a result resting on one realization rather
        # than one that has none to rest on.
        if c.get("noise_seed") is None:
            deterministic[store].add(c.get("phantom"))
            continue
        key = (c.get("phantom"), c.get("eps_mode"), c.get("itermax"),
               c.get("eps_lo_mult"))
        # Count DISTINCT noise seeds, not manifests: two runs of the same
        # realization are one realization, and counting files would turn a
        # duplicated run into a larger n.
        out[key].add(c.get("noise_seed"))
        # Also record the full config behind each key. A key is only a label;
        # two genuinely different configurations can share one -- the breast
        # phantom carries flat-tolerance runs from before eps_cal_draws
        # existed alongside runs from after -- and a seed count alone reports
        # that as one healthy sample rather than two half-samples.
        variants[(store, key)].add(tuple(sorted((k, repr(v))
                                                for k, v in c.items()
                                                if k != "noise_seed")))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-reps", type=int, default=20)
    ap.add_argument("--itermax-2d", type=int, default=500,
                    help="the 2D budget the manuscript reports; other budgets "
                         "in the store are listed but not enforced")
    ap.add_argument("--itermax-3d", type=int, default=600,
                    help="the 3D budget the manuscript reports")
    args = ap.parse_args()

    # Which budgets each store is read at. A configuration at any other budget
    # is a pilot or a superseded run, not a manuscript result.
    cited = {"2d_noise_stats": {args.itermax_2d},
             "3d_noise_stats": {args.itermax_3d},
             "2d_tolerance_sensitivity": {args.itermax_2d, 500},
             "2d_band_resolved": {args.itermax_2d, 500}}

    bad = []
    skipped = 0
    deterministic = defaultdict(set)
    variants = defaultdict(set)
    mixed = []
    for store, backs in BACKS.items():
        print(f"\n=== {store}\n    backs: {backs}")
        c = counts(store, deterministic, variants)
        if not c:
            print("    (no runs)")
            continue
        for key in sorted(c, key=str):
            ph, mode, it, mult = key
            n = len(c[key])
            tag = "" if mult is None else f" lo_mult={mult:g}"
            used = it in cited.get(store, set())
            if not used:
                skipped += 1
                flag = "   (not cited; not enforced)"
            elif n < args.min_reps:
                flag = "   <-- BELOW MINIMUM"
            else:
                flag = ""
            print(f"    phantom={ph} eps_mode={mode} itermax={it}{tag}"
                  f"  n={n}{flag}")
            if len(variants.get((store, key), ())) > 1:
                v = len(variants[(store, key)])
                print(f"           ^ note: {v} config variants share this key")
                if used:
                    mixed.append((store, key, v))
            if used and n < args.min_reps:
                bad.append((store, key, n))
        for ph in sorted(deterministic.get(store, ())):
            print(f"    phantom={ph} noiseless  (deterministic; no "
                  f"realization applies)")

    print("\n=== deterministic, no realization count applies")
    for what, why in DETERMINISTIC:
        print(f"    {what}: {why}")

    if UNSTORED:
        print("\n=== stochastic but not store-backed, NOT CHECKED HERE")
        for what, why in UNSTORED:
            print(f"    {what}: {why}")

    print()
    if skipped:
        print(f"({skipped} configuration(s) at budgets the manuscript does "
              f"not cite were listed but not enforced.)")
    if mixed:
        # A warning, not a failure. Variants usually mean a field was added to
        # the manifest between runs rather than that the computation changed:
        # the breast flat-tolerance runs differ only in recording
        # eps_cal_draws/eps_cal_k, which a single-tolerance run never reads,
        # and the two generations agree to 4e-16 over the whole trace. Only a
        # human can tell those apart, so this says where to look instead of
        # failing. To settle one, load both generations' traces for a shared
        # seed and difference them; if they agree the variants are cosmetic.
        print(f"NOTE: {len(mixed)} key(s) hold more than one config variant. "
              f"Check whether the difference is computational before trusting "
              f"the pooled n:")
        for store, key, v in mixed:
            print(f"  {store} {key} -- {v} variants")
    if bad:
        print(f"FAIL: {len(bad)} configuration(s) below {args.min_reps} "
              f"realizations.")
        for store, key, n in bad:
            print(f"  {store} {key} n={n}")
    if bad:
        return 1
    print(f"OK: every cited configuration has at least {args.min_reps} "
          f"realizations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
