"""How much of each phantom's data actually lives in the low channel.

The two-channel method accelerates the low-frequency channel. It can only help
to the extent that a phantom HAS low-frequency content for that channel to
carry, so the natural predictor of "will this method help here" is the share of
the phantom's projection energy that the low filter passes:

    frac_lo = <W_lo^2 g, g> / <(W_hi^2 + W_lo^2) g, g>,     g = X f

measured with the SAME ramp-filtered Hann^(1/2) weights and the same fan-beam
geometry the solver uses (imported from stability_diagnostic, not re-derived,
so the two cannot drift apart). This is a projection-domain quantity, not an
image-domain proxy: the channels split the data term, so that is where the
split has to be measured.

Companion to stability_metrics.py. Together they address two open questions:
which phantoms the method suits, and whether defrise's anomalous stability
behaviour tracks its band composition or its noise operating point.

Run under an environment with numba (the projector is jitted):

    python src/band_energy.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stability as sd            # noqa: E402  geometry + filters
# The authoritative phantom registry, so this cannot drift from what the
# convergence and noise runs actually reconstructed. "breast" is built inside
# the solver and registers as None, so it comes from the noise-stats mirror.
from convergence2d import PHANTOMS as _REG   # noqa: E402
from noise2d import _breast_phantom    # noqa: E402

from paths import TAB                                     # noqa: E402

PHANTOMS = list(_REG)


def build(name: str) -> np.ndarray:
    f = _REG[name]()
    return _breast_phantom() if f is None else f


def band_fractions(f: np.ndarray) -> dict:
    """Projection-domain energy split for one phantom."""
    g = np.zeros((sd.nviews, sd.nbins))
    sd.project(np.ascontiguousarray(f * sd.mask), g)

    def energy(W):
        return float((sd.fft_weight(g, W) * g).sum())

    e_hi, e_lo = energy(sd.W_hi2), energy(sd.W_lo2)
    e_single = energy(sd.W_single2)
    tot = e_hi + e_lo
    return {
        "e_hi": e_hi, "e_lo": e_lo, "e_single": e_single,
        "frac_lo": e_lo / tot if tot > 0 else np.nan,
        "lo_over_hi": e_lo / e_hi if e_hi > 0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantom", default="all")
    args = ap.parse_args()

    names = PHANTOMS if args.phantom == "all" else [args.phantom]
    TAB.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        f = build(name)
        b = band_fractions(f)
        rows.append((name, b))
        print(f"{name:12s} frac_lo={b['frac_lo']:.4f}  "
              f"lo/hi={b['lo_over_hi']:.4f}")

    L = [f"# Projection-domain band energy by phantom\n",
         f"Fraction of each phantom's projection energy passed by the low "
         f"channel, using the solver's own ramp-filtered Hann^(1/2) weights "
         f"(cutoff_hi={sd.cutoffparm}, cutoff_lo={sd.cutoffparm_lo}) and "
         f"geometry ({sd.nx}x{sd.ny}, {sd.nviews} views, {sd.nbins} bins).\n",
         "frac_lo = <W_lo^2 g, g> / <(W_hi^2 + W_lo^2) g, g>, g = X f.\n",
         "| phantom | frac_lo | lo/hi |", "|---|---|---|"]
    for name, b in rows:
        L.append(f"| {name} | {b['frac_lo']:.4f} | {b['lo_over_hi']:.4f} |")
    L += ["", "This is a property of the phantom and the geometry alone -- no "
              "reconstruction is involved, and it does not depend on the noise "
              "level, the tolerance, or the step sizes. With only five "
              "phantoms it can support a mechanism-consistent reading, not a "
              "fitted relationship."]
    p = TAB / "band_energy.md"
    p.write_text("\n".join(L) + "\n")
    print(f"saved {p}")

    # The same numbers as data. The .md above is for a human to read; figures
    # read this, so reformatting the report cannot move a plotted point.
    q = TAB / "band_energy.json"
    q.write_text(json.dumps({name: {k: float(v) for k, v in b.items()}
                             for name, b in rows}, indent=1) + "\n")
    print(f"saved {q}")


if __name__ == "__main__":
    main()
