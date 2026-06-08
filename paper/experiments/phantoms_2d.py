"""2D slices of the ct-2 analytic 3D phantoms, framed onto the fan-beam
pipeline grid (256x256, 10 cm FOV) for the 2D convergence experiments.

Each 3D phantom (phantom3d.py) is embedded into a thin slab and one
representative slice is taken, then peak-scaled to ~1.6 so ||g|| and the
eps regime match the breast paper run (same trick as the Shepp-Logan slide):

    head     axial   (x,y) at z=0        skull ring + brain (strong LF) + dots
    jaw      axial   (x,y) at z=5.5       dental arch, teeth, gold crowns (HF/metal)
    defrise  coronal (x,z) at y=0         stacked disks -- directional LF stress

The slice fills a square window sized so the object sits inside the circular
FOV; the pipeline applies its own r=5 cm mask. Use --preview to dump intro
PNGs and sanity-check framing before running reconstructions.
"""
import sys
from pathlib import Path

import numpy as np

PHANTOMS3D = Path(__file__).resolve().parents[1] / "phantoms3d"
sys.path.insert(0, str(PHANTOMS3D))

from phantom3d import (image3D, phantom3D,            # noqa: E402
                       ellipsoid, ellipsoidal, gen_object, cylindrical, planar)

N = 256
PEAK = 1.6


def _defrise_lower_half():
    """Lower-half Defrise: a short cylinder with a stack of thin disks
    (reimplemented from emil_code/phantoms/half_defrise.py on phantom3d)."""
    bkgd, low, high = 1.0, -0.7, 0.7
    r, length = 0.85, 0.85
    ph = phantom3D()
    # background cylinder, z in [-length, 0]
    ph.add_component(gen_object(z0=-length / 2., att=bkgd, nsurf=3, surfaces=[
        cylindrical(ax=r, ay=r), planar(zc=length / 2., beta=np.pi),
        planar(zc=-length / 2.)]))
    ax_e = ay_e = 0.6
    az_e = 0.03
    # half ellipsoid at the midplane
    ph.add_component(gen_object(att=low, nsurf=2, surfaces=[
        ellipsoidal(ax=ax_e, ay=ay_e, az=az_e),
        planar(beta=np.pi)]))
    for i in range(3):
        ph.add_component(ellipsoid(z0=-0.2 - i * 0.2, att=low,
                                   ax=ax_e, ay=ay_e, az=az_e))
    for i in range(4):
        ph.add_component(ellipsoid(z0=-0.1 - i * 0.2, att=high,
                                   ax=ax_e, ay=ay_e, az=az_e))
    return ph


def _load_builder(name):
    if name == "head":
        from head_phantom_demo import build_head_phantom
        return build_head_phantom()
    if name == "jaw":
        from jaw_phantom_demo import build_jaw_phantom
        return build_jaw_phantom()
    if name == "breast3d":
        from breast_phantom_demo import build_breast_phantom
        return build_breast_phantom()
    if name == "defrise":
        return _defrise_lower_half()
    raise ValueError(name)


# (plane, center xyz, window cm, attenuation cap)
# cap bounds the dynamic range before peak-scaling: the jaw's att=19.3 gold
# crowns would otherwise drive soft tissue to near-black and dominate eps.
CONFIG = {
    "head":    ("axial",   (0.0, 0.0, 0.0), 28.0, None),
    "jaw":     ("axial",   (0.0, 4.0, 5.5), 25.0, 2.5),
    "defrise": ("coronal", (0.0, 0.0, -0.4), 2.0, None),
    "breast3d": ("axial",  (3.0, 0.0, 0.0), 22.0, None),
}


def load_2d_phantom(name):
    plane, (cx, cy, cz), W, cap = CONFIG[name]
    ph = _load_builder(name)
    thin = W * 0.012                      # slab thickness, a few voxels
    if plane == "axial":                  # slice z, keep (x,y)
        img = image3D(shape=(N, N, 3), xlen=W, ylen=W, zlen=thin,
                      x0=cx - W / 2, y0=cy - W / 2, z0=cz - thin / 2)
        ph.embed_in(img)
        sl = img.mat[:, :, 1]
    elif plane == "coronal":              # slice y, keep (x,z)
        img = image3D(shape=(N, 3, N), xlen=W, ylen=thin, zlen=W,
                      x0=cx - W / 2, y0=cy - thin / 2, z0=cz - W / 2)
        ph.embed_in(img)
        sl = img.mat[:, 1, :]
    else:
        raise ValueError(plane)
    sl = np.maximum(sl.astype(np.float64), 0.0)
    if cap is not None:
        sl = np.minimum(sl, cap)
    sl *= PEAK / max(sl.max(), 1e-12)
    return sl


def main():
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=["head", "jaw", "defrise"])
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / "figs"
    out.mkdir(parents=True, exist_ok=True)
    n = len(args.names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    # show the circular FOV the pipeline will mask to
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, name in zip(axes, args.names):
        sl = load_2d_phantom(name)
        print(f"{name:8s} shape={sl.shape} range=[{sl.min():.3f},{sl.max():.3f}] "
              f"frac>0={np.mean(sl > 0):.3f}")
        ax.imshow(sl.T, cmap="gray", vmin=0, vmax=PEAK, origin="lower")
        ax.plot((N/2 - 1) + (N/2) * np.cos(th), (N/2 - 1) + (N/2) * np.sin(th),
                "r-", lw=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = out / "phantoms_2d_preview.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
