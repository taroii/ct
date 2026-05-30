"""Quick visual check of the downloaded compressed+cropped VICTRE
phantom (data/compressed_victre/). Renders mid-axial / mid-coronal /
mid-sagittal slices using the same attenuation lookup and display
window as the analytic breast, so we can compare apples-to-apples.

The compressed phantom is shipped as `*.raw.gz` even though the .mhd
header refers to the un-gzipped filename, so we decompress on the fly.
"""
import argparse
import gzip
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# (folder, fallback DimSize if no .mhd, fallback spacing mm, default out name)
KNOWN_SOURCES = {
    "compressed_victre": (
        None, None,
        "compressed_victre_check.png"),
    # Lesion-inserted dense phantom: same dimensions as compressed+cropped
    # per the VICTRE readme (DimSize 810 1920 745 @ 0.05 mm).
    "compressed_legion_victre": (
        (810, 1920, 745), (0.05, 0.05, 0.05),
        "compressed_legion_victre_check.png"),
}

# VICTRE label -> (linear attenuation cm^-1 at 30 keV, friendly name).
# Values picked to be consistent with the analytic-breast scale (so the
# display window from the analytic phantom transfers).
VICTRE_MU = {
    0:   (0.000, "air"),
    1:   (0.275, "fat"),
    2:   (0.375, "skin"),
    29:  (0.368, "glandular"),
    33:  (0.368, "nipple"),
    40:  (0.368, "muscle"),
    50:  (0.000, "compression paddle"),
    88:  (0.368, "ligament"),
    95:  (0.368, "TDLU"),
    125: (0.368, "duct"),
    150: (0.368, "artery"),
    200: (0.450, "cancerous mass"),
    225: (0.368, "vein"),
    250: (4.310, "calcification"),
}


def parse_mhd(path):
    text = path.read_text()
    def field(name):
        m = re.search(rf"^{name}\s*=\s*(.+)$", text, re.MULTILINE)
        return m.group(1).strip() if m else None
    dim = [int(x) for x in field("DimSize").split()]
    spacing = [float(x) for x in field("ElementSpacing").split()]
    raw_name = field("ElementDataFile")
    return dim, spacing, raw_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="compressed_victre",
                    choices=list(KNOWN_SOURCES),
                    help="folder under data/ to inspect")
    args = ap.parse_args()

    data_dir = ROOT / "data" / args.source
    fallback_dim, fallback_spacing, out_name = KNOWN_SOURCES[args.source]
    out_png = ROOT / "presentation" / "figs" / out_name

    mhd_paths = list(data_dir.glob("*.mhd"))
    if mhd_paths:
        mhd_path = mhd_paths[0]
        print(f"Reading mhd {mhd_path}")
        (nx, ny, nz), (dx, dy, dz), raw_name = parse_mhd(mhd_path)
    else:
        assert fallback_dim is not None, (
            f"No .mhd in {data_dir} and no fallback DimSize")
        nx, ny, nz = fallback_dim
        dx, dy, dz = fallback_spacing
        # pick the first .raw.gz/.raw in the folder (skip mcgpu images).
        candidates = sorted(p for p in data_dir.iterdir()
                            if p.suffix in (".gz",) and "mcgpu" not in p.name)
        assert candidates, f"No phantom .raw.gz in {data_dir}"
        raw_name = candidates[0].name
        print(f"No .mhd; using readme fallback dimensions for {args.source}")
        print(f"  raw file    : {raw_name}")

    print(f"  DimSize     : {nx} x {ny} x {nz}  (X Y Z)")
    print(f"  Spacing     : {dx} x {dy} x {dz} mm")
    print(f"  Physical    : {nx*dx/10:.2f} x {ny*dy/10:.2f} x {nz*dz/10:.2f} cm")
    print(f"  Data        : {raw_name}")

    # The raw file may be gzipped; the mhd points at the un-gz name.
    raw_path = data_dir / raw_name
    if not raw_path.exists():
        raw_path = data_dir / (raw_name + ".gz")
    print(f"  Reading raw : {raw_path}")
    if raw_path.suffix == ".gz":
        with gzip.open(raw_path, "rb") as f:
            buf = f.read()
    else:
        buf = raw_path.read_bytes()
    expected = nx * ny * nz
    print(f"  Bytes       : {len(buf):,}  (expected {expected:,})")
    assert len(buf) == expected, "phantom size mismatch"

    vol_lbl = np.frombuffer(buf, np.uint8).reshape(nz, ny, nx)  # (z, y, x)
    print(f"  Shape       : {vol_lbl.shape}")

    # Label statistics.
    labels, counts = np.unique(vol_lbl, return_counts=True)
    total = vol_lbl.size
    print("\n  Label breakdown:")
    for l, c in zip(labels, counts):
        l = int(l)
        name = VICTRE_MU.get(l, (None, f"unknown_{l}"))[1]
        print(f"    {l:>3} {name:<22} {100*c/total:6.2f}% ({c:>14,d} voxels)")

    # Convert to mu (cm^-1) using the lookup. Anything not in the table
    # maps to 0 -- we only need a sensible render here.
    mu_lut = np.zeros(256, dtype=np.float32)
    for k, (mu, _) in VICTRE_MU.items():
        mu_lut[k] = mu
    vol_mu = mu_lut[vol_lbl]
    print(f"\n  mu range    : [{vol_mu.min():.4f}, {vol_mu.max():.4f}] cm^-1")
    print(f"  Non-zero %  : {100 * (vol_mu > 1e-6).mean():.2f}%")

    # Three orthogonal mid slices, same display window as the analytic
    # breast (vmax = 0.6 cm^-1).
    vmin, vmax = 0.0, 0.6

    # Axial (xy slice, perpendicular to z = compression).
    z_mid = nz // 2
    axial = vol_mu[z_mid]
    # Coronal (xz slice, perpendicular to y).
    y_mid = ny // 2
    coronal = vol_mu[:, y_mid, :]
    # Sagittal (yz slice, perpendicular to x).
    x_mid = nx // 2
    sagittal = vol_mu[:, :, x_mid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, img, label, extent in zip(
        axes,
        (axial, coronal, sagittal),
        (f"axial (z = {z_mid}, perpendicular to compression)",
         f"coronal (y = {y_mid})",
         f"sagittal (x = {x_mid})"),
        ((0, nx*dx/10, 0, ny*dy/10),
         (0, nx*dx/10, 0, nz*dz/10),
         (0, ny*dy/10, 0, nz*dz/10)),
    ):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="lower", aspect="equal", extent=extent)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("cm")
        ax.set_ylabel("cm")

    fig.suptitle(
        f"{args.source}  "
        f"({nx}x{ny}x{nz} @ {dx*10:.2f} mm  =  "
        f"{nx*dx/10:.1f} x {ny*dy/10:.1f} x {nz*dz/10:.1f} cm)",
        fontsize=11,
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
