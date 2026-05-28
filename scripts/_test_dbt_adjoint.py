"""Quick adjoint-test sweep over cone_vec u/v sign conventions for the DBT
geometry. We try the four sign combinations of u_y, v_y on the detector
basis, plus swapping u and v, and report which one gives a clean
adjoint.

Runs with a small phantom shape so it returns in seconds.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import astra  # noqa: E402
import victre_reconstruction as vr  # noqa: E402

np.random.seed(0)

PHANTOM_SHAPE = (32, 144, 144)  # (nz, ny, nx) — matches the breast
DX_CM = 0.15
DET_ROW = 240
DET_COL = 240
DET_SP = 0.05
NVIEWS = 25
ARC_DEG = 50.0
SOD = 65.0
ODD = 5.0


def build_geom(u_xyz, v_xyz):
    nz_d, ny_d, nx_d = PHANTOM_SHAPE
    vol_geom = astra.create_vol_geom(
        ny_d, nx_d, nz_d,
        -nx_d * DX_CM / 2, nx_d * DX_CM / 2,
        -ny_d * DX_CM / 2, ny_d * DX_CM / 2,
        -nz_d * DX_CM / 2, nz_d * DX_CM / 2,
    )
    angles = np.deg2rad(np.linspace(-ARC_DEG / 2, ARC_DEG / 2, NVIEWS))
    vectors = np.zeros((NVIEWS, 12), dtype=np.float64)
    for i, t in enumerate(angles):
        vectors[i, 0:3]  = (0.0, SOD * np.sin(t), SOD * np.cos(t))
        vectors[i, 3:6]  = (0.0, 0.0, -ODD)
        vectors[i, 6:9]  = u_xyz
        vectors[i, 9:12] = v_xyz
    proj_geom = astra.create_proj_geom("cone_vec", DET_ROW, DET_COL, vectors)
    return vol_geom, proj_geom


def adjoint_rel(A, At, vol_shape, sino_shape):
    x = np.random.randn(*vol_shape).astype(np.float32)
    y = np.random.randn(*sino_shape).astype(np.float32)
    lhs = float((A(x) * y).sum())
    rhs = float((x * At(y)).sum())
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    return lhs, rhs, rel


def main():
    sino_shape = (DET_ROW, NVIEWS, DET_COL)
    # Reference: existing circular cone geom (known good)
    print("=== reference: circular cone (existing build_geometry) ===")
    vg, pg, _ = vr.build_geometry(PHANTOM_SHAPE, DX_CM,
                                  det_row_count=DET_ROW, det_col_count=DET_COL,
                                  det_spacing=DET_SP, nviews=NVIEWS,
                                  arc_deg=ARC_DEG)
    A, At = vr.make_projector(vg, pg)
    lhs, rhs, rel = adjoint_rel(A, At, PHANTOM_SHAPE, sino_shape)
    print(f"   <Ax,y>={lhs:+.4e}  <x,Aty>={rhs:+.4e}  rel={rel:.3e}\n")

    # DBT geom: sweep u/v sign and swap conventions
    s = DET_SP
    trials = [
        ("u=(+s,0,0), v=(0,+s,0)", (+s, 0, 0), (0, +s, 0)),
        ("u=(+s,0,0), v=(0,-s,0)", (+s, 0, 0), (0, -s, 0)),
        ("u=(-s,0,0), v=(0,+s,0)", (-s, 0, 0), (0, +s, 0)),
        ("u=(-s,0,0), v=(0,-s,0)", (-s, 0, 0), (0, -s, 0)),
        ("swap: u=(0,+s,0), v=(+s,0,0)", (0, +s, 0), (+s, 0, 0)),
        ("swap: u=(0,-s,0), v=(+s,0,0)", (0, -s, 0), (+s, 0, 0)),
        ("swap: u=(0,+s,0), v=(-s,0,0)", (0, +s, 0), (-s, 0, 0)),
        ("swap: u=(0,-s,0), v=(-s,0,0)", (0, -s, 0), (-s, 0, 0)),
    ]
    print("=== DBT cone_vec orientation sweep ===")
    for label, u, v in trials:
        vg, pg = build_geom(u, v)
        A, At = vr.make_projector(vg, pg)
        lhs, rhs, rel = adjoint_rel(A, At, PHANTOM_SHAPE, sino_shape)
        flag = " <-- clean" if rel < 1e-2 else ""
        print(f"   {label}: rel={rel:.3e}  lhs={lhs:+.3e}  rhs={rhs:+.3e}{flag}")


if __name__ == "__main__":
    main()
