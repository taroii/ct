"""
BREAST PHANTOM
==============
Attenuation values are at 30 keV, in cm^-1.
Masses and inserts use DIFFERENTIAL attenuation (att = insert_att - background_att),
so that embedding is additive over the base tissue layer.

Tissue attenuation coefficients (absolute, 30 keV):
   breast_att      = 0.245   (fatty breast tissue)
   muscle_att      = 0.3972  (pectoralis muscle / chest wall)
   duct_att        = 0.3931  (ductal / glandular tissue)
   mass_att        = 0.40768 (soft tissue mass)
   dense_mass_att  = mass_att * 1.3  (dense mass, e.g. in pectoralis)
   dense_att       = 0.3931  (dense glandular, same as duct)
   calc_att        = 4.31    (calcification)
   metal_att       = 22.6    (metal artifact marker)

STRUCTURES (assembly order):
----------------------------------------------------------------------
1. BREAST MAIN         att = breast_att (absolute)
   truncated ellipsoid: ax=6.0 (length), ay=6.5, az=3.25
   clipped by: z in [-2.5, 2.5], x >= breast_back (0.0)

2. NIPPLE              net att = 0 vs breast (pos + neg cancel inside breast)
   nipple_pos: cylinder r=0.5, from x=0 to x=breast_length+0.5
   nipple_neg: same cylinder clipped by breast ellipsoid (removes interior overlap)

3. CRESCENT (ductal)   att = duct_att - breast_att (differential)
   thin shell of ductal tissue just inside the breast surface:
   crescent_pos: slightly smaller ellipsoid (all axes - 0.5), x >= 0 half
   crescent_neg: same ellipsoid clipped by large sphere (xc=-8, r=12)
                 hollows out the interior, leaving only the outer shell

4. CHEST WALL          att = muscle_att (absolute)
   box: x in [-2, 0], y in [-9.5, 9.5], z in [-2.5, 2.5]

5. MASSES IN PECTORALIS (3)   att = dense_mass_att - muscle_att
   mass11: ellipsoid ax=ay=0.5, az=0.375  at offset (-0.65, -2.0, 0)
   mass12: sphere r=0.5                   at offset (-0.65,  0.0, 0)
   mass13: ellipsoid ax=ay=0.5, az=0.25   at offset (-0.65, +2.0, 0)

6. MASSES IN FATTY TISSUE (6 spheres, varying radius)  att = mass_att - breast_att
   r1=0.125, r6=0.75, r2=0.25, r5=0.625, r3=0.375, r4=0.5
   (non-monotonic size order, at x offset +1.25, y offsets +-3.75 to +-0.75)

7. STACKED MASS PAIRS (3 pairs, r=0.5)  att = mass_att - breast_att
   all at x offset +3.0, y offsets -2/0/+2, z-separations 1.0/2.0/3.0 cm

8. PECTORALIS BORDER MASS (split object at x=0)
   mass4A: fatty-tissue half  att = mass_att - breast_att
   mass4B: muscle half        att = mass_att - muscle_att
   at offset (0.0, -4.0, 0)

9. MASS IN DUCTAL TISSUE  att = mass_att - duct_att
   sphere r=0.5  at offset (+4.75, 0, 0)

10. METAL ARTIFACT MARKER  att = metal_att - breast_att
    thin cylinder r=0.015, length=0.3, tilted alpha=pi/4, beta=pi/2
    at offset (+0.25, +4.5, 0)

11. CALCIFICATION CLUSTERS (4 clusters x 6 spheres = 24 total)
    Each cluster: 6 spheres in two z-planes (z=+-cc_rad=0.3),
    3 per plane at 120-deg intervals, planes rotated 60 deg from each other.

    cluster1: r=0.015  fatty tissue   att = calc_att - breast_att
              center offset (+2.5,  -3.5, 0)
    cluster2: r=0.015  dense tissue   att = calc_att - dense_att
              center offset (+4.15, -3.0, 0)
    cluster3: r=0.0075 fatty tissue   att = calc_att - breast_att
              center offset (+2.5,  +3.5, 0)
    cluster4: r=0.0075 dense tissue   att = calc_att - dense_att
              center offset (+4.15, +3.0, 0)

COORDINATE SYSTEM
   x : compression direction (breast points toward +x, chest wall at x<=0)
   y : lateral
   z : superior/inferior
   All offsets above are relative to (breast_xc, breast_yc, breast_zc).
   Demo uses breast_xc=0, breast_yc=0, breast_zc=0 (re-centered at origin).

   Full extent (demo): x=[-2.5, 8.5], y=[-10, 10], z=[-3, 3]  (in cm)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phantom3d import (
    pi, sin, cos,
    phantom3D, image3D,
    gen_object, planar, cylindrical, ellipsoidal
)


def build_breast_phantom(breast_xc=0., breast_yc=0., breast_zc=0.):
    """
    Builds the breast phantom. Returns a phantom3D object.

    Parameters
    ----------
    breast_xc, breast_yc, breast_zc : float
        Centre of the breast in world coordinates.
        Default (0,0,0) for standalone use; original codebase used (1.2, 0, -15)
        to position the breast within a full tomosynthesis acquisition geometry.
    """

    # ── attenuation coefficients (30 keV, cm^-1) ────────────────────────
    breast_att     = 0.245
    muscle_att     = 0.3972
    duct_att       = 0.3931
    mass_att       = 0.40768
    dense_mass_att = 0.40768 * 1.3
    dense_att      = 0.3931
    calc_att       = 4.31
    metal_att      = 22.6

    # ── geometry parameters ──────────────────────────────────────────────
    breast_length    = 6.0
    breast_width     = 6.5
    breast_thickness = 5.0
    breast_back      = 0.

    wall_width     = 2.0
    wall_length    = 19.0
    wall_thickness = 5.0

    crescent_diff = 0.5
    crx_sphere    = -8.
    cr_sphere_rad = 12.0

    nipple_length = 0.5
    nipple_width  = 0.5

    cc_rad = 0.3

    bx, by, bz = breast_xc, breast_yc, breast_zc

    # ── reusable mass surface primitives ────────────────────────────────
    def _ell(ax, ay, az):
        return ellipsoidal(ax=ax, ay=ay, az=az)

    mass_r1 = _ell(0.125, 0.125, 0.125)
    mass_r2 = _ell(0.25,  0.25,  0.25)
    mass_r3 = _ell(0.375, 0.375, 0.375)
    mass_r4 = _ell(0.5,   0.5,   0.5)
    mass_r5 = _ell(0.625, 0.625, 0.625)
    mass_r6 = _ell(0.75,  0.75,  0.75)

    mass_ell1 = _ell(0.5, 0.5, 0.375)
    mass_ell2 = _ell(0.5, 0.5, 0.25)

    # ── breast main: truncated ellipsoid ─────────────────────────────────
    zl = -breast_thickness / 2.
    zu =  breast_thickness / 2.

    ell_surf = ellipsoidal(ax=breast_length, ay=breast_width, az=breast_width/2.)

    breast_main = gen_object(
        x0=bx, y0=by, z0=bz, att=breast_att, nsurf=4,
        surfaces=[
            planar(zc=zl),
            planar(zc=zu, beta=pi),
            planar(xc=breast_back, beta=pi/2.),
            ell_surf,
        ])

    # ── chest wall ───────────────────────────────────────────────────────
    xl = breast_back - wall_width
    xu = breast_back
    yl = -wall_length / 2.
    yu =  wall_length / 2.

    chest_wall = gen_object(
        x0=bx, y0=by, z0=bz, att=muscle_att, nsurf=6,
        surfaces=[
            planar(xc=xl, beta=pi/2.),
            planar(xc=xu, alpha=pi, beta=pi/2.),
            planar(yc=yl, alpha=pi/2.,  beta=pi/2.),
            planar(yc=yu, alpha=-pi/2., beta=pi/2.),
            planar(zc=zl),
            planar(zc=zu, beta=pi),
        ])

    # ── nipple ───────────────────────────────────────────────────────────
    xu_nip  = breast_length + nipple_length
    nip_cyl = cylindrical(alpha=0., beta=pi/2., ax=nipple_width, ay=nipple_width)

    nipple_pos = gen_object(
        x0=bx, y0=by, z0=bz, att=breast_att, nsurf=3,
        surfaces=[
            planar(xc=0., beta=pi/2.),
            nip_cyl,
            planar(xc=xu_nip, alpha=pi, beta=pi/2.),
        ])

    nipple_neg = gen_object(
        x0=bx, y0=by, z0=bz, att=-breast_att, nsurf=3,
        surfaces=[
            planar(xc=0., beta=pi/2.),
            nip_cyl,
            ell_surf,
        ])

    # ── crescent (ductal shell) ───────────────────────────────────────────
    ell_crescent = ellipsoidal(
        ax=breast_length - crescent_diff,
        ay=breast_width  - crescent_diff,
        az=breast_width / 2. - crescent_diff)

    sphere_crescent = ellipsoidal(
        xc=crx_sphere,
        ax=cr_sphere_rad, ay=cr_sphere_rad, az=cr_sphere_rad)

    cr_side = planar(xc=0., beta=pi/2.)

    crescent_pos = gen_object(
        x0=bx, y0=by, z0=bz, att=duct_att - breast_att, nsurf=2,
        surfaces=[cr_side, ell_crescent])

    crescent_neg = gen_object(
        x0=bx, y0=by, z0=bz, att=-(duct_att - breast_att), nsurf=3,
        surfaces=[cr_side, ell_crescent, sphere_crescent])

    # ── masses in pectoralis ─────────────────────────────────────────────
    mass11 = gen_object(
        x0=bx-0.65, y0=by-2., z0=bz,
        att=dense_mass_att - muscle_att, nsurf=1, surfaces=[mass_ell1])
    mass12 = gen_object(
        x0=bx-0.65, y0=by,    z0=bz,
        att=dense_mass_att - muscle_att, nsurf=1, surfaces=[mass_r4])
    mass13 = gen_object(
        x0=bx-0.65, y0=by+2., z0=bz,
        att=dense_mass_att - muscle_att, nsurf=1, surfaces=[mass_ell2])

    # ── masses in fatty tissue ────────────────────────────────────────────
    fatty_specs = [
        (1.25, -3.75, mass_r1),
        (1.25, -2.25, mass_r6),
        (1.25, -0.75, mass_r2),
        (1.25,  0.75, mass_r5),
        (1.25,  2.25, mass_r3),
        (1.25,  3.75, mass_r4),
    ]
    fatty_masses = [
        gen_object(x0=bx+dx, y0=by+dy, z0=bz,
                   att=mass_att - breast_att, nsurf=1, surfaces=[surf])
        for dx, dy, surf in fatty_specs
    ]

    # ── stacked mass pairs ────────────────────────────────────────────────
    stacked = []
    for dy, dza, dzb in [(-2., -1.5, -0.5),
                         ( 0., -1.5,  0.5),
                         ( 2., -1.5,  1.5)]:
        stacked.append(gen_object(x0=bx+3., y0=by+dy, z0=bz+dza,
                                  att=mass_att - breast_att, nsurf=1,
                                  surfaces=[mass_r4]))
        stacked.append(gen_object(x0=bx+3., y0=by+dy, z0=bz+dzb,
                                  att=mass_att - breast_att, nsurf=1,
                                  surfaces=[mass_r4]))

    # ── pectoralis border mass ────────────────────────────────────────────
    mass4A = gen_object(
        x0=bx, y0=by-4., z0=bz,
        att=mass_att - breast_att, nsurf=2,
        surfaces=[mass_r4, planar(xc=0., beta=pi/2.)])
    mass4B = gen_object(
        x0=bx, y0=by-4., z0=bz,
        att=mass_att - muscle_att, nsurf=2,
        surfaces=[mass_r4, planar(xc=0., alpha=pi, beta=pi/2.)])

    # ── mass in ductal tissue ─────────────────────────────────────────────
    mass5 = gen_object(
        x0=bx+4.75, y0=by, z0=bz,
        att=mass_att - duct_att, nsurf=1, surfaces=[mass_r4])

    # ── metal artifact marker ─────────────────────────────────────────────
    ang1, ang2   = pi/4., pi/2.
    metal_length = 0.3
    metal_rad    = 0.015

    metal = gen_object(
        x0=bx+0.25, y0=by+4.5, z0=bz,
        att=metal_att - breast_att, nsurf=3,
        surfaces=[
            planar(xc=0., yc=0., zc=0., alpha=ang1, beta=ang2),
            cylindrical(alpha=ang1, beta=ang2, ax=metal_rad, ay=metal_rad),
            planar(xc=metal_length, yc=metal_length, zc=metal_length,
                   alpha=ang1 + pi, beta=ang2),
        ])

    # ── calcification clusters ────────────────────────────────────────────
    calc_sphere       = ellipsoidal(ax=0.015,  ay=0.015,  az=0.015)
    calc_sphere_small = ellipsoidal(ax=0.015,  ay=0.0075, az=0.015)

    def _calc_cluster(dx, dy, dz, diff_att, sphere):
        pts = []
        for plane_z, plane_offs in [(-cc_rad, 20./180.*pi),
                                    ( cc_rad, 80./180.*pi)]:
            for i in range(3):
                ang = plane_offs + i * 120./180.*pi
                pts.append(gen_object(
                    x0=bx + dx + cc_rad * sin(ang),
                    y0=by + dy + cc_rad * cos(ang),
                    z0=bz + dz + plane_z,
                    att=diff_att, nsurf=1, surfaces=[sphere]))
        return pts

    cluster1 = _calc_cluster( 2.5,  -3.5, 0., calc_att - breast_att, calc_sphere)
    cluster2 = _calc_cluster( 4.15, -3.0, 0., calc_att - dense_att,  calc_sphere)
    cluster3 = _calc_cluster( 2.5,   3.5, 0., calc_att - breast_att, calc_sphere_small)
    cluster4 = _calc_cluster( 4.15,  3.0, 0., calc_att - dense_att,  calc_sphere_small)

    # ── assemble ──────────────────────────────────────────────────────────
    phantom = phantom3D()
    for comp in ([breast_main, nipple_pos, nipple_neg,
                  crescent_pos, crescent_neg, chest_wall,
                  mass11, mass12, mass13]
                 + fatty_masses
                 + stacked
                 + [mass4A, mass4B, mass5, metal]
                 + cluster1 + cluster2 + cluster3 + cluster4):
        phantom.add_component(comp)

    return phantom


if __name__ == '__main__':
    print("Building breast phantom ...")
    breast = build_breast_phantom()

    print("Embedding in 3D image array ...")
    image = image3D(shape=(220, 300, 60),
                    xlen=11., ylen=20., zlen=6.,
                    x0=-2.5, y0=-10., z0=-3.)
    breast.embed_in(image)
    vol = image.mat

    print(f"  Volume shape : {vol.shape}")
    print(f"  Value range  : [{vol.min():.4f}, {vol.max():.4f}]")

    proj_z = vol.sum(axis=2)
    proj_x = vol.sum(axis=0)
    proj_y = vol.sum(axis=1)
    mid_z  = vol[:, :, vol.shape[2]//2]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#0d0d0d')
    fig.suptitle("Breast Phantom -- Projections & Central Slice",
                 color='white', fontsize=14, y=1.01)

    panels = [
        (proj_z.T,  "Axial projection  (sum z)",  "x (compression)", "y (lateral)"),
        (proj_x.T,  "MLO projection (sum x)",      "y (lateral)",     "z (SI)"),
        (proj_y.T,  "CC projection  (sum y)",       "x (compression)", "z (SI)"),
        (mid_z.T,   "Central axial slice",           "x (compression)", "y (lateral)"),
    ]

    for ax, (data, title, xlabel, ylabel) in zip(axes, panels):
        ax.set_facecolor('#0d0d0d')
        im = ax.imshow(data, cmap='bone', origin='lower', aspect='equal')
        ax.set_title(title, color='#e0e0e0', fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color='#888888', fontsize=8)
        ax.set_ylabel(ylabel, color='#888888', fontsize=8)
        ax.tick_params(colors='#555555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
            colors='#888888', labelsize=7)

    plt.tight_layout()
    out_path = 'breast_phantom_projections.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"Plot saved to {out_path}")