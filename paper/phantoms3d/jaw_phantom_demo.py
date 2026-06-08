"""
1. BODY (soft tissue)  att = 1.0
   neck1        right elliptic cylinder  r=6.0      z=0.0-3.0    full circle cross-section
   neck2        right elliptic cylinder  r=6.0      z=3.0-10.0   front half only (y>=0)
   throat       box                      5x2x5      centered at (0, 4, 6)
   tongue1      right elliptic cylinder  r=2.0      z=3.5-5.0    front half only (y>=0)
   tongue2      box                      4x2.5x1.5  centered at (0, 6.25, 4.25)
   head1        right elliptic cylinder  r=6.0      z=3.0-10.0   front half only (y>=0)
   head2        box                      12x6x7     centered at (0, 3, 6.5)
   disc1/2/3    right elliptic cylinder  2.5x1.5    height=1.0   at z=1.5, 4.5, 7.5  (vertebral discs)

   -- holes carved out of body (att = -1.0) --
   spine_hole   right elliptic cylinder  2.5x1.5    z=0-10       spinal canal cavity
   trachea_hole right elliptic cylinder  r=0.75     z=0-5        airway
   jaw_hole1    right elliptic cylinder  r=3.55     z=3.5-8.5    oral cavity (front half)
   jaw_hole2    box                      7.1x4.5x5  centered at (0, 5.25, 6)
   jaw_hole3    right elliptic cylinder  r=2.0      z=5.0-10.0   palate cavity (back half)
   sinus_hole1  right elliptic cylinder  r=2.0      z=8.5-10.0   sinus (front half)
   sinus_hole2  box                      4x2.5x1.5  centered at (0, 6.25, 9.25)
   chord_holes  3x small cylinders       r=0.5      at z=1.5,4.5,7.5  spinal chord
   tumor_hole   ellipsoid                r=0.5      at (-1, 7, 4)     cavity for tumor

2. BONE                att = 2.0
   spine1       right elliptic cylinder  2.5x1.5    z=0-10       vertebral column
   lower_jaw1   right elliptic cylinder  r=3.55     z=3.5-5.0    lower jaw arc (front half)
   lower_jaw2   box                      7.1x2.5x1.5 at (0, 6.25, 4.25)
   back_jaw1/2  box                      1.05x2x5   at x=+-3.025  jaw rami (vertical struts)
   upper_jaw1   right elliptic cylinder  r=3.55     z=7.0-8.5    upper jaw arc (front half)
   upper_jaw2   box                      7.1x2.5x1.5 at (0, 6.25, 7.75)

   -- holes carved out of bone (att = -2.0) --
   tongue_hole1/2  same shape as tongue1/2      hollow center of lower jaw
   roof_hole1/2    right elliptic cyl + box     hollow center of upper jaw / palate
   disc_holes      3x elliptic cylinders        removes bone at disc locations
   chord_holes     4x small cylinders r=0.5     spinal cord channel through bone

3. MARROW              att = 1.1
   spine_marrow right elliptic cylinder  r=0.5      z=0-10
                sits inside the spine bone, slightly posterior (y=-1.5)
                subtle contrast vs soft tissue (1.1 vs 1.0)

4. TEETH               att = 2.2
   28 cylinders total arranged in two arcs:

   Lower arch  z=5.475  (14 teeth, 7 per side):
     molars    r=0.5    at x=+-3.0,  y=5.5-7.5
     premolars r=0.4    curving inward to x~0, y~10.6 (front incisors)

   Upper arch  z=6.525  (14 teeth, 7 per side):
     same x/y layout as lower arch

   Three lower-arch positions are missing (replaced by crowns):
     (3.0, 6.5) and (3.0, 7.5) on right, (-3.0, 7.5) on left

5. GOLD CROWNS         att = 19.3
   3 cylinders  r=0.5, height=0.95, all at z=5.475 (lower arch):
     crown1   ( 3.0,  6.5)   right side, 2nd molar
     crown2   ( 3.0,  7.5)   right side, 3rd molar
     crown3   (-3.0,  6.5)   left side,  2nd molar

6. CANCER (tumor)      att = 1.01
   1 ellipsoid  r=0.5 (isotropic)  at (-1.0, 7.0, 4.0)
   Sits in the soft tissue of the tongue/floor of mouth region.
   Contrast vs surrounding soft tissue: 1.01 vs 1.0 -- nearly invisible.
   Left of center (x=-1), away from the crowns.

COORDINATE SYSTEM
   x : left/right  (patient left = positive x)
   y : front/back  (anterior = positive y)
   z : inferior/superior  (head up = increasing z)

   Full extent: x=[-10,10], y=[-6,14], z=[0,12]  (in cm)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phantom3d import (
    pi,
    tissue_map, phantom3D, image3D,
    ellipsoid as Ellipsoid,
    gen_object, planar, cylindrical
)

def right_elliptical_cylinder(x0=0., y0=0., z0=0., att=1.,
                               ax=1., ay=1., length=1.,
                               alpha=0., beta=0., gamma=0.):
    half_len = length / 2.
    ell_cyl     = cylindrical(ax=ax, ay=ay)
    ell_cyl_top = planar(zc= half_len, beta=pi)
    ell_cyl_bot = planar(zc=-half_len)
    obj = gen_object(x0=x0, y0=y0, z0=z0, att=att, nsurf=3,
                     surfaces=[ell_cyl, ell_cyl_top, ell_cyl_bot])
    obj.rotate(alpha=alpha, beta=beta, gamma=gamma)
    return obj

def box(x0=0., y0=0., z0=0., att=1.,
        lenx=1., leny=1., lenz=1.,
        alpha=0., beta=0., gamma=0.):
    hx, hy, hz = lenx/2., leny/2., lenz/2.
    xside1 = planar(xc= hx, beta=pi/2, alpha=pi)
    xside2 = planar(xc=-hx, beta=pi/2)
    yside1 = planar(yc= hy, alpha=-pi/2, beta=pi/2)
    yside2 = planar(yc=-hy, alpha= pi/2, beta=pi/2)
    zside1 = planar(zc= hz, beta=pi)
    zside2 = planar(zc=-hz)
    obj = gen_object(x0=x0, y0=y0, z0=z0, att=att, nsurf=6,
                     surfaces=[xside1, xside2, yside1, yside2, zside1, zside2])
    obj.rotate(alpha=alpha, beta=beta, gamma=gamma)
    return obj

def build_jaw_phantom():
    body_att      = 1.0
    bone_att      = 2.0
    marrow_att    = 1.1
    teeth_att     = 2.2
    gold_crown_att = 19.3
    cancer_att    = 1.01

    # marrow 
    marrow_map = tissue_map()
    marrow_map.add_component(right_elliptical_cylinder(
        x0=0., y0=-1.5, z0=5., att=marrow_att, ax=.5, ay=.5, length=10.))

    # cancer 
    cancer_map = tissue_map()
    cancer_map.add_component(Ellipsoid(
        x0=-1., y0=7., z0=4., att=cancer_att, ax=.5, ay=.5, az=.5))

    # body 
    body_map = tissue_map()
    neck1 = right_elliptical_cylinder(
        x0=0., y0=0., z0=1.5, att=body_att, ax=6., ay=6., length=3.)
    neck2 = right_elliptical_cylinder(
        x0=0., y0=0., z0=6.5, att=body_att, ax=6., ay=6., length=7.)
    neck2.add_surface(planar(yc=0., alpha=-pi/2, beta=pi/2))

    throat = box(x0=0., y0=4., z0=6., att=body_att, lenx=5., leny=2., lenz=5.)

    tongue1 = right_elliptical_cylinder(
        x0=0., y0=7.5, z0=4.25, att=body_att, ax=2., ay=2., length=1.5)
    tongue1.add_surface(planar(yc=0., alpha=pi/2, beta=pi/2))
    tongue2 = box(x0=0., y0=6.25, z0=4.25, att=body_att, lenx=4., leny=2.5, lenz=1.5)

    head1 = right_elliptical_cylinder(
        x0=0., y0=6., z0=6.5, att=body_att, ax=6., ay=6., length=7.)
    head1.add_surface(planar(yc=0., alpha=pi/2, beta=pi/2))
    head2 = box(x0=0., y0=3., z0=6.5, att=body_att, lenx=12., leny=6., lenz=7.)

    spine_hole = right_elliptical_cylinder(
        x0=0., y0=-1., z0=5., att=-body_att, ax=2.5, ay=1.5, length=10.)
    trachea_hole = right_elliptical_cylinder(
        x0=0., y0=3.75, z0=2.5, att=-body_att, ax=.75, ay=.75, length=5.)
    tumor_hole = Ellipsoid(
        x0=-1., y0=7., z0=4., att=-body_att, ax=.5, ay=.5, az=.5)

    jaw_hole1 = right_elliptical_cylinder(
        x0=0., y0=7.5, z0=6., att=-body_att, ax=3.55, ay=3.55, length=5.)
    jaw_hole1.add_surface(planar(yc=0., alpha=pi/2, beta=pi/2))
    jaw_hole2 = box(x0=0., y0=5.25, z0=6., att=-body_att, lenx=7.1, leny=4.5, lenz=5.)
    jaw_hole3 = right_elliptical_cylinder(
        x0=0., y0=5., z0=7.5, att=-body_att, ax=2., ay=2., length=5.)
    jaw_hole3.add_surface(planar(yc=0., alpha=-pi/2, beta=pi/2))

    sinus_hole1 = right_elliptical_cylinder(
        x0=0., y0=7.5, z0=9.25, att=-body_att, ax=2., ay=2., length=1.5)
    sinus_hole1.add_surface(planar(yc=0., alpha=pi/2, beta=pi/2))
    sinus_hole2 = box(
        x0=0., y0=6.25, z0=9.25, att=-body_att, lenx=4., leny=2.5, lenz=1.5)

    for comp in [neck1, neck2, head1, head2, throat, tongue1, tongue2,
                 trachea_hole, spine_hole, jaw_hole1, jaw_hole2, jaw_hole3,
                 sinus_hole1, sinus_hole2, tumor_hole]:
        body_map.add_component(comp)

    # chord holes + discs
    for z_ch, z_disc in [(1.5, 1.5), (4.5, 4.5), (7.5, 7.5)]:
        body_map.add_component(right_elliptical_cylinder(
            x0=0., y0=-1.5, z0=z_ch, att=-body_att, ax=.5, ay=.5, length=1.))
        body_map.add_component(right_elliptical_cylinder(
            x0=0., y0=-1., z0=z_disc, att=body_att, ax=2.5, ay=1.5, length=1.))

    # bone 
    bone_map = tissue_map()
    spine1 = right_elliptical_cylinder(
        x0=0., y0=-1., z0=5., att=bone_att, ax=2.5, ay=1.5, length=10.)

    lower_jaw1 = right_elliptical_cylinder(
        x0=0., y0=7.5, z0=4.25, att=bone_att, ax=3.55, ay=3.55, length=1.5)
    lower_jaw1.add_surface(planar(yc=0., alpha=pi/2, beta=pi/2))
    lower_jaw2 = box(x0=0., y0=6.25, z0=4.25, att=bone_att, lenx=7.1, leny=2.5, lenz=1.5)

    back_jaw1 = box(x0= 3.025, y0=4., z0=6., att=bone_att, lenx=1.05, leny=2., lenz=5.)
    back_jaw2 = box(x0=-3.025, y0=4., z0=6., att=bone_att, lenx=1.05, leny=2., lenz=5.)

    upper_jaw1 = right_elliptical_cylinder(
        x0=0., y0=7.5, z0=7.75, att=bone_att, ax=3.55, ay=3.55, length=1.5)
    upper_jaw1.add_surface(planar(yc=0., alpha=pi/2, beta=pi/2))
    upper_jaw2 = box(x0=0., y0=6.25, z0=7.75, att=bone_att, lenx=7.1, leny=2.5, lenz=1.5)

    for comp in [spine1, lower_jaw1, lower_jaw2, back_jaw1, back_jaw2,
                 upper_jaw1, upper_jaw2]:
        bone_map.add_component(comp)

    # bone holes
    tongue_hole1 = right_elliptical_cylinder(
        x0=0., y0=7.5, z0=4.25, att=-bone_att, ax=2., ay=2., length=1.5)
    tongue_hole1.add_surface(planar(yc=0., alpha=pi/2, beta=pi/2))
    tongue_hole2 = box(
        x0=0., y0=6.25, z0=4.25, att=-bone_att, lenx=4., leny=2.5, lenz=1.5)
    roof_hole1 = right_elliptical_cylinder(
        x0=0., y0=7.5, z0=7.5, att=-bone_att, ax=2., ay=2., length=1.)
    roof_hole1.add_surface(planar(yc=0., alpha=pi/2, beta=pi/2))
    roof_hole2 = box(
        x0=0., y0=6.25, z0=7.5, att=-bone_att, lenx=4., leny=2.5, lenz=1.)
    for comp in [tongue_hole1, tongue_hole2, roof_hole1, roof_hole2]:
        bone_map.add_component(comp)

    for z_ch, z_disc in [(0.5, 1.5), (4.5, 4.5), (7.5, 7.5)]:
        bone_map.add_component(right_elliptical_cylinder(
            x0=0., y0=-1.5, z0=z_ch, att=-bone_att, ax=.5, ay=.5, length=1.))
        bone_map.add_component(right_elliptical_cylinder(
            x0=0., y0=-1., z0=z_disc, att=-bone_att, ax=2.5, ay=1.5, length=1.))

    # crowns 
    crown_map = tissue_map()
    for cx, cy in [(3., 6.5), (3., 7.5), (-3., 6.5)]:
        crown_map.add_component(right_elliptical_cylinder(
            x0=cx, y0=cy, z0=5.475, att=gold_crown_att, ax=.5, ay=.5, length=.95))

    # teeth
    teeth_map = tissue_map()
    teeth_loc = [
        (3.0, 5.5, 5.475),
        (2.9, 8.4, 5.475), (2.5, 9.2, 5.475), (1.9, 9.9, 5.475),
        (1.2, 10.4, 5.475), (0.4, 10.6, 5.475),
        (3.0, 5.5, 6.525), (3.0, 6.5, 6.525), (3.0, 7.5, 6.525),
        (2.9, 8.4, 6.525), (2.5, 9.2, 6.525), (1.9, 9.9, 6.525),
        (1.2, 10.4, 6.525), (0.4, 10.6, 6.525),
        (-3.0, 5.5, 5.475),
        (-3.0, 7.5, 5.475), (-2.9, 8.4, 5.475), (-2.5, 9.2, 5.475),
        (-1.9, 9.9, 5.475), (-1.2, 10.4, 5.475), (-0.4, 10.6, 5.475),
        (-3.0, 5.5, 6.525), (-3.0, 6.5, 6.525),
        (-2.9, 8.4, 6.525), (-2.5, 9.2, 6.525), (-1.9, 9.9, 6.525),
        (-1.2, 10.4, 6.525), (-0.4, 10.6, 6.525),
    ]
    for tloc in teeth_loc:
        r = 0.5 if abs(tloc[0]) > 2.99 else 0.4
        teeth_map.add_component(right_elliptical_cylinder(
            x0=tloc[0], y0=tloc[1], z0=tloc[2],
            att=teeth_att, ax=r, ay=r, length=.95))

    # assemble 
    jaw = phantom3D()
    jaw.add_component(crown_map)
    jaw.add_component(teeth_map)
    jaw.add_component(body_map)
    jaw.add_component(bone_map)
    jaw.add_component(marrow_map)
    jaw.add_component(cancer_map)
    return jaw

if __name__ == '__main__':
    print("Building jaw phantom ...")
    jaw = build_jaw_phantom()

    print("Embedding in 3D image array ...")
    image = image3D(shape=(150, 150, 80),
                    xlen=20., ylen=20., zlen=12.,
                    x0=-10., y0=-6., z0=0.)
    jaw.embed_in(image)
    vol = image.mat
    print(f"  Volume shape : {vol.shape}")
    print(f"  Value range  : [{vol.min():.3f}, {vol.max():.3f}]")

    # projections
    # Sum (Beer-Lambert line integrals) along each axis
    proj_z  = vol.sum(axis=2)          # DRR: looking down  (axial)
    proj_x  = vol.sum(axis=0)          # lateral view
    proj_y  = vol.sum(axis=1)          # PA/AP view
    mid_z   = vol[:, :, vol.shape[2]//2]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5),
                             facecolor='#0d0d0d')
    fig.suptitle("Jaw Phantom — Projections & Central Slice",
                 color='white', fontsize=14, y=1.01)

    panels = [
        (proj_z.T,  "Axial projection  (sum z)",  "x →", "← y"),
        (proj_x.T,  "Lateral projection (sum x)",  "y →", "z ↑"),
        (proj_y.T,  "AP projection  (sum y)",       "x →", "z ↑"),
        (mid_z.T,   "Central axial slice",           "x →", "← y"),
    ]

    for ax, (data, title, xlabel, ylabel) in zip(axes, panels):
        ax.set_facecolor('#0d0d0d')
        im = ax.imshow(data, cmap='inferno', origin='lower', aspect='equal')
        ax.set_title(title, color='#e0e0e0', fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color='#888888', fontsize=8)
        ax.set_ylabel(ylabel, color='#888888', fontsize=8)
        ax.tick_params(colors='#555555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
            colors='#888888', labelsize=7)

    plt.tight_layout()
    out_path = 'jaw_phantom_projections.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d')
    print(f"Plot saved to {out_path}")