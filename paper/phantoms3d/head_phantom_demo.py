"""
1. BONE                att = 1.8
   outer_ell    ellipsoid                9.6x12.0x12.5   skull outer surface
   inner_ell    ellipsoid (hole)        -9.0x11.4x11.9   skull inner surface (carves shell)
   sinus_bone1  ellipsoid                tilted beta=-pi/4   left frontal sinus wall  at (-1.9, 5.4, 0)
   sinus_bone2  ellipsoid                tilted beta=+pi/4   right frontal sinus wall at ( 1.9, 5.4, 0)
   sinus_bone3  right elliptic cylinder  4.0x1.2         nasal bridge             at (0, 3.6, 0)    rotated alpha=pi/2, beta=pi/3
   sinus_bone4  right elliptic cylinder  2.0x0.53        zygomatic arch stub      at (0, 9.6, 0)    rotated beta=pi/2, gamma=-pi/6
   sinus_bone5  right elliptic cylinder  1.8x0.24        left orbital plate       at (-4.3, 6.8, -1) length=4, tilted gamma=-pi/6
   sinus_bone6  right elliptic cylinder  1.8x0.24        right orbital plate      at ( 4.3, 6.8, -1) length=4, tilted gamma=+pi/6
   protuberance cone intersect inner ellipsoid           occipital protuberance   at (0, -10, 0.2)
   ear_right    ellipsoid intersect inner ellipsoid  4.2x1.8x1.8  right temporal bone  at (9.1, 0, 0)
   ear_holes    ~300 small ellipsoids     r=0.15          hex-packed porous mastoid structure
                arranged in 7 z-layers (z=0, +-sha, +-2sha, +-3sha), sha=0.2*sqrt(3)
                x range ~5.6-8.6, tapering inward at higher |z| and |y|
   dot1/2/3     ellipsoids               r=0.10/0.05/0.02  bone calibration markers at (-8.0,0), (-8.5,+-0.5)

2. EYES                att = 1.06
   left_eye     ellipsoid                r=2.0           at (-4.7, 4.3, 0.872)
   right_eye    ellipsoid                r=2.0           at ( 4.7, 4.3, 0.872)

3. VENTRICLE           att = 1.045
   ventricle    ellipsoid                1.8x3.6x3.6     at (0, -3.6, 0)   posterior/central brain

4. FORBILD SPOT 1      att = 1.0525  (slightly hyper-dense vs brain)
   spot1        ellipsoid                r=0.4           at (-1.08, -9.0, 0)   posterior brain, left

5. FORBILD SPOT 2      att = 1.0475  (slightly hypo-dense vs brain)
   spot2        ellipsoid                r=0.4           at ( 1.08, -9.0, 0)   posterior brain, right
                Spots 1 and 2 are symmetric about x=0, separated by 2.16 cm.
                Contrast vs brain: +-0.0025 -- extremely subtle, low-contrast detection challenge.

6. HEMATOMA            att = 1.055
   hematoma     ellipsoid                1.2x0.42x1.4    at (6.39, -6.39, 0)
                Tilted beta=58.1 deg in the xy plane, sits in the right posterior quadrant.
                Elongated and rotated to simulate a subdural bleed along the skull inner surface.

7. BRAIN               att = 1.05
   brain        ellipsoid                9.0x11.4x11.9   fills inner skull (same dims as inner_ell)

   -- holes carved out of brain (att = -1.05) --
   ear_right_hole    same shape as ear_right          removes brain from ear cavity
   protuberance_hole same shape as protuberance       removes brain from protuberance
   left/right eye holes  same shapes as eyes          removes brain from eye sockets
   sinus_bone1/2/3/5/6 holes  same shapes as bones    removes brain from sinus regions
   sinus_hole    ellipsoid   1.8x3.0x3.0              frontal sinus air cavity  at (0, 8.4, 0)
   ventricle_hole   same shape as ventricle            removes brain from ventricle
   spot1/2_holes    same shapes as spots               removes brain from spot locations
   hematoma_hole    same shape as hematoma             removes brain from hematoma location
   dot1/2/3_holes   same shapes as bone dots           removes brain from marker locations

COORDINATE SYSTEM
   x : left/right  (patient left = positive x)
   y : anterior/posterior  (front of face = positive y)
   z : inferior/superior   (head up = increasing z)
   Origin at isocentre of the head.

   Full extent: x=[-13,13], y=[-13,13], z=[-13/16, 13/16]  (in cm)
   Default image: 512x512x32, 26cm FOV, thin axial slab (26/16 cm thick)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phantom3d import (
    pi, sqrt,
    tissue_map, phantom3D, image3D,
    ellipsoid, gen_object,
    planar, cylindrical, ellipsoidal, conical
)

def right_elliptical_cylinder(x0=0., y0=0., z0=0., att=1.,
                               ax=1., ay=1., length=1.,
                               alpha=0., beta=0., gamma=0.):
    half_len = length / 2.
    obj = gen_object(x0=x0, y0=y0, z0=z0, att=att, nsurf=3, surfaces=[
        cylindrical(ax=ax, ay=ay),
        planar(zc= half_len, beta=pi),
        planar(zc=-half_len),
    ])
    obj.rotate(alpha=alpha, beta=beta, gamma=gamma)
    return obj

def make_ear_holes(bone_att=1.8):
    """
    Hex-packed array of small ellipsoids that approximate the porous
    trabecular structure of the temporal bone / mastoid region.
    ~300 ellipsoids, all at x >= 5.6, centred around (x~7, y~0, z~0).
    """
    eh = []
    rad = 0.15
    sha = 0.2 * sqrt(3.)

    # Each 'floor' is a 2D hex layer at a fixed z offset.
    # Rows within each layer step in y by sha, with x starting further right
    # as |y| increases (approximating a circular cross-section).
    layers = [
        # (z_offset, row_specs)  row_spec = (x_start, n_points, y_offset)
        (0.,      [(5.6,8,0.), (5.8,7, sha),(6.0,7, 2*sha),(6.6,5, 3*sha),
                               (5.8,7,-sha),(6.0,7,-2*sha),(6.6,5,-3*sha)]),
        ( sha,    [(5.8,7,0.), (6.0,7, sha),(6.2,6, 2*sha),(6.8,5, 3*sha),
                               (6.0,7,-sha),(6.2,6,-2*sha),(6.8,5,-3*sha)]),
        (-sha,    [(5.8,7,0.), (6.0,7, sha),(6.2,6, 2*sha),(6.8,5, 3*sha),
                               (6.0,7,-sha),(6.2,6,-2*sha),(6.8,5,-3*sha)]),
        ( 2*sha,  [(6.0,7,0.), (6.2,6, sha),(6.4,6, 2*sha),(7.0,4, 3*sha),
                               (6.2,6,-sha),(6.4,6,-2*sha),(7.0,4,-3*sha)]),
        (-2*sha,  [(6.0,7,0.), (6.2,6, sha),(6.4,6, 2*sha),(7.0,4, 3*sha),
                               (6.2,6,-sha),(6.4,6,-2*sha),(7.0,4,-3*sha)]),
        ( 3*sha,  [(6.6,5,0.), (6.8,5, sha),(7.0,4, 2*sha),(7.6,3, 3*sha),
                               (6.8,5,-sha),(7.0,4,-2*sha),(7.6,3,-3*sha)]),
        (-3*sha,  [(6.6,5,0.), (6.8,5, sha),(7.0,4, 2*sha),(7.6,3, 3*sha),
                               (6.8,5,-sha),(7.0,4,-2*sha),(7.6,3,-3*sha)]),
    ]

    for z_off, rows in layers:
        for x_start, n, y_off in rows:
            for i in range(n):
                eh.append(ellipsoid(
                    x0=x_start + 0.4*i, y0=y_off, z0=z_off,
                    att=-bone_att, ax=rad, ay=rad, az=rad))
    return eh

def build_head_phantom():
    """Builds the full head phantom. Returns a phantom3D object."""

    bone_att      = 1.8
    eye_att       = 1.06
    brain_att     = 1.05
    ventricle_att = 1.045
    spot1_att     = 1.0525
    spot2_att     = 1.0475
    hematoma_att  = 1.055

    # skull radii
    ax_oe, ay_oe, az_oe = 9.6, 12.0, 12.5   # outer skull
    ax_ie, ay_ie, az_ie = 9.0, 11.4, 11.9   # inner skull / brain boundary

    # bone map 
    bone_map = tissue_map(physical_material=False)

    # skull shell = outer ellipsoid - inner ellipsoid
    bone_map.add_component(ellipsoid(att=bone_att,
                                     ax=ax_oe, ay=ay_oe, az=az_oe))
    bone_map.add_component(ellipsoid(att=-bone_att,
                                     ax=ax_ie, ay=ay_ie, az=az_ie))

    # frontal sinuses (paired, tilted ellipsoids)
    bone_map.add_component(ellipsoid(x0=-1.9, y0=5.4, z0=0., att=bone_att,
                                     ax=1.165, ay=0.406, az=3.,
                                     alpha=0., beta=-pi/4.))
    bone_map.add_component(ellipsoid(x0= 1.9, y0=5.4, z0=0., att=bone_att,
                                     ax=1.165, ay=0.406, az=3.,
                                     alpha=0., beta= pi/4.))

    # nasal bridge bone (short rotated cylinder)
    bone_map.add_component(right_elliptical_cylinder(
        x0=0., y0=3.6, z0=0., att=bone_att,
        ax=4.0, ay=1.2, length=0.483,
        alpha=pi/2., beta=pi/3.))

    # zygomatic arch stub (short flat cylinder, no sinus hole)
    bone_map.add_component(right_elliptical_cylinder(
        x0=0., y0=9.6, z0=0., att=bone_att,
        ax=2.0, ay=0.5256, length=0.4,
        alpha=0., beta=pi/2., gamma=-pi/6.))

    # left/right orbital plates (long thin tilted cylinders)
    bone_map.add_component(right_elliptical_cylinder(
        x0=-4.3, y0=6.8, z0=-1., att=bone_att,
        ax=1.8, ay=0.24, length=4.,
        alpha=0., beta=0., gamma=-pi/6.))
    bone_map.add_component(right_elliptical_cylinder(
        x0= 4.3, y0=6.8, z0=-1., att=bone_att,
        ax=1.8, ay=0.24, length=4.,
        alpha=0., beta=0., gamma= pi/6.))

    # occipital protuberance (cone intersected with inner skull ellipsoid)
    pro_cone  = conical(xc=0., yc=-10., zc=0.2,
                        alpha=pi/2., beta=-pi/2., ax=0.5, ay=0.2)
    skull_ell = ellipsoidal(ax=ax_ie, ay=ay_ie, az=az_ie)
    bone_map.add_component(gen_object(att=bone_att, nsurf=2,
                                      surfaces=[pro_cone, skull_ell]))

    # right ear bony protrusion
    ear_ell   = ellipsoidal(ax=4.2, ay=1.8, az=1.8)
    skull_ell2 = ellipsoidal(xc=-9.1, ax=ax_ie, ay=ay_ie, az=az_ie)
    bone_map.add_component(gen_object(x0=9.1, y0=0., z0=0., att=bone_att,
                                      nsurf=2, surfaces=[ear_ell, skull_ell2]))

    # ear porous holes (NOTE: ~300 ellipsoids — slow to embed)
    print("  Building ear hole structure (~300 ellipsoids) ...")
    for eh in make_ear_holes(bone_att):
        bone_map.add_component(eh)

    # three small high-contrast bone dots (calibration / marker objects)
    for x0, y0, r in [(-8.0, 0.0, 0.10), (-8.5, 0.5, 0.05), (-8.5, -0.5, 0.02)]:
        bone_map.add_component(ellipsoid(x0=x0, y0=y0, att=bone_att,
                                         ax=r, ay=r, az=r))

    # eye map 
    eye_map = tissue_map(physical_material=False)
    eye_map.add_component(ellipsoid(x0=-4.7, y0=4.3, z0=0.872,
                                    att=eye_att, ax=2., ay=2., az=2.))
    eye_map.add_component(ellipsoid(x0= 4.7, y0=4.3, z0=0.872,
                                    att=eye_att, ax=2., ay=2., az=2.))

    # ventricle map 
    ventricle_map = tissue_map(physical_material=False)
    ventricle_map.add_component(ellipsoid(y0=-3.6, att=ventricle_att,
                                          ax=1.8, ay=3.6, az=3.6))

    # FORBILD spot 1 (slightly hyper-dense) 
    spot1_map = tissue_map(physical_material=False)
    spot1_map.add_component(ellipsoid(x0=-1.08, y0=-9., att=spot1_att,
                                      ax=0.4, ay=0.4, az=0.4))

    # FORBILD spot 2 (slightly hypo-dense) 
    spot2_map = tissue_map(physical_material=False)
    spot2_map.add_component(ellipsoid(x0=1.08, y0=-9., att=spot2_att,
                                      ax=0.4, ay=0.4, az=0.4))

    # hematoma 
    hematoma_map = tissue_map(physical_material=False)
    hematoma_map.add_component(ellipsoid(
        x0=6.393945, y0=-6.393945, att=hematoma_att,
        ax=1.2, ay=0.42, az=1.4, beta=pi*58.1/180.))

    # brain map
    brain_map = tissue_map(physical_material=False)

    # brain fill
    brain_map.add_component(ellipsoid(att=brain_att,
                                      ax=ax_ie, ay=ay_ie, az=az_ie))

    # carve out ear cavity
    brain_map.add_component(gen_object(x0=9.1, y0=0., z0=0., att=-brain_att,
                                       nsurf=2, surfaces=[ear_ell, skull_ell2]))

    # carve out protuberance
    brain_map.add_component(gen_object(att=-brain_att, nsurf=2,
                                       surfaces=[pro_cone, skull_ell]))

    # carve out eyes
    brain_map.add_component(ellipsoid(x0=-4.7, y0=4.3, z0=0.872,
                                      att=-brain_att, ax=2., ay=2., az=2.))
    brain_map.add_component(ellipsoid(x0= 4.7, y0=4.3, z0=0.872,
                                      att=-brain_att, ax=2., ay=2., az=2.))

    # carve out sinuses
    brain_map.add_component(ellipsoid(x0=-1.9, y0=5.4, z0=0., att=-brain_att,
                                      ax=1.165, ay=0.406, az=3.,
                                      alpha=0., beta=-pi/4.))
    brain_map.add_component(ellipsoid(x0= 1.9, y0=5.4, z0=0., att=-brain_att,
                                      ax=1.165, ay=0.406, az=3.,
                                      alpha=0., beta= pi/4.))
    brain_map.add_component(right_elliptical_cylinder(
        x0=0., y0=3.6, z0=0., att=-brain_att,
        ax=4.0, ay=1.2, length=0.483,
        alpha=pi/2., beta=pi/3.))
    brain_map.add_component(right_elliptical_cylinder(
        x0=-4.3, y0=6.8, z0=-1., att=-brain_att,
        ax=1.8, ay=0.24, length=4.,
        alpha=0., beta=0., gamma=-pi/6.))
    brain_map.add_component(right_elliptical_cylinder(
        x0= 4.3, y0=6.8, z0=-1., att=-brain_att,
        ax=1.8, ay=0.24, length=4.,
        alpha=0., beta=0., gamma= pi/6.))

    # carve out frontal sinus cavity
    brain_map.add_component(ellipsoid(x0=0., y0=8.4, z0=0., att=-brain_att,
                                      ax=1.8, ay=3.0, az=3.0))

    # carve out ventricle, spots, hematoma
    brain_map.add_component(ellipsoid(y0=-3.6, att=-brain_att,
                                      ax=1.8, ay=3.6, az=3.6))
    brain_map.add_component(ellipsoid(x0=-1.08, y0=-9., att=-brain_att,
                                      ax=0.4, ay=0.4, az=0.4))
    brain_map.add_component(ellipsoid(x0= 1.08, y0=-9., att=-brain_att,
                                      ax=0.4, ay=0.4, az=0.4))
    brain_map.add_component(ellipsoid(x0=6.393945, y0=-6.393945, att=-brain_att,
                                      ax=1.2, ay=0.42, az=1.4,
                                      beta=pi*58.1/180.))

    # carve out bone marker dots
    for x0, y0, r in [(-8.0, 0.0, 0.10), (-8.5, 0.5, 0.05), (-8.5, -0.5, 0.02)]:
        brain_map.add_component(ellipsoid(x0=x0, y0=y0, att=-brain_att,
                                          ax=r, ay=r, az=r))

    # assemble 
    head = phantom3D()
    head.add_component(bone_map)
    head.add_component(eye_map)
    head.add_component(ventricle_map)
    head.add_component(spot1_map)
    head.add_component(spot2_map)
    head.add_component(hematoma_map)
    head.add_component(brain_map)

    return head

if __name__ == '__main__':
    print("Building head phantom ...")
    head = build_head_phantom()

    # 200x200x32 is a reasonable balance of speed vs. detail.
    # Match the professor's aspect ratio: 26cm FOV, thin slab.
    print("Embedding in 3D image array ...")
    image = image3D(shape=(200, 200, 32),
                    xlen=26., ylen=26., zlen=26./16.,
                    x0=-13., y0=-13., z0=-13./16.)
    head.embed_in(image)
    vol = image.mat

    print(f"  Volume shape : {vol.shape}")
    print(f"  Value range  : [{vol.min():.4f}, {vol.max():.4f}]")

    # projections
    proj_z = vol.sum(axis=2)           # axial DRR  (looking down)
    proj_x = vol.sum(axis=0)           # lateral
    proj_y = vol.sum(axis=1)           # AP
    mid_z  = vol[:, :, vol.shape[2]//2]  # central axial slice

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#0d0d0d')
    fig.suptitle("Head Phantom — Projections & Central Slice",
                 color='white', fontsize=14, y=1.01)

    panels = [
        (proj_z.T,  "Axial projection  (sum z)",   "x →", "← y"),
        (proj_x.T,  "Lateral projection (sum x)",   "y →", "z ↑"),
        (proj_y.T,  "AP projection  (sum y)",        "x →", "z ↑"),
        (mid_z.T,   "Central axial slice",            "x →", "← y"),
    ]

    for ax, (data, title, xlabel, ylabel) in zip(axes, panels):
        ax.set_facecolor('#0d0d0d')
        im = ax.imshow(data, cmap='magma', origin='lower', aspect='equal')
        ax.set_title(title, color='#e0e0e0', fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color='#888888', fontsize=8)
        ax.set_ylabel(ylabel, color='#888888', fontsize=8)
        ax.tick_params(colors='#555555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
            colors='#888888', labelsize=7)

    plt.tight_layout()
    out_path = 'head_phantom_projections.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"Plot saved to {out_path}")