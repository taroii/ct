"""Fan-beam projector for the 2D experiments, as a reusable factory.

The projector kernels below are lifted VERBATIM from the ``@njit`` closures
inside ``reconstruction.run_reconstruction_for_mfact``. They are duplicated
rather than refactored out of that function on purpose: every number in the
manuscript comes out of ``reconstruction.py``, so the published solver is left
byte-for-byte untouched and nothing here can move a reported result.

The duplication is only safe because it is checked. ``verify_against_reference``
projects and back-projects a random image through both this factory and the
reference solver and asserts bit-identical output; ``python src/geometry2d.py``
runs that check. If the two ever drift, that check fails loudly instead of
quietly giving a side experiment a different forward model from the paper's.

Once (and only once) an experiment built on this module earns a place in the
manuscript, the right move is to delete the copy here and have
``reconstruction.py`` import the factory instead -- a refactor that must be
accompanied by a full re-run, not done in passing.
"""
import numpy as np
from numba import njit
from numpy import arange, arcsin, cos, floor, newaxis, ones, pi, sin, sqrt, tan


class FanBeam:
    """Geometry plus its forward/adjoint pair.

    Attributes mirror the names used in ``reconstruction.py`` so that code can
    be moved between the two without a translation step.
    """

    def __init__(self, nx, ny, nviews, nbins, nrays, mask, ximageside,
                 yimageside, project, backproject):
        self.nx, self.ny = nx, ny
        self.nviews, self.nbins, self.nrays = nviews, nbins, nrays
        self.mask = mask
        self.ximageside, self.yimageside = ximageside, yimageside
        self.project, self.backproject = project, backproject

    def A(self, image):
        """Forward projection, allocating its own sinogram."""
        sino = np.zeros([self.nviews, self.nbins])
        self.project(image, sino)
        return sino

    def AT(self, sino):
        """Back projection, masked to the reconstruction circle."""
        im = np.zeros([self.nx, self.ny])
        self.backproject(sino, im)
        im *= self.mask
        return im


def make_fanbeam(nx, ny, ximageside=10.0, yimageside=10.0, radius=50.0,
                 source_to_detector=100.0, arc_degrees=50.0, nviews=25,
                 nbins=1024, larc=1.0):
    """Build the 2D limited-angle fan-beam geometry of the paper.

    Defaults reproduce the manuscript geometry exactly: a 50-degree arc of 25
    views onto 1024 detector bins, with a 10x10 field of view.
    """
    dx = ximageside/nx
    dy = yimageside/ny
    xar = arange(-ximageside/2. + dx/2, ximageside/2., dx)[:, newaxis]*ones([ny])
    yar = ones([nx, ny])*arange(-yimageside/2. + dy/2, yimageside/2., dy)
    rar = sqrt(xar**2 + yar**2)
    mask = np.zeros([nx, ny])
    mask[rar <= ximageside/2.] = 1.

    srad = radius
    sd = source_to_detector
    slen = (arc_degrees/180.)*pi
    slen0 = -slen/2.0
    nrays = nbins*nviews

    fanangle2 = arcsin((ximageside/2.)/radius)
    detectorlength = 2.*tan(fanangle2)*source_to_detector
    larc_local = larc

    @njit
    def circularFanbeamProjection(image, sinogram):
        dx = ximageside/nx
        dy = yimageside/ny
        x0 = -ximageside/2.
        y0 = -yimageside/2.
        u0 = -detectorlength/2.
        du = detectorlength/nbins
        ds = slen/(nviews-larc_local)

        for sindex in range(nviews):
            s = sindex*ds + slen0
            xsource = radius*cos(s)
            ysource = radius*sin(s)
            xDetCenter = (radius - source_to_detector)*cos(s)
            yDetCenter = (radius - source_to_detector)*sin(s)
            eux = -sin(s)
            euy = cos(s)

            for uindex in range(nbins):
                u = u0 + (uindex + 0.5)*du
                xbin = xDetCenter + eux*u
                ybin = yDetCenter + euy*u
                xl = x0
                yl = y0
                xdiff = xbin - xsource
                ydiff = ybin - ysource
                xad = abs(xdiff)*dy
                yad = abs(ydiff)*dx

                if (xad > yad):
                    slope = ydiff/xdiff
                    travPixlen = dx*sqrt(1.0 + slope*slope)
                    yIntOld = ysource + slope*(xl - xsource)
                    iyOld = int(floor((yIntOld - y0)/dy))
                    raysum = 0.
                    for ix in range(nx):
                        x = xl + dx*(ix + 1.0)
                        yIntercept = ysource + slope*(x - xsource)
                        iy = int(floor((yIntercept - y0)/dy))
                        if iy == iyOld:
                            if ((iy >= 0) and (iy < ny)):
                                raysum = raysum + travPixlen*image[ix, iy]
                        else:
                            yMid = dy*(iy if iy > iyOld else iyOld) + yl
                            ydist1 = abs(yMid - yIntOld)
                            ydist2 = abs(yIntercept - yMid)
                            frac1 = ydist1/(ydist1 + ydist2)
                            frac2 = 1.0 - frac1
                            if ((iyOld >= 0) and (iyOld < ny)):
                                raysum = raysum + frac1*travPixlen*image[ix, iyOld]
                            if ((iy >= 0) and (iy < ny)):
                                raysum = raysum + frac2*travPixlen*image[ix, iy]
                        iyOld = iy
                        yIntOld = yIntercept
                else:
                    slopeinv = xdiff/ydiff
                    travPixlen = dy*sqrt(1.0 + slopeinv*slopeinv)
                    xIntOld = xsource + slopeinv*(yl - ysource)
                    ixOld = int(floor((xIntOld - x0)/dx))
                    raysum = 0.
                    for iy in range(ny):
                        y = yl + dy*(iy + 1.0)
                        xIntercept = xsource + slopeinv*(y - ysource)
                        ix = int(floor((xIntercept - x0)/dx))
                        if (ix == ixOld):
                            if ((ix >= 0) and (ix < nx)):
                                raysum = raysum + travPixlen*image[ix, iy]
                        else:
                            xMid = dx*(ix if ix > ixOld else ixOld) + xl
                            xdist1 = abs(xMid - xIntOld)
                            xdist2 = abs(xIntercept - xMid)
                            frac1 = xdist1/(xdist1 + xdist2)
                            frac2 = 1.0 - frac1
                            if ((ixOld >= 0) and (ixOld < nx)):
                                raysum = raysum + frac1*travPixlen*image[ixOld, iy]
                            if ((ix >= 0) and (ix < nx)):
                                raysum = raysum + frac2*travPixlen*image[ix, iy]
                        ixOld = ix
                        xIntOld = xIntercept
                sinogram[sindex, uindex] = raysum


    @njit
    def circularFanbeamBackProjection(sinogram, image):
        image.fill(0.)
        dx = ximageside/nx
        dy = yimageside/ny
        x0 = -ximageside/2.
        y0 = -yimageside/2.
        u0 = -detectorlength/2.
        du = detectorlength/nbins
        ds = slen/(nviews - larc_local)

        for sindex in range(nviews):
            s = sindex*ds + slen0
            xsource = radius*cos(s)
            ysource = radius*sin(s)
            xDetCenter = (radius - source_to_detector)*cos(s)
            yDetCenter = (radius - source_to_detector)*sin(s)
            eux = -sin(s)
            euy = cos(s)

            for uindex in range(nbins):
                sinoval = sinogram[sindex, uindex]
                u = u0 + (uindex + 0.5)*du
                xbin = xDetCenter + eux*u
                ybin = yDetCenter + euy*u
                xl = x0
                yl = y0
                xdiff = xbin - xsource
                ydiff = ybin - ysource
                xad = abs(xdiff)*dy
                yad = abs(ydiff)*dx

                if (xad > yad):
                    slope = ydiff/xdiff
                    travPixlen = dx*sqrt(1.0 + slope*slope)
                    yIntOld = ysource + slope*(xl - xsource)
                    iyOld = int(floor((yIntOld - y0)/dy))
                    for ix in range(nx):
                        x = xl + dx*(ix + 1.0)
                        yIntercept = ysource + slope*(x - xsource)
                        iy = int(floor((yIntercept - y0)/dy))
                        if iy == iyOld:
                            if ((iy >= 0) and (iy < ny)):
                                image[ix, iy] = image[ix, iy] + sinoval*travPixlen
                        else:
                            yMid = dy*(iy if iy > iyOld else iyOld) + yl
                            ydist1 = abs(yMid - yIntOld)
                            ydist2 = abs(yIntercept - yMid)
                            frac1 = ydist1/(ydist1 + ydist2)
                            frac2 = 1.0 - frac1
                            if ((iyOld >= 0) and (iyOld < ny)):
                                image[ix, iyOld] = image[ix, iyOld] + frac1*sinoval*travPixlen
                            if ((iy >= 0) and (iy < ny)):
                                image[ix, iy] = image[ix, iy] + frac2*sinoval*travPixlen
                        iyOld = iy
                        yIntOld = yIntercept
                else:
                    slopeinv = xdiff/ydiff
                    travPixlen = dy*sqrt(1.0 + slopeinv*slopeinv)
                    xIntOld = xsource + slopeinv*(yl - ysource)
                    ixOld = int(floor((xIntOld - x0)/dx))
                    for iy in range(ny):
                        y = yl + dy*(iy + 1.0)
                        xIntercept = xsource + slopeinv*(y - ysource)
                        ix = int(floor((xIntercept - x0)/dx))
                        if (ix == ixOld):
                            if ((ix >= 0) and (ix < nx)):
                                image[ix, iy] = image[ix, iy] + sinoval*travPixlen
                        else:
                            xMid = dx*(ix if ix > ixOld else ixOld) + xl
                            xdist1 = abs(xMid - xIntOld)
                            xdist2 = abs(xIntercept - xMid)
                            frac1 = xdist1/(xdist1 + xdist2)
                            frac2 = 1.0 - frac1
                            if ((ixOld >= 0) and (ixOld < nx)):
                                image[ixOld, iy] = image[ixOld, iy] + frac1*sinoval*travPixlen
                            if ((ix >= 0) and (ix < nx)):
                                image[ix, iy] = image[ix, iy] + frac2*sinoval*travPixlen
                        ixOld = ix
                        xIntOld = xIntercept

    return FanBeam(nx, ny, nviews, nbins, nrays, mask, ximageside, yimageside,
                   circularFanbeamProjection, circularFanbeamBackProjection)


def verify_against_reference(mfact=2, seed=0):
    """Assert this factory's operators match ``reconstruction.py``'s exactly.

    Returns the two max-absolute differences (forward, adjoint). Both must be
    exactly zero: the kernels are the same code on the same geometry, so any
    nonzero difference means the copy has drifted and must be re-synced.
    """
    import reconstruction as ref

    nx = ny = int(512/mfact)
    geo = make_fanbeam(nx, ny)

    # Rebuild the reference closures by replaying the geometry block that
    # precedes them. Importing them directly is not possible: they are nested
    # inside the 900-line run function and close over its locals.
    import inspect, textwrap as _tw
    srclines = inspect.getsource(ref.run_reconstruction_for_mfact).splitlines()
    start = next(i for i, l in enumerate(srclines) if "@njit" in l)
    end = next(i for i, l in enumerate(srclines) if "GRADIENT OPERATORS" in l)
    ns = {"njit": njit, "cos": cos, "sin": sin, "sqrt": sqrt, "floor": floor,
          "nx": nx, "ny": ny, "nviews": geo.nviews, "nbins": geo.nbins,
          "ximageside": geo.ximageside, "yimageside": geo.yimageside,
          "radius": 50.0, "source_to_detector": 100.0,
          "slen": (50./180.)*pi, "slen0": -(50./180.)*pi/2.0,
          "detectorlength": 2.*tan(arcsin((geo.ximageside/2.)/50.0))*100.0,
          "larc_local": ref.larc}
    exec(_tw.dedent("\n".join(srclines[start:end])), ns)

    rng = np.random.default_rng(seed)
    im = rng.standard_normal((nx, ny))*geo.mask

    s_ours = geo.A(im)
    s_ref = np.zeros([geo.nviews, geo.nbins])
    ns["circularFanbeamProjection"](im, s_ref)
    dfwd = float(np.abs(s_ours - s_ref).max())

    b_ours = np.zeros([nx, ny])
    geo.backproject(s_ref, b_ours)
    b_ref = np.zeros([nx, ny])
    ns["circularFanbeamBackProjection"](s_ref, b_ref)
    dadj = float(np.abs(b_ours - b_ref).max())

    assert dfwd == 0.0, f"forward projector has drifted: max|diff|={dfwd:g}"
    assert dadj == 0.0, f"adjoint projector has drifted: max|diff|={dadj:g}"
    return dfwd, dadj


if __name__ == "__main__":
    for m in (2, 4):
        d = verify_against_reference(mfact=m)
        print(f"mfact={m}: forward and adjoint match reconstruction.py exactly "
              f"(max|diff| = {d[0]:g}, {d[1]:g})")
