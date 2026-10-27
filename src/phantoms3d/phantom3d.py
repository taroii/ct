import numpy as np

pi    = np.pi
sin   = np.sin
cos   = np.cos
sqrt  = np.sqrt

class surface:
    """Base class: an oriented surface used to bound a gen_object volume."""

    def __init__(self, xc=0., yc=0., zc=0., alpha=0., beta=0., gamma=0.):
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        # Build the 3×3 rotation frame (ZYX Euler: gamma -> beta -> alpha)
        for vec_in, attr in [((1,0,0), 'xu'), ((0,1,0), 'yu'), ((0,0,1), 'zu')]:
            x, y, z = vec_in

            # gamma rotation (around Z)
            xp =  x*cos(gamma) - y*sin(gamma)
            yp =  x*sin(gamma) + y*cos(gamma)
            zp =  z

            # beta rotation (around Y)
            xpp =  xp*cos(beta) + zp*sin(beta)
            ypp =  yp
            zpp = -xp*sin(beta) + zp*cos(beta)

            # alpha rotation (around Z again)
            xppp =  xpp*cos(alpha) - ypp*sin(alpha)
            yppp =  ypp*cos(alpha) + xpp*sin(alpha)
            zppp =  zpp

            setattr(self, attr+'1', xppp)
            setattr(self, attr+'2', yppp)
            setattr(self, attr+'3', zppp)

    def __str__(self):
        return (f"surface: {self.label}\n"
                f"  at ({self.xc:.2f},{self.yc:.2f},{self.zc:.2f}), "
                f"normal ({self.zu1:.2f},{self.zu2:.2f},{self.zu3:.2f}), "
                f"x-axis ({self.xu1:.2f},{self.xu2:.2f},{self.xu3:.2f})")


class planar(surface):
    def __init__(self, xc=0., yc=0., zc=0., alpha=0., beta=0., gamma=0.):
        self.label = 'plane'
        super().__init__(xc, yc, zc, alpha, beta, gamma)


class ellipsoidal(surface):
    def __init__(self, xc=0., yc=0., zc=0., alpha=0., beta=0., gamma=0.,
                 ax=1., ay=1., az=1.):
        self.label = 'ellipsoid'
        self.ax = ax
        self.ay = ay
        self.az = az
        super().__init__(xc, yc, zc, alpha, beta, gamma)


class cylindrical(surface):
    def __init__(self, xc=0., yc=0., zc=0., alpha=0., beta=0., gamma=0.,
                 ax=1., ay=1.):
        self.label = 'cylinder'
        self.ax = ax
        self.ay = ay
        super().__init__(xc, yc, zc, alpha, beta, gamma)


class conical(surface):
    def __init__(self, xc=0., yc=0., zc=0., alpha=0., beta=0., gamma=0.,
                 ax=1., ay=1.):
        self.label = 'cone'
        self.ax = ax
        self.ay = ay
        super().__init__(xc, yc, zc, alpha, beta, gamma)

class image3D:
    """3D voxel array with physical extent metadata."""

    def __init__(self, shape=(128, 128, 128),
                 xlen=2.0, ylen=2.0, zlen=2.0,
                 x0=-1.0, y0=-1.0, z0=-1.0):
        self.mat = np.zeros(shape, dtype=np.float32)
        self.nx, self.ny, self.nz = shape
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.xlen, self.ylen, self.zlen = xlen, ylen, zlen
        self.dx = xlen / self.nx
        self.dy = ylen / self.ny
        self.dz = zlen / self.nz

    def __str__(self):
        return (f"nx={self.nx} ny={self.ny} nz={self.nz}\n"
                f"dx={self.dx:.4f} dy={self.dy:.4f} dz={self.dz:.4f}\n"
                f"x0={self.x0} y0={self.y0} z0={self.z0}\n"
                f"xlen={self.xlen} ylen={self.ylen} zlen={self.zlen}")

    def clear(self):
        self.mat[:] = 0.

    def _grids(self):
        """Return (X,Y,Z) voxel-center coordinate arrays, shape = image shape."""
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        z = self.z0 + (np.arange(self.nz) + 0.5) * self.dz
        return np.meshgrid(x, y, z, indexing='ij')   # shape (nx,ny,nz) each

class shape3D:
    def __init__(self, x0=0., y0=0., z0=0., att=1.0):
        self.x0  = x0
        self.y0  = y0
        self.z0  = z0
        self.att = att

    def __str__(self):
        return (f"attenuation={self.att}\n"
                f"center=({self.x0},{self.y0},{self.z0})\n")

    def embed_in(self, image):
        pass

class ellipsoid(shape3D):
    """Axis-aligned (with optional rotation) solid ellipsoid."""

    def __init__(self, x0=0., y0=0., z0=0., att=1.0,
                 ax=1., ay=1., az=1., alpha=0., beta=0.):
        self.ax    = ax
        self.ay    = ay
        self.az    = az
        self.alpha = alpha
        self.beta  = beta
        super().__init__(x0, y0, z0, att)

    # Pure-Python point test kept for reference / scalar use
    def voxval(self, x, y, z):
        ca, sa = cos(self.alpha), sin(self.alpha)
        cb, sb = cos(self.beta),  sin(self.beta)
        rx, ry, rz = x - self.x0, y - self.y0, z - self.z0
        r1x =  cb*rx + sb*ry
        r1y = -sb*rx + cb*ry
        r1z =  rz
        r2x =  ca*r1x + sa*r1z
        r2y =  r1y
        r2z = -sa*r1x + ca*r1z
        if (r2x/self.ax)**2 + (r2y/self.ay)**2 + (r2z/self.az)**2 <= 1.0:
            return self.att
        return 0.

    def embed_in(self, image: image3D):
        X, Y, Z = image._grids()
        ca, sa = cos(self.alpha), sin(self.alpha)
        cb, sb = cos(self.beta),  sin(self.beta)
        rx = X - self.x0
        ry = Y - self.y0
        rz = Z - self.z0
        r1x =  cb*rx + sb*ry
        r1y = -sb*rx + cb*ry
        r1z =  rz
        r2x =  ca*r1x + sa*r1z
        r2y =  r1y
        r2z = -sa*r1x + ca*r1z
        mask = (r2x/self.ax)**2 + (r2y/self.ay)**2 + (r2z/self.az)**2 <= 1.0
        image.mat[mask] += self.att

class gen_object(shape3D):
    """
    Analytic shape defined by the intersection of a list of surface half-spaces.

    A voxel is inside the object iff it satisfies the inside condition for
    *every* surface in self.surfaces.
    """

    def __init__(self, x0=0., y0=0., z0=0., att=1.0, nsurf=0, surfaces=None):
        self.nsurf    = nsurf
        self.surfaces = list(surfaces) if surfaces else []
        super().__init__(x0, y0, z0, att)

    def add_surface(self, surf=None):
        if surf is None:
            surf = planar()
        self.nsurf += 1
        self.surfaces.append(surf)

    @staticmethod
    def _rot(x, y, z, alpha, beta, gamma):
        xp  =  x*cos(gamma) - y*sin(gamma)
        yp  =  x*sin(gamma) + y*cos(gamma)
        zp  =  z
        xpp =  xp*cos(beta)  + zp*sin(beta)
        ypp =  yp
        zpp = -xp*sin(beta)  + zp*cos(beta)
        xppp =  xpp*cos(alpha) - ypp*sin(alpha)
        yppp =  ypp*cos(alpha) + xpp*sin(alpha)
        zppp =  zpp
        return xppp, yppp, zppp

    def rotate(self, alpha=0., beta=0., gamma=0.):
        for s in self.surfaces:
            # rotate surface centre about object origin
            s.xc, s.yc, s.zc = self._rot(s.xc, s.yc, s.zc, alpha, beta, gamma)
            # rotate each frame axis
            for attr in ('xu', 'yu', 'zu'):
                v = (getattr(s, attr+'1'),
                     getattr(s, attr+'2'),
                     getattr(s, attr+'3'))
                rv = self._rot(*v, alpha, beta, gamma)
                setattr(s, attr+'1', rv[0])
                setattr(s, attr+'2', rv[1])
                setattr(s, attr+'3', rv[2])

    @staticmethod
    def _inside_surface(s, ox, oy, oz, X, Y, Z):
        """
        Return boolean mask: True where (X,Y,Z) is on the interior side of
        surface s (whose origin is offset by object centre ox,oy,oz).
        """
        # Position relative to surface centre (in world coords)
        rx = X - (s.xc + ox)
        ry = Y - (s.yc + oy)
        rz = Z - (s.zc + oz)

        # Project onto surface local frame
        lx = s.xu1*rx + s.xu2*ry + s.xu3*rz
        ly = s.yu1*rx + s.yu2*ry + s.yu3*rz
        lz = s.zu1*rx + s.zu2*ry + s.zu3*rz

        if s.label == 'plane':
            # Inside = positive side of the plane normal (lz >= 0).
            # The plane normal (zu) points inward; a point is inside when its
            # projection onto the normal is non-negative (i.e. on the same side
            # as the normal direction).
            return lz >= 0.

        elif s.label == 'cylinder':
            # Inside = within the elliptic cross-section (lx,ly plane)
            return (lx/s.ax)**2 + (ly/s.ay)**2 <= 1.

        elif s.label == 'ellipsoid':
            return (lx/s.ax)**2 + (ly/s.ay)**2 + (lz/s.az)**2 <= 1.

        elif s.label == 'cone':
            # Cone opens in +lz direction; inside = within elliptic cross at lz
            if np.isscalar(lz):
                if lz <= 0:
                    return False
                return (lx/(s.ax*lz))**2 + (ly/(s.ay*lz))**2 <= 1.
            else:
                mask = lz > 0
                mask[mask] &= ((lx[mask]/(s.ax*lz[mask]))**2 +
                               (ly[mask]/(s.ay*lz[mask]))**2 <= 1.)
                return mask

        else:
            raise ValueError(f"Unknown surface label: {s.label}")

    def embed_in(self, image: image3D):
        X, Y, Z = image._grids()

        # Start with all-True mask; intersect over every surface
        mask = np.ones(image.mat.shape, dtype=bool)
        for s in self.surfaces:
            mask &= self._inside_surface(s, self.x0, self.y0, self.z0, X, Y, Z)

        image.mat[mask] += self.att

class phantom3D:
    """Ordered collection of shapes / tissue_maps."""

    def __init__(self):
        self.num_components = 0
        self.components = []

    def __str__(self):
        lines = []
        for i, c in enumerate(self.components, 1):
            lines.append(f"component {i}: {c}")
        return "\n".join(lines)

    def __add__(self, other):
        new = phantom3D()
        for c in self.components:
            new.add_component(c)
        for c in other.components:
            new.add_component(c)
        return new

    def add_component(self, component):
        self.num_components += 1
        self.components.append(component)

    def embed_in(self, image: image3D):
        for c in self.components:
            c.embed_in(image)


class tissue_map(phantom3D):
    """A phantom3D that groups shapes belonging to a single tissue type."""

    def __init__(self, material='polymethyl_methacrylate', physical_material=False):
        # physical_material is a no-op placeholder for future spectral CT support.
        #
        # When implemented, each tissue_map would carry:
        #   - a mass attenuation coefficient curve (\mu/\rho vs. keV), loaded from a
        #     material data file and interpolated log-linearly
        #   - a bulk density (g/cm^3)
        #
        # This would enable polychromatic forward projection: instead of integrating
        # a single scalar attenuation along each ray, you integrate density × \mu/\rho(E)
        # per energy bin and sum weighted by the source spectrum. Unlocks beam
        # hardening simulation, dual-energy / spectral CT forward models, and
        # material decomposition ground truth.
        #
        # The original tomo3D.py has the skeleton: tissue_map.mass_attenuation_coef()
        # and tissue_map.get_density(), with material tables stored as two-column
        # log(E) vs. log(\mu/\rho) .dat files. The phantom builder API would not need
        # to change — only the projector needs to become energy-aware.
        super().__init__()