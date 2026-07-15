"""
ehd_geom.py -- Analytic geometry primitives for the sessile-droplet EHD electrostatics study.

Every conductor exposes a vectorised implicit function f(x, y, z) that is negative
inside the conductor and positive outside, and is continuous across the surface.
The zero level set is the conductor surface.  The embedded-boundary (Shortley-Weller)
discretisation in ehd_solver.py locates surface crossings along grid lines by
bisection on f, which is exact to machine precision for the smooth primitives used
here and needs no assumption that f is a true signed distance.  This is what allows
the curved droplet interface and the rounded pin tip to be resolved to second order
without stair-stepping (Reviewer 4, comment 4).

Lengths in metres, potentials in volts.
"""

import numpy as np


# --------------------------------------------------------------------------- #
#  Base
# --------------------------------------------------------------------------- #
class Conductor:
    """A conductor held at a fixed potential, defined by an implicit function."""

    potential = 0.0

    def f(self, x, y, z):
        raise NotImplementedError

    def inside(self, x, y, z):
        return self.f(x, y, z) < 0.0


# --------------------------------------------------------------------------- #
#  Droplet interfaces
# --------------------------------------------------------------------------- #
class SphericalCapDroplet(Conductor):
    """
    Sessile droplet as a spherical cap resting on the plane z = 0.

    Rc          cap curvature radius
    theta_c     contact angle (radians); pi/2 gives the hemisphere of the
                original submission, for which Rc is simultaneously the curvature
                radius, the footprint radius and the apex height.

    The cap centre lies at z0 = -Rc*cos(theta_c) so the interface meets z = 0 at
    footprint radius a_foot = Rc*sin(theta_c) with the prescribed contact angle.
    """

    def __init__(self, Rc, theta_c=np.pi / 2, potential=0.0):
        self.Rc = float(Rc)
        self.theta_c = float(theta_c)
        self.z0 = -Rc * np.cos(theta_c)
        self.a_foot = Rc * np.sin(theta_c)
        self.potential = float(potential)

    def f(self, x, y, z):
        # exact signed distance to the sphere; the substrate plane z = 0 truncates it
        return np.sqrt(x ** 2 + y ** 2 + (z - self.z0) ** 2) - self.Rc

    @property
    def h_cap(self):
        return self.z0 + self.Rc          # apex height above the substrate

    def volume(self):
        h = self.h_cap
        return np.pi / 3.0 * h ** 2 * (3.0 * self.Rc - h)

    def surface_points(self, n=2001):
        """
        Arc-length parameterisation of the generatrix in the xz-plane.
        s = 0 at the +x contact line, increasing over the surface to the -x
        contact line.  Returns (s, x, z, nx, nz) with (nx, nz) the outward normal.
        """
        psi0 = np.arctan2(-self.z0, self.a_foot)       # polar angle of contact line
        psi = np.linspace(psi0, np.pi - psi0, n)
        x = self.Rc * np.cos(psi)
        z = self.z0 + self.Rc * np.sin(psi)
        s = self.Rc * (psi - psi0)
        return s, x, z, np.cos(psi), np.sin(psi)

    def normal_at(self, x, y, z):
        vx, vy, vz = x, y, z - self.z0
        n = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        return vx / n, vy / n, vz / n


class ProfileDroplet(Conductor):
    """
    Axisymmetric droplet whose generatrix is supplied numerically as (r_i, z_i)
    from the contact line to the apex -- used for the gravitationally equilibrated
    Young-Laplace shapes (Reviewer 4, comment 2).

    The implicit function is built from the exact distance to the revolved
    generatrix, signed by an inside test in the (r, z) half-plane.
    """

    def __init__(self, r_prof, z_prof, potential=0.0):
        r_prof = np.asarray(r_prof, float)
        z_prof = np.asarray(z_prof, float)
        if z_prof[0] > z_prof[-1]:                      # order: contact line -> apex
            r_prof, z_prof = r_prof[::-1], z_prof[::-1]
        self.r_prof, self.z_prof = r_prof, z_prof
        self.a_foot = float(r_prof[0])
        self.h_apex = float(z_prof[-1])
        self.potential = float(potential)
        # densified generatrix for distance queries
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(r_prof), np.diff(z_prof)))])
        sd = np.linspace(0.0, s[-1], 1501)
        self._rd = np.interp(sd, s, r_prof)
        self._zd = np.interp(sd, s, z_prof)
        self._s_tot = s[-1]
        self._s = s

    def f(self, x, y, z):
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        r = np.sqrt(x ** 2 + y ** 2)
        r, zz = np.broadcast_arrays(r, z)
        shp = r.shape
        rf = np.ascontiguousarray(r).ravel(); zf = np.ascontiguousarray(zz).ravel()
        d = np.full(rf.shape, 1e30)
        # only points within a margin of the droplet bounding box need the true distance
        margin = 1.0e-3
        near = (rf <= self.a_foot + margin) & (zf <= self.h_apex + margin) & (zf >= -margin)
        idx = np.where(near)[0]
        if idx.size:
            chunk = max(1, int(3e6 // max(self._rd.size, 1)))
            for i0 in range(0, idx.size, chunk):
                sl = idx[i0:i0 + chunk]
                dd = np.hypot(rf[sl, None] - self._rd[None, :], zf[sl, None] - self._zd[None, :])
                d[sl] = dd.min(axis=1)
        # NB: r_prof[::-1] increases from ~0 (apex) to a_foot (contact line). For query
        # radii below the smallest tabulated radius -- i.e. grid nodes lying exactly on
        # the symmetry axis -- the profile height is the apex height, NOT undefined.
        # Using left=NaN here misclassifies the whole axis line as outside the droplet,
        # which punches a spurious non-conductor filament up the axis and produces a
        # false field spike at the apex.
        z_of_r = np.interp(rf, self.r_prof[::-1], self.z_prof[::-1],
                           left=self.h_apex, right=-1.0)
        inside = (zf >= 0.0) & (rf <= self.a_foot) & (zf <= z_of_r)
        d = np.where(inside, -np.minimum(d, 1e29), d)
        return d.reshape(shp)

    def surface_points(self, n=2001):
        """
        FULL great circle of the revolved profile in the xz-plane, parameterised by
        arc length s from the +x contact line (s = 0), over the apex (s = s_tot), to
        the -x contact line (s = 2*s_tot).  This matches the convention used by
        SphericalCapDroplet so that the axisymmetric and three-dimensional solvers,
        and the hemisphere and Young-Laplace shapes, are all directly comparable.
        """
        half = max(n // 2, 2)
        sg = np.linspace(0.0, self._s_tot, half)          # generatrix: foot -> apex
        rg = np.interp(sg, self._s, self.r_prof)
        zg = np.interp(sg, self._s, self.z_prof)
        # +x branch (s: 0 -> s_tot), then mirrored -x branch (s: s_tot -> 2 s_tot)
        s = np.concatenate([sg, self._s_tot + (self._s_tot - sg[::-1][1:])])
        x = np.concatenate([rg, -rg[::-1][1:]])
        z = np.concatenate([zg, zg[::-1][1:]])
        # outward normal in the (x, z) plane from the local tangent
        dx = np.gradient(x, s)
        dz = np.gradient(z, s)
        nx, nz = dz, -dx
        nrm = np.hypot(nx, nz)
        nx, nz = nx / nrm, nz / nrm
        # enforce outward: normal must point away from the axis/interior
        flip = (nx * x + nz * (z - 0.5 * self.h_apex)) < 0
        nx = np.where(flip, -nx, nx)
        nz = np.where(flip, -nz, nz)
        return s, x, z, nx, nz


# --------------------------------------------------------------------------- #
#  Pin electrodes
# --------------------------------------------------------------------------- #
class RoundedPin(Conductor):
    """
    Vertical pin of shank radius a with axis at (xc, yc), spanning z in
    [z_tip, z_top], whose lower end is closed by a flat face of radius
    (a - r_tip) blended into the shank by a toroidal fillet of radius r_tip.

    r_tip = a gives a fully hemispherical tip; r_tip -> 0 recovers the sharp-edged
    flat pin of the original submission, whose tip field is a mesh-dependent
    singular quantity.  A finite, stated r_tip is what makes the tip field a
    converged number and permits a corona-inception criterion to be applied
    (Reviewer 4, comment 7).
    """

    def __init__(self, xc, yc, a, z_tip, z_top, r_tip, potential=0.0):
        self.xc, self.yc = float(xc), float(yc)
        self.a = float(a)
        self.z_tip, self.z_top = float(z_tip), float(z_top)
        self.r_tip = float(np.clip(r_tip, 0.0, a))
        self.potential = float(potential)

    def f(self, x, y, z):
        rho = np.sqrt((np.asarray(x) - self.xc) ** 2 + (np.asarray(y) - self.yc) ** 2)
        rho, z = np.broadcast_arrays(rho, np.asarray(z, float))
        rt = self.r_tip
        rc = self.a - rt                 # fillet centreline radius
        zc = self.z_tip + rt             # fillet centreline height

        # 2-D profile of the solid in the (rho, z) half-plane, as a rounded rectangle
        # over rho in [0, a], z in [z_tip, z_top], with only the lower outer corner
        # rounded.  Distance via the standard rounded-box construction restricted to
        # that corner.
        dz_top = z - self.z_top          # > 0 above the top face
        dz_bot = self.z_tip - z          # > 0 below the tip face
        drho = rho - self.a              # > 0 outside the shank

        # corner-rounded region: rho > rc and z < zc
        qx = np.maximum(rho - rc, 0.0)
        qz = np.maximum(zc - z, 0.0)
        d_corner = np.hypot(qx, qz) - rt

        # elsewhere: plain box distance
        d_box = np.maximum.reduce([drho, dz_bot, dz_top])

        in_corner = (rho > rc) & (z < zc)
        d = np.where(in_corner, np.maximum(d_corner, dz_top), d_box)
        return d


class FlatPin(Conductor):
    """Flat-ended, sharp-edged pin: the electrode of the original submission.
    Retained only for the explicit sharp-versus-rounded tip comparison."""

    def __init__(self, xc, yc, a, z_tip, z_top, potential=0.0):
        self.xc, self.yc = float(xc), float(yc)
        self.a = float(a)
        self.z_tip, self.z_top = float(z_tip), float(z_top)
        self.potential = float(potential)

    def f(self, x, y, z):
        rho = np.sqrt((np.asarray(x) - self.xc) ** 2 + (np.asarray(y) - self.yc) ** 2)
        rho, z = np.broadcast_arrays(rho, np.asarray(z, float))
        dr = rho - self.a
        dz = np.maximum(self.z_tip - z, z - self.z_top)
        outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
        inside = np.minimum(np.maximum(dr, dz), 0.0)
        return outside + inside


class Plate(Conductor):
    """Disc/slab top electrode of the parallel-plate cell."""

    def __init__(self, half_width, z_bot, z_top, potential=0.0):
        self.half_width = float(half_width)
        self.z_bot, self.z_top = float(z_bot), float(z_top)
        self.potential = float(potential)

    def f(self, x, y, z):
        rho = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
        rho, z = np.broadcast_arrays(rho, np.asarray(z, float))
        dr = rho - self.half_width
        dz = np.maximum(self.z_bot - z, z - self.z_top)
        outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
        inside = np.minimum(np.maximum(dr, dz), 0.0)
        return outside + inside


class Disc(Conductor):
    """
    Finite cylindrical disc electrode (the bottom electrode as actually specified in
    the manuscript: 20 mm diameter, 2 mm thick copper), occupying rho <= radius and
    z in [z_bot, z_top].

    Modelling the bottom electrode with its true finite extent, rather than as an
    infinite plane, is what makes the pin-plate configuration a well-posed free-space
    problem: with an infinite energised plane the total electrode charge, and hence
    the droplet-surface field, depends on where the outer boundary is placed.
    """

    def __init__(self, radius, z_bot, z_top, potential=0.0):
        self.radius = float(radius)
        self.z_bot, self.z_top = float(z_bot), float(z_top)
        self.potential = float(potential)

    def f(self, x, y, z):
        rho = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
        rho, z = np.broadcast_arrays(rho, np.asarray(z, float))
        dr = rho - self.radius
        dz = np.maximum(self.z_bot - z, z - self.z_top)
        outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
        inside = np.minimum(np.maximum(dr, dz), 0.0)
        return outside + inside


class GroundPlane(Conductor):
    """Bottom electrode: the half-space z <= 0 (its top face is the substrate)."""

    def __init__(self, z_surf=0.0, potential=0.0):
        self.z_surf = float(z_surf)
        self.potential = float(potential)

    def f(self, x, y, z):
        z = np.asarray(z, float)
        return z - self.z_surf


# --------------------------------------------------------------------------- #
#  Crossing location by bisection (exact for smooth f, no SDF assumption)
# --------------------------------------------------------------------------- #
def crossing_fraction(fun, p0, p1, iters=50):
    """
    For grid points p0 (outside, f>0) and p1 (inside, f<0), return theta in (0, 1]
    such that the conductor surface lies at p0 + theta*(p1 - p0).

    p0, p1 are (3, N) arrays.  Bisection is used rather than linear interpolation of
    f so that the result is independent of any signed-distance assumption; 50
    iterations converge to machine precision.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    lo = np.zeros(p0.shape[1])
    hi = np.ones(p0.shape[1])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pm = p0 + mid * (p1 - p0)
        fm = fun(pm[0], pm[1], pm[2])
        neg = fm < 0.0
        hi = np.where(neg, mid, hi)
        lo = np.where(neg, lo, mid)
    theta = 0.5 * (lo + hi)
    return np.clip(theta, 1e-8, 1.0)
