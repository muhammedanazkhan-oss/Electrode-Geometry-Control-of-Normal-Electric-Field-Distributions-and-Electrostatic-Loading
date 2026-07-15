"""
ehd_solver3d.py -- Three-dimensional Cartesian finite-difference Laplace solver with
a Shortley-Weller embedded-boundary treatment.

Solves  lap(phi) = 0  on a box, with Dirichlet conditions on every conductor and a
choice of homogeneous Neumann (open), mirror-symmetry, or antisymmetry conditions on
each outer face.

Relative to the original submission this solver differs in three ways that respond
directly to Reviewer 4, comment 4:

  * curved conductor surfaces (the droplet, and the rounded pin tip) are resolved by
    the Shortley-Weller non-uniform stencil rather than by a stair-stepped mask, which
    restores second-order accuracy and hence a meaningful Richardson/GCI estimate;
  * the pin tip carries a finite, stated edge radius, so the tip field is a converged
    physical quantity rather than a mesh-dependent singular value;
  * surface fields are extracted by probing along the exact outward normal at fixed
    physical offsets, so the extracted field converges with the interior solution.

The linear system is solved by conjugate gradients preconditioned with smoothed-
aggregation algebraic multigrid (pyamg).
"""

import numpy as np
from scipy.sparse import coo_matrix
import pyamg

from ehd_geom import crossing_fraction

EPS_INFLATE = 1e-9


class Solver3D:
    def __init__(self, xlim, ylim, zlim, h, conductors,
                 ybc='neumann', outer_bc=None, robin=False):
        """
        xlim, ylim, zlim : (lo, hi) in metres
        h                : uniform spacing
        ybc              : condition on the y = ylim[0] face
                           'neumann'  -> mirror symmetry  (dphi/dy = 0)
                           'dirichlet0' -> antisymmetry   (phi = 0)
        outer_bc         : optional callable(x, y, z) -> phi prescribed on the
                           remaining outer faces (used for code verification)
        """
        self.h = float(h)
        self.x = np.arange(xlim[0], xlim[1] + 0.5 * h, h)
        self.y = np.arange(ylim[0], ylim[1] + 0.5 * h, h)
        self.z = np.arange(zlim[0], zlim[1] + 0.5 * h, h)
        self.nx, self.ny, self.nz = self.x.size, self.y.size, self.z.size
        self.conductors = conductors
        self.ybc = ybc
        self.outer_bc = outer_bc
        self.robin = bool(robin)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self._build_masks()
        self.phi = None

    @staticmethod
    def _fc(c):
        return lambda x, y, z: c.f(x, y, z) - EPS_INFLATE

    # --------------------------------------------------------------------- #
    def _build_masks(self):
        self.dirichlet = np.zeros((self.nx, self.ny, self.nz), bool)
        self.dval = np.zeros((self.nx, self.ny, self.nz))
        for c in self.conductors:
            m = self._fc(c)(self.X, self.Y, self.Z) < 0.0
            self.dirichlet |= m
            self.dval[m] = c.potential
        if self.ybc == 'dirichlet0':
            # antisymmetry plane: phi = 0 on y = ylim[0]
            self.dirichlet[:, 0, :] = True
            self.dval[:, 0, :] = 0.0
        if self.outer_bc is not None:
            ob = np.zeros_like(self.dirichlet)
            ob[0, :, :] = ob[-1, :, :] = True
            ob[:, :, 0] = ob[:, :, -1] = True
            ob[:, -1, :] = True
            ob &= ~self.dirichlet
            self.dval[ob] = self.outer_bc(self.X[ob], self.Y[ob], self.Z[ob])
            self.dirichlet |= ob
        self.unknown = ~self.dirichlet

    # --------------------------------------------------------------------- #
    def _cut_info(self, I, J, K, d, axis):
        h = self.h
        p0 = np.vstack([self.x[I], self.y[J], self.z[K]])
        p1 = p0.copy()
        p1[axis] = p1[axis] + d * h
        nb = [I, J, K]
        nb[axis] = nb[axis] + d
        theta = np.ones(I.size)
        gval = self.dval[nb[0], nb[1], nb[2]].copy()
        for c in self.conductors:
            fc = self._fc(c)
            f1 = fc(p1[0], p1[1], p1[2])
            f0 = fc(p0[0], p0[1], p0[2])
            sel = (f1 < 0.0) & (f0 >= 0.0)
            if not np.any(sel):
                continue
            th = crossing_fraction(fc, p0[:, sel], p1[:, sel])
            idx = np.where(sel)[0]
            upd = th < theta[idx]
            theta[idx[upd]] = th[upd]
            gval[idx[upd]] = c.potential
        return theta, gval

    # --------------------------------------------------------------------- #
    def assemble(self):
        h = self.h
        n = (self.nx, self.ny, self.nz)
        unk = np.argwhere(self.unknown)
        idx = -np.ones(n, np.int64)
        idx[self.unknown] = np.arange(unk.shape[0])
        N = unk.shape[0]
        I, J, K = unk[:, 0], unk[:, 1], unk[:, 2]
        me = idx[I, J, K]

        rows, cols, vals = [], [], []
        rhs = np.zeros(N)
        diag = np.zeros(N)

        def add(r_, c_, v_):
            rows.append(np.ascontiguousarray(r_, np.int64))
            cols.append(np.ascontiguousarray(c_, np.int64))
            vals.append(np.ascontiguousarray(v_, np.float64))

        ijk = [I, J, K]
        for axis in (0, 1, 2):
            for d in (+1, -1):
                nb = [I, J, K]
                nb = [a.copy() for a in nb]
                nb[axis] = nb[axis] + d
                inside = (nb[axis] >= 0) & (nb[axis] <= n[axis] - 1)
                # asymptotic (Robin) monopole far-field on the true outer faces;
                # the y = ylim[0] mirror plane keeps its zero-flux (dropped-link) form
                if self.robin:
                    ext = ~inside
                    if axis == 1 and d == -1:      # y = ylim[0] : mirror symmetry, keep Neumann
                        ext = np.zeros_like(ext)
                    se = np.where(ext)[0]
                    if se.size:
                        xg = self.X[I[se], J[se], K[se]].copy()
                        yg = self.Y[I[se], J[se], K[se]].copy()
                        zg = self.Z[I[se], J[se], K[se]].copy()
                        coord = [xg, yg, zg]
                        rho = np.sqrt(xg**2 + yg**2 + zg**2)
                        coord[axis] = coord[axis] + d * h
                        rho_g = np.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2)
                        diag[se] += (1.0 / h**2) * (rho / np.maximum(rho_g, 1e-12) - 1.0)
                sel = np.where(inside)[0]
                if sel.size == 0:
                    continue
                nbs = [a[sel] for a in nb]
                nb_dir = self.dirichlet[nbs[0], nbs[1], nbs[2]]
                w = 1.0 / h ** 2
                a_ = sel[~nb_dir]
                if a_.size:
                    na = [x[a_] for x in nb]
                    add(me[a_], idx[na[0], na[1], na[2]], np.full(a_.size, w))
                    diag[a_] -= w
                b_ = sel[nb_dir]
                if b_.size:
                    th, g = self._cut_info(I[b_], J[b_], K[b_], d, axis)
                    wb = w / th
                    rhs[b_] -= wb * g
                    diag[b_] -= wb

        add(me, me, diag)
        A = coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(N, N)).tocsr()
        self.A, self.rhs, self.idx, self.N = A, rhs, idx, N
        return A, rhs

    # --------------------------------------------------------------------- #
    def solve(self, tol=1e-10, verbose=False):
        A, b = self.assemble()
        # A is the negative-definite Laplacian; flip sign for an SPD system
        ml = pyamg.smoothed_aggregation_solver(-A.tocsr(), max_coarse=500)
        res = []
        x = ml.solve(-b, tol=tol, accel='cg', residuals=res, maxiter=300)
        self.iters = len(res)
        self.relres = res[-1] / max(res[0], 1e-300)
        if verbose:
            print(f"    AMG-CG: {self.iters} iters, rel. residual {self.relres:.2e}")
        phi = self.dval.copy()
        phi[self.unknown] = x
        self.phi = phi
        return phi

    # --------------------------------------------------------------------- #
    def interp(self, xq, yq, zq):
        """Trilinear interpolation of phi, with mirror folding across y = ylim[0]
        when that face is a symmetry plane."""
        h = self.h
        xq = np.asarray(xq, float)
        yq = np.asarray(yq, float)
        zq = np.asarray(zq, float)
        if self.ybc == 'neumann':
            yq = self.y[0] + np.abs(yq - self.y[0])
        xq = np.clip(xq, self.x[0], self.x[-1] - 1e-12)
        yq = np.clip(yq, self.y[0], self.y[-1] - 1e-12)
        zq = np.clip(zq, self.z[0], self.z[-1] - 1e-12)
        i = np.clip(((xq - self.x[0]) / h).astype(int), 0, self.nx - 2)
        j = np.clip(((yq - self.y[0]) / h).astype(int), 0, self.ny - 2)
        k = np.clip(((zq - self.z[0]) / h).astype(int), 0, self.nz - 2)
        tx = (xq - self.x[i]) / h
        ty = (yq - self.y[j]) / h
        tz = (zq - self.z[k]) / h
        p = self.phi
        c00 = p[i, j, k] * (1 - tx) + p[i + 1, j, k] * tx
        c01 = p[i, j, k + 1] * (1 - tx) + p[i + 1, j, k + 1] * tx
        c10 = p[i, j + 1, k] * (1 - tx) + p[i + 1, j + 1, k] * tx
        c11 = p[i, j + 1, k + 1] * (1 - tx) + p[i + 1, j + 1, k + 1] * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    # --------------------------------------------------------------------- #
    def droplet_surface_field(self, droplet, n_theta=181, n_phi=361, probe=None, off=1.5e-4):
        """
        E_n on the whole droplet cap, on a (polar, azimuth) grid.
        Returns (TH, PH, XS, YS, ZS, EN).
        """
        if probe is not None:
            off = probe
        probe = off
        Rc, z0 = droplet.Rc, droplet.z0
        th_max = np.pi / 2 if droplet.theta_c >= np.pi / 2 else droplet.theta_c
        th = np.linspace(0.0, np.arccos(-z0 / Rc) if abs(z0) < Rc else np.pi / 2, n_theta)
        ph = np.linspace(0.0, 2 * np.pi, n_phi)
        TH, PH = np.meshgrid(th, ph, indexing='ij')
        nx = np.sin(TH) * np.cos(PH)
        ny = np.sin(TH) * np.sin(PH)
        nz = np.cos(TH)
        XS = Rc * nx
        YS = Rc * ny
        ZS = z0 + Rc * nz
        p1 = self.interp(XS + probe * nx, YS + probe * ny, ZS + probe * nz)
        p2 = self.interp(XS + 2 * probe * nx, YS + 2 * probe * ny, ZS + 2 * probe * nz)
        phis = droplet.potential
        dphidn = (4.0 * (p1 - phis) - (p2 - phis)) / (2.0 * probe)
        return TH, PH, XS, YS, ZS, -dphidn

    # --------------------------------------------------------------------- #
    # ------------------------------------------------------------------ #
    #  Peak detectors
    #
    #  The manuscript reports two distinct surface-peak detectors, because the
    #  steering displacement is a sub-grid quantity and is therefore detector
    #  dependent. Both are provided here so that every published value can be
    #  reproduced exactly:
    #
    #    peak_primary : used for the Table 2 single-pin sweep (Delta_s column).
    #                   Plain parabolic refinement about the discrete maximum of
    #                   the raw profile, 1.2 mm fitting window, no smoothing.
    #
    #    peak_robust  : used for the Table 3b shape comparison and quoted in the
    #                   text as the "alternative robust detector". Light Hann
    #                   smoothing, search restricted to the pin side of the apex
    #                   (so an unsteered peak is correctly returned as ~0), and a
    #                   narrow 0.5 mm parabolic window. The magnitude is taken
    #                   from the raw profile, because a peak sharper than
    #                   quadratic is under-fitted by the parabola.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _smooth(y, k=15):
        if y.size < 2 * k + 1:
            return y
        ker = np.hanning(2 * k + 1); ker /= ker.sum()
        return np.convolve(y, ker, mode='same')

    @staticmethod
    def peak_primary(s, E, win=1.2e-3):
        """Primary detector: reproduces the Table 2 Delta_s column."""
        s = np.asarray(s, float); E = np.asarray(E, float)
        i0 = int(np.argmax(E))
        m = np.abs(s - s[i0]) <= win
        if m.sum() < 3:
            return float(s[i0]), float(E[i0])
        c = np.polyfit(s[m], E[m], 2)
        if c[0] >= 0:
            return float(s[i0]), float(E[i0])
        sp = float(np.clip(-c[1] / (2 * c[0]), s[m].min(), s[m].max()))
        return sp, float(np.polyval(c, sp))

    @staticmethod
    def peak_robust(s, E, s_apex, k=9, win=0.5e-3):
        """Robust detector: reproduces the Table 3b shape-comparison values."""
        s = np.asarray(s, float); E = np.asarray(E, float)
        Ew = Solver3D._smooth(E, k=k)
        idx = np.where(s <= s_apex + 1e-4)[0]          # pin side, apex included
        i0 = idx[int(np.argmax(Ew[idx]))]
        sel = np.abs(s - s[i0]) <= win
        if sel.sum() >= 3:
            c = np.polyfit(s[sel], Ew[sel], 2)
            if c[0] < 0:
                sp = float(np.clip(-c[1] / (2 * c[0]), s[sel].min(), s[sel].max()))
                Ep = float(np.polyval(c, sp))
            else:
                sp, Ep = float(s[i0]), float(Ew[i0])
        else:
            sp, Ep = float(s[i0]), float(Ew[i0])
        return sp, float(max(Ep, E[idx].max()))

    # retained for backward compatibility with the double-pin |E_n| peak search
    @staticmethod
    def _parabolic_peak(s, E, win=1.0e-3, smooth=True, srange=None):
        s = np.asarray(s, float); E = np.asarray(E, float)
        Ew = Solver3D._smooth(E) if smooth else E
        mask = np.ones(s.size, bool)
        if srange is not None:
            mask = (s >= srange[0]) & (s <= srange[1])
        idx = np.where(mask)[0]
        i0 = idx[int(np.argmax(Ew[idx]))]
        sel = np.abs(s - s[i0]) <= win
        if sel.sum() < 3:
            return float(s[i0]), float(E[i0])
        c = np.polyfit(s[sel], Ew[sel], 2)
        if c[0] >= 0:
            return float(s[i0]), float(Ew[i0])
        sp = float(np.clip(-c[1] / (2 * c[0]), s[sel].min(), s[sel].max()))
        return sp, float(np.polyval(c, sp))

    def great_circle_field(self, droplet, n_s=1201, probe=None, off=1.5e-4):
        """
        E_n along the xz-plane great circle of the droplet, parameterised by
        arc length s from the +x contact line -- the same coordinate used by the
        axisymmetric solver, so the two solvers can be compared directly.
        """
        if probe is not None:
            off = probe
        probe = off
        s, x, z, nx, nz = droplet.surface_points(n_s)
        y = np.zeros_like(x)
        p1 = self.interp(x + probe * nx, y, z + probe * nz)
        p2 = self.interp(x + 2 * probe * nx, y, z + 2 * probe * nz)
        phis = droplet.potential
        dphidn = (4.0 * (p1 - phis) - (p2 - phis)) / (2.0 * probe)
        return s, x, z, -dphidn
