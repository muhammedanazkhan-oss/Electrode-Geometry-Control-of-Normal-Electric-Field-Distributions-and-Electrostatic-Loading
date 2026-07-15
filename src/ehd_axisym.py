"""
ehd_axisym.py -- Axisymmetric cylindrical finite-difference Laplace solver with a
Shortley-Weller embedded-boundary treatment of the curved droplet interface.

Solves   (1/r) d/dr ( r dphi/dr ) + d2phi/dz2 = 0

on 0 <= r <= r_max, 0 <= z <= z_max, subject to Dirichlet conditions on every
conductor (perfect-conductor closure, Section 1.1) and homogeneous Neumann (open)
conditions on the lateral and upper outer faces.

Discretisation
--------------
The radial operator uses the flux-conserving stencil that retains the 1/r term:

  [ (r_{i+1/2}/r_i)(phi_{i+1,j}-phi_{i,j}) - (r_{i-1/2}/r_i)(phi_{i,j}-phi_{i-1,j}) ]/dr^2

with the axis limit (1/r)(r phi_r)_r -> 2 phi_rr enforced at i = 0 through the
mirror symmetry phi(-h) = phi(+h).

Where a grid link is cut by a conductor surface at a fraction theta in (0,1] of the
cell, the uniform stencil is replaced by the non-uniform three-point
(Shortley-Weller) formula.  This restores second-order accuracy on the curved
droplet interface and the rounded pin tip, in place of the first-order, erratic
convergence produced by a stair-stepped mask (Reviewer 4, comment 4).

Surface-field extraction
------------------------
E_n = -dphi/dn is evaluated from the air side only, by probing phi along the exact
outward normal at two fixed physical offsets and combining with the known surface
potential in a second-order one-sided difference.  Because the probe offsets are
fixed in metres rather than in cells, the extracted field converges with the
interior solution rather than with the local stencil width, and the stencil never
samples the conductor interior.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import pyamg

from ehd_geom import crossing_fraction

# Conductor faces frequently coincide exactly with grid planes (e.g. the substrate
# at z = 0).  Inflating every conductor by 1 nm makes such nodes unambiguously
# interior without any physically meaningful change of geometry.
EPS_INFLATE = 1e-9


class AxisymSolver:
    def __init__(self, r_max, z_max, h, conductors, outer_bc=None, enclosure=None, z_min=0.0, robin=False):
        """outer_bc : callable(r, z) -> phi.  If given, the lateral (r = r_max) and
        upper (z = z_max) faces are Dirichlet at the prescribed value instead of
        homogeneous Neumann.  Used to impose an exact analytic far field for
        code verification against a known solution."""
        self.outer_bc = outer_bc
        self.enclosure = enclosure
        self.robin = bool(robin)
        self.h = float(h)
        self.nr = int(round(r_max / h)) + 1
        self.nz = int(round((z_max - z_min) / h)) + 1
        self.r = np.arange(self.nr) * h
        self.z = z_min + np.arange(self.nz) * h
        self.conductors = conductors
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing='ij')
        self._build_masks()
        self.phi = None

    # -- inflated implicit function of conductor c ------------------------- #
    @staticmethod
    def _fc(c):
        return lambda x, y, z: c.f(x, y, z) - EPS_INFLATE

    # --------------------------------------------------------------------- #
    def _build_masks(self):
        self.dirichlet = np.zeros((self.nr, self.nz), bool)
        self.dval = np.zeros((self.nr, self.nz))
        zeros = np.zeros_like(self.R)
        for c in self.conductors:
            m = self._fc(c)(self.R, zeros, self.Z) < 0.0
            self.dirichlet |= m
            self.dval[m] = c.potential
        if self.enclosure is not None:
            ob = np.zeros((self.nr, self.nz), bool)
            ob[-1, :] = True
            ob[:, -1] = True
            if self.z[0] < -1e-12:
                ob[:, 0] = True
            ob &= ~self.dirichlet
            self.dval[ob] = float(self.enclosure)
            self.dirichlet |= ob
        if self.outer_bc is not None:
            ob = np.zeros((self.nr, self.nz), bool)
            ob[-1, :] = True
            ob[:, -1] = True
            ob &= ~self.dirichlet
            self.dval[ob] = self.outer_bc(self.R[ob], self.Z[ob])
            self.dirichlet |= ob
        self.unknown = ~self.dirichlet

    # --------------------------------------------------------------------- #
    def _cut_info(self, i, j, di, dj):
        """theta in (0,1] and surface potential for links from unknown (i,j) into a
        conductor at (i+di, j+dj).  Crossings are found by bisection on the
        conductor's implicit function -- no signed-distance assumption."""
        h = self.h
        r0, z0 = self.r[i], self.z[j]
        r1, z1 = r0 + di * h, z0 + dj * h
        p0 = np.vstack([r0, np.zeros_like(r0), z0])
        p1 = np.vstack([r1, np.zeros_like(r1), z1])
        # Default: the neighbour node is itself Dirichlet at distance theta = 1
        # (this covers prescribed outer-boundary nodes, which are not conductors).
        # A conductor surface strictly inside the link overrides it below.
        theta = np.ones(i.size)
        gval = self.dval[i + di, j + dj].copy()
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
        h, nr, nz = self.h, self.nr, self.nz
        unk = np.argwhere(self.unknown)
        idx = -np.ones((nr, nz), int)
        idx[self.unknown] = np.arange(unk.shape[0])
        N = unk.shape[0]
        I, J = unk[:, 0], unk[:, 1]
        me = idx[I, J]

        rows, cols, vals = [], [], []
        rhs = np.zeros(N)
        diag = np.zeros(N)

        def add(r_, c_, v_):
            r_ = np.atleast_1d(np.asarray(r_, np.int64))
            c_ = np.atleast_1d(np.asarray(c_, np.int64))
            v_ = np.atleast_1d(np.asarray(v_, np.float64))
            n = max(r_.size, c_.size, v_.size)
            rows.append(np.ascontiguousarray(np.broadcast_to(r_, (n,))))
            cols.append(np.ascontiguousarray(np.broadcast_to(c_, (n,))))
            vals.append(np.ascontiguousarray(np.broadcast_to(v_, (n,))))

        ri = self.r[I]
        off_axis = I > 0

        # ---------------- radial links (off-axis nodes) ---------------------
        for di in (+1, -1):
            Ii = I + di
            valid = off_axis & (Ii >= 0) & (Ii <= nr - 1)
            r_face = ri + di * 0.5 * h
            w = np.zeros(N)
            w[off_axis] = (r_face[off_axis] / ri[off_axis]) / h ** 2
            sel = np.where(valid)[0]
            if sel.size:
                nb_dir = self.dirichlet[Ii[sel], J[sel]]
                a = sel[~nb_dir]
                if a.size:
                    add(me[a], idx[Ii[a], J[a]], w[a])
                    diag[a] -= w[a]
                b = sel[nb_dir]
                if b.size:
                    th, g = self._cut_info(I[b], J[b], di, 0)
                    wb = w[b] / th
                    rhs[b] -= wb * g
                    diag[b] -= wb
            # links leaving the domain at r = r_max: asymptotic monopole (Robin) BC
            if self.robin and di == +1:
                extern = off_axis & (Ii > nr - 1)
                se = np.where(extern)[0]
                if se.size:
                    rho = np.sqrt(self.r[I[se]]**2 + self.z[J[se]]**2)
                    rho_g = np.sqrt((self.r[I[se]]+h)**2 + self.z[J[se]]**2)
                    # phi_ghost = phi_node * rho/rho_g  ->  contributes w*(rho/rho_g) to diagonal
                    diag[se] += (r_face[se]/self.r[I[se]]/h**2) * (rho/rho_g - 1.0)

        # ---------------- axis nodes: (1/r)(r phi_r)_r -> 2 phi_rr ----------
        ax = np.where(~off_axis)[0]
        if ax.size:
            nb_dir = self.dirichlet[np.ones(ax.size, int), J[ax]]
            w0 = np.full(ax.size, 4.0 / h ** 2)     # 2*(2/h^2), mirrored links
            a = ax[~nb_dir]
            if a.size:
                add(me[a], idx[np.ones(a.size, int), J[a]], w0[~nb_dir])
                diag[a] -= w0[~nb_dir]
            b = ax[nb_dir]
            if b.size:
                th, g = self._cut_info(I[b], J[b], +1, 0)
                wb = w0[nb_dir] / th
                rhs[b] -= wb * g
                diag[b] -= wb

        # ---------------- axial links --------------------------------------
        for dj in (+1, -1):
            Jj = J + dj
            valid = (Jj >= 0) & (Jj <= nz - 1)
            w = np.full(N, 1.0 / h ** 2)
            sel = np.where(valid)[0]
            if sel.size:
                nb_dir = self.dirichlet[I[sel], Jj[sel]]
                a = sel[~nb_dir]
                if a.size:
                    add(me[a], idx[I[a], Jj[a]], w[a])
                    diag[a] -= w[a]
                b = sel[nb_dir]
                if b.size:
                    th, g = self._cut_info(I[b], J[b], 0, dj)
                    wb = w[b] / th
                    rhs[b] -= wb * g
                    diag[b] -= wb
            # links leaving the domain at z = z_max / z_min: asymptotic monopole (Robin)
            if self.robin:
                extern = (Jj < 0) | (Jj > nz - 1)
                se = np.where(extern)[0]
                if se.size:
                    zc = self.z[J[se]] + dj*h
                    rho = np.sqrt(self.r[I[se]]**2 + self.z[J[se]]**2)
                    rho_g = np.sqrt(self.r[I[se]]**2 + zc**2)
                    diag[se] += (1.0/h**2) * (rho/np.maximum(rho_g,1e-9) - 1.0)

        add(me, me, diag)
        A = coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(N, N)).tocsr()
        self.A, self.rhs, self.idx, self.N = A, rhs, idx, N
        return A, rhs

    # --------------------------------------------------------------------- #
    def solve(self, tol=1e-9, direct_below=550000):
        A, b = self.assemble()
        if self.N <= direct_below:
            x = spsolve(A.tocsc(), b)
            self.iters, self.relres = 0, 0.0
        else:
            ml = pyamg.ruge_stuben_solver(-A.tocsr(), max_coarse=400)
            res = []
            x = ml.solve(-b, tol=tol, accel='cg', residuals=res, maxiter=400)
            self.iters = len(res)
            self.relres = res[-1] / max(res[0], 1e-300)
        phi = self.dval.copy()
        phi[self.unknown] = x
        self.phi = phi
        return phi

    # --------------------------------------------------------------------- #
    def interp(self, r_q, z_q):
        """Bilinear interpolation of phi."""
        h = self.h
        rq = np.clip(np.asarray(r_q, float), 0, self.r[-1] - 1e-12)
        zq = np.clip(np.asarray(z_q, float), self.z[0], self.z[-1] - 1e-12)
        i = np.clip(np.floor(rq / h).astype(int), 0, self.nr - 2)
        j = np.clip(np.floor((zq - self.z[0]) / h).astype(int), 0, self.nz - 2)
        tr, tz = rq / h - i, (zq - self.z[0]) / h - j
        p = self.phi
        return ((1 - tr) * (1 - tz) * p[i, j] + tr * (1 - tz) * p[i + 1, j]
                + (1 - tr) * tz * p[i, j + 1] + tr * tz * p[i + 1, j + 1])

    # --------------------------------------------------------------------- #
    def surface_normal_field(self, droplet, n_s=1601, probe=None, off=1.5e-4):
        """Air-side normal field E_n on the droplet surface.

        The potential is probed at two FIXED physical offsets (off, 2*off) along
        the exact outward normal -- independent of the mesh spacing h -- and
        combined with the known surface potential phi_s in a second-order one-sided
        difference.  Fixing the offsets in metres (default 0.15 mm) decouples the
        extracted surface field from the local stencil width, so a mesh-refinement
        study measures the discretisation error of the interior solution rather than
        a changing extraction stencil (verified against the analytic hemisphere).
        """
        if probe is not None:
            off = probe
        s, x, z, nx, nz = droplet.surface_points(n_s)
        rr = np.abs(x)
        nr_ = np.where(x >= 0, nx, -nx)
        p1 = self.interp(rr + off * nr_, z + off * nz)
        p2 = self.interp(rr + 2 * off * nr_, z + 2 * off * nz)
        phis = droplet.potential
        dphidn = (4.0 * (p1 - phis) - (p2 - phis)) / (2.0 * off)
        return s, x, z, -dphidn
