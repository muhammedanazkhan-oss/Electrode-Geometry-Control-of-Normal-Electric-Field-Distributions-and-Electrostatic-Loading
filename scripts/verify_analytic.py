"""
verify_analytic.py -- Code verification against an exact solution.

A grounded conducting sphere of radius R in a uniform field E0 has the exact
potential  phi(rho, z) = -E0 * z * (1 - R^3/rho^3).  Restricted to z >= 0 this is
also the exact solution for a conducting HEMISPHERE of radius R sitting on a
grounded plane in a uniform normal field E0, because phi = 0 on z = 0 identically.
Its apex surface field is exactly 3*E0 (Jackson, 3rd ed.).

Imposing this analytic potential on the outer faces of a compact domain turns the
problem into a verification against a known solution: the computed potential field,
the droplet-surface normal field, and the apex enhancement factor must all converge
to the analytic values at the design order of the scheme.  This simultaneously
verifies (i) the cylindrical operator including the 1/r term, (ii) the
Shortley-Weller treatment of the curved interface, and (iii) the air-side
surface-field extraction.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

import numpy as np, sys
from ehd_geom import SphericalCapDroplet, GroundPlane
from ehd_axisym import AxisymSolver

mm = 1e-3
R = 4 * mm
E0 = 4e5          # arbitrary uniform field, V/m

def phi_exact(r, z):
    rho = np.sqrt(r ** 2 + z ** 2)
    rho = np.maximum(rho, 1e-12)
    return -E0 * z * (1.0 - R ** 3 / rho ** 3)

def En_exact(theta):
    """Analytic surface normal field on the hemisphere: 3 E0 cos(theta)."""
    return 3.0 * E0 * np.cos(theta)

print("=" * 82)
print("Verification against the exact hemisphere-on-plane solution, phi = -E0 z (1-R^3/rho^3)")
print("Domain 0<=r<=20 mm, 0<=z<=20 mm, analytic Dirichlet on the outer faces.")
print("=" * 82)
print(f"{'h (mm)':>8} {'E_apex/E0':>12} {'apex err (%)':>13} {'L2 err on E_n (%)':>19} {'max|phi| err (%)':>17}")

rows = []
for h in [0.20 * mm, 0.10 * mm, 0.05 * mm, 0.025 * mm]:
    drop = SphericalCapDroplet(R, np.pi / 2, potential=0.0)
    bot = GroundPlane(0.0, potential=0.0)
    s = AxisymSolver(r_max=20 * mm, z_max=20 * mm, h=h, conductors=[bot, drop],
                     outer_bc=phi_exact)
    s.solve()
    ss, xs, zs, En = s.surface_normal_field(drop, n_s=1601, probe=3 * h)

    # polar angle from the apex, for the analytic comparison
    theta = np.arctan2(np.abs(xs), zs)
    Ean = En_exact(theta)
    ia = np.argmin(np.abs(xs))
    apex_ratio = En[ia] / E0

    # restrict the L2 norm to theta <= 75 deg: near the contact line the analytic
    # field -> 0 and the relative measure becomes meaningless
    m = theta < np.deg2rad(75)
    l2 = np.sqrt(np.mean((En[m] - Ean[m]) ** 2)) / (3 * E0) * 100

    # interior potential error on a probe line above the apex
    zt = np.linspace(R + 0.5 * mm, 15 * mm, 200)
    pe = s.interp(np.zeros_like(zt), zt)
    perr = np.max(np.abs(pe - phi_exact(np.zeros_like(zt), zt))) / (E0 * 15 * mm) * 100

    rows.append((h / mm, apex_ratio, l2, perr))
    print(f"{h/mm:8.3f} {apex_ratio:12.5f} {100*(apex_ratio-3)/3:13.4f} {l2:19.4f} {perr:17.5f}")

a = np.array([r[1] for r in rows])
e = np.abs(a - 3.0)
print()
for k in range(len(e) - 1):
    print(f"  order between h={rows[k][0]:.3f} and h={rows[k+1][0]:.3f} mm : "
          f"p = {np.log(e[k]/e[k+1])/np.log(2.0):.2f}")
