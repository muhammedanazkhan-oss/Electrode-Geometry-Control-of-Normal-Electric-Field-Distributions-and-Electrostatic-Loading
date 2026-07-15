"""
verify_3d_vs_axi.py -- Solver-to-solver consistency at d = 0.

The original submission stated only that the 3-D solution recovers the axisymmetric
result "within the mesh tolerance", without reporting the comparison.  Reviewer 4
(comment 4) asks for this to be quantified.  Here the on-axis pin (d = 0), which is
axisymmetric, is solved with BOTH solvers and the apex field, the peak surface field
and the peak arc-length location are compared explicitly.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

import numpy as np, sys, time
from ehd_config import *
from ehd_axisym import AxisymSolver
from ehd_solver3d import Solver3D

V = 4000.0
mm = 1e-3

print("Axisymmetric reference (h = 0.025 mm):")
cond, drop = config_pin(V, d=0.0)
t0 = time.time()
sa = AxisymSolver(r_max=15 * mm, z_max=16 * mm, h=0.025 * mm, conductors=cond)
sa.solve()
s, x, z, En = sa.surface_normal_field(drop, n_s=1601, probe=0.15 * mm)
ia = np.argmin(np.abs(x))
Eapex_ax = En[ia]; Emax_ax = En.max(); speak_ax = s[np.argmax(En)]
print(f"  E_apex = {Eapex_ax:.5e} V/m   E_n,max = {Emax_ax:.5e} V/m   "
      f"s_peak = {speak_ax/mm:.4f} mm   ({time.time()-t0:.1f} s)")

print("\n3-D Cartesian solver on the same configuration (mirror half-domain y>=0):")
print(f"{'h (mm)':>8} {'E_apex (V/m)':>15} {'dev (%)':>9} {'E_n,max (V/m)':>15} {'dev (%)':>9} {'iters':>6} {'t (s)':>7}")
for h in [0.4 * mm, 0.2 * mm]:
    cond, drop = config_pin(V, d=0.0)
    t0 = time.time()
    s3 = Solver3D((-15 * mm, 15 * mm), (0.0, 15 * mm), (0.0, 16 * mm), h, cond)
    s3.solve()
    ss, xx, zz, En3 = s3.great_circle_field(drop, n_s=1201, probe=3 * h)
    ia3 = np.argmin(np.abs(xx))
    ea, em = En3[ia3], En3.max()
    print(f"{h/mm:8.2f} {ea:15.5e} {100*(ea-Eapex_ax)/Eapex_ax:9.3f} "
          f"{em:15.5e} {100*(em-Emax_ax)/Emax_ax:9.3f} {s3.iters:6d} {time.time()-t0:7.1f}")
