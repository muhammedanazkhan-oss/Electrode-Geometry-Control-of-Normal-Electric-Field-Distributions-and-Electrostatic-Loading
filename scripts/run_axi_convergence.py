"""
run_axi_convergence.py -- Domain-independence and mesh-convergence (GCI) for the
axisymmetric free-space parallel-plate and on-axis-pin cells.  Checkpoints to JSON.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

import numpy as np, sys, json, time
from ehd_config import *
from ehd_axisym import AxisymSolver

V = 4000.0
mmv = 1e-3

def apex_and_max(cfg_fn, rmax, zmin, zmax, h):
    cond, drop = cfg_fn(V)
    s = AxisymSolver(rmax, zmax, h, cond, enclosure=0.0, z_min=zmin)
    s.solve()
    ss, xx, zz, En = s.surface_normal_field(drop, n_s=1601, probe=max(3 * h, 0.3 * mmv))
    ia = np.argmin(np.abs(xx))
    return dict(Eapex=float(En[ia]), Emax=float(En.max()),
                speak=float(ss[np.argmax(En)]), N=int(s.unknown.sum()))

out = {}
# ---- domain independence at h = 0.1 mm ----
t0 = time.time()
dom = []
for rmax, zmin, zmax in [(30, -10, 30), (45, -15, 45), (60, -20, 60)]:
    row = {'rmax_mm': rmax, 'zrange_mm': [zmin, zmax]}
    row['PP'] = apex_and_max(fs_pp, rmax * mmv, zmin * mmv, zmax * mmv, 0.1 * mmv)
    row['PT'] = apex_and_max(fs_pin, rmax * mmv, zmin * mmv, zmax * mmv, 0.1 * mmv)
    dom.append(row)
out['domain_independence'] = dom
out['domain_time_s'] = time.time() - t0
json.dump(out, open(os.path.join(DATA, 'axi_domain.json'), 'w'), indent=2)
print("domain independence done", f"{out['domain_time_s']:.1f}s")
for r in dom:
    print(f"  r={r['rmax_mm']:>3} PP={r['PP']['Eapex']:.4e} PT={r['PT']['Eapex']:.4e}")
