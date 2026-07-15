"""Off-axis pin sweep (3-D, Robin far-field, finite disc). One or more offsets per
call, checkpointed to off_sweep.json so the campaign survives the per-call
time limit."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

import numpy as np, sys, json, os, time
from ehd_config import *
from ehd_solver3d import Solver3D
V=4000.; mmv=1e-3
RES=os.path.join(DATA, 'off_sweep.json')
db=json.load(open(RES)) if os.path.exists(RES) else {}
DOM=dict(x=14, y=12, zmin=-4, zmax=18, h=0.20)

def peak_info(s_arr, E_arr):
    ipk=int(np.argmax(E_arr)); return float(E_arr[ipk]), float(s_arr[ipk])

def run(d, h=DOM['h']):
    key=f"d{d}|h{h}"
    if key in db: print("cached",key,flush=True); return
    cond,drop=fs_pin(V, d=d*mmv)
    t0=time.time()
    s3=Solver3D((-DOM['x']*mmv,DOM['x']*mmv),(0.,DOM['y']*mmv),(DOM['zmin']*mmv,DOM['zmax']*mmv),
                h*mmv, cond, ybc='neumann', robin=True)
    N=int(s3.unknown.sum()); s3.solve()
    ss,xg,zg,E=s3.great_circle_field(drop, n_s=1401, off=0.15*mmv)
    ia=np.argmin(np.abs(xg))
    speak,Emax=s3.peak_primary(ss,E)
    s_apex=float(ss[ia])
    db[key]=dict(d=d,h=h,N=N,it=int(s3.iters),relres=float(s3.relres),
                 Eapex=float(E[ia]), Emax=Emax, speak_mm=float(speak/mmv),
                 s_apex_mm=float(s_apex/mmv), ds_mm=float(abs(speak-s_apex)/mmv),
                 t=round(time.time()-t0,1))
    json.dump(db,open(RES,'w'),indent=2)
    print(f"{key}: Eapex={db[key]['Eapex']:.4e} Emax={Emax:.4e} ds={db[key]['ds_mm']:.3f}mm N={N} {db[key]['t']}s",flush=True)

if __name__=='__main__':
    for d in [int(x) for x in sys.argv[1:]]:
        run(d)
