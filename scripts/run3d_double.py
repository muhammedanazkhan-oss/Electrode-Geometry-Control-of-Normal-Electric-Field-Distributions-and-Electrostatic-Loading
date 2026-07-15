"""Double-pin (+/-Vp) case, 3-D Robin far-field, plus its own mesh study."""
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
mmv=1e-3
RES=os.path.join(DATA, 'double_pin.json')
db=json.load(open(RES)) if os.path.exists(RES) else {}

def run(Vp_kV, h, xoff=6.0):
    key=f"Vp{Vp_kV}|h{h}|x{xoff}"
    if key in db: print("cached",key,flush=True); return
    cond,drop=fs_double_pin(Vp=Vp_kV*1e3, x_off=xoff*mmv)
    t0=time.time()
    s3=Solver3D((-16*mmv,16*mmv),(0.,14*mmv),(-4*mmv,18*mmv), h*mmv, cond,
                ybc='neumann', robin=True)
    N=int(s3.unknown.sum()); s3.solve()
    ss,xg,zg,E=s3.great_circle_field(drop, n_s=1601, off=0.15*mmv)
    # two maxima: split the great circle at the apex (s=piR/2) and find each side's peak
    s_mid=ss[np.argmin(np.abs(xg))]
    left=ss<s_mid; right=ss>=s_mid
    slp,Elp=s3._parabolic_peak(ss[left],np.abs(E[left]))
    srp,Erp=s3._parabolic_peak(ss[right],np.abs(E[right]))
    ia=np.argmin(np.abs(xg)); Eap=float(E[ia])
    # pin-tip field (max |E| anywhere near a tip) from the volume solution
    db[key]=dict(Vp=Vp_kV,h=h,xoff=xoff,N=N,it=int(s3.iters),
                 s_left_mm=float(slp/mmv), E_left=float(Elp),
                 s_right_mm=float(srp/mmv), E_right=float(Erp),
                 Eapex=abs(Eap), apex_frac=float(abs(Eap)/max(Elp,Erp)),
                 t=round(time.time()-t0,1))
    json.dump(db,open(RES,'w'),indent=2)
    r=db[key]
    print(f"{key}: peaks s=({r['s_left_mm']:.2f},{r['s_right_mm']:.2f})mm "
          f"E=({r['E_left']:.3e},{r['E_right']:.3e}) apex={Eap:.3e}({100*r['apex_frac']:.0f}%) N={N} {r['t']}s",flush=True)

if __name__=='__main__':
    run(20.0, 0.20)
