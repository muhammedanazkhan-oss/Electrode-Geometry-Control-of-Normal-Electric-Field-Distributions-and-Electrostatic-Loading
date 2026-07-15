"""Formal 3-mesh GCI for the 3-D cases (rev#10), domain shifted so pins never land
on grid nodes. Consistent refinement ratio 4/3. Checkpointed per (case,h)."""
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
mm=1e-3; V=4000.; SH=0.07*mm   # domain shift to avoid pin-on-node
RES=os.path.join(DATA, 'gci3d.json')
db=json.load(open(RES)) if os.path.exists(RES) else {}

def single(d,h):
    key=f"single_d{d}_h{h:.3f}"
    if key in db: print("cached",key,flush=True); return
    cond,drop=fs_pin(V,d=d*mm); t0=time.time()
    s3=Solver3D((-14*mm+SH,14*mm+SH),(0.,13*mm),(-4*mm+SH,17*mm+SH),h*mm,cond,ybc='neumann',robin=True)
    N=int(s3.unknown.sum()); s3.solve()
    s,xg,zg,E=s3.great_circle_field(drop,n_s=1401,off=0.15*mm); sap=s[np.argmin(np.abs(xg))]
    sp,Ep=s3._parabolic_peak(s,E,srange=(sap-3.2e-3,sap+0.3e-3))
    db[key]=dict(d=d,h=h,N=N,Eapex=float(E[np.argmin(np.abs(xg))]),Emax=float(Ep),
                 speak_mm=float(sp/mm),ds_mm=float(abs(sp-sap)/mm),t=round(time.time()-t0,1))
    json.dump(db,open(RES,'w'),indent=2); print(key,{k:db[key][k] for k in['Emax','ds_mm','N','t']},flush=True)

def double(h):
    key=f"double_h{h:.3f}"
    if key in db: print("cached",key,flush=True); return
    cond,drop=fs_double_pin(Vp=20e3,x_off=6*mm); t0=time.time()
    s3=Solver3D((-16*mm+SH,16*mm+SH),(0.,13*mm),(-4*mm+SH,17*mm+SH),h*mm,cond,ybc='neumann',robin=True)
    N=int(s3.unknown.sum()); s3.solve()
    s,xg,zg,E=s3.great_circle_field(drop,n_s=1601,off=0.15*mm); sm=s[np.argmin(np.abs(xg))]
    sl,El=s3._parabolic_peak(s[s<sm],np.abs(E[s<sm])); sr,Er=s3._parabolic_peak(s[s>=sm],np.abs(E[s>=sm]))
    db[key]=dict(h=h,N=N,s_left_mm=float(sl/mm),s_right_mm=float(sr/mm),Epeak=float(0.5*(abs(El)+abs(Er))),t=round(time.time()-t0,1))
    json.dump(db,open(RES,'w'),indent=2); print(key,{k:db[key][k] for k in['s_left_mm','s_right_mm','Epeak','N','t']},flush=True)

for a in sys.argv[1:]:
    if a.startswith('s'): single(6, float(a[1:]))
    elif a.startswith('D'): double(float(a[1:]))
