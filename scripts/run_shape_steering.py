"""
run_shape_steering.py -- Shape-dependence of the off-axis steering (round-2 review, #6).

Recomputes the three-dimensional off-axis-pin steering on the gravitationally
equilibrated Young-Laplace profiles (theta_c = 70, 90, 110 deg, equal volume) and on
the idealised hemisphere, at IDENTICAL mesh and extraction settings, so that the
shape dependence is isolated from the (known) mesh limitation of the absolute
steering displacement. Checkpointed per (shape, offset).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

import numpy as np, sys, json, os, time
from ehd_config import *
from ehd_geom import ProfileDroplet, SphericalCapDroplet, Disc, RoundedPin
from ehd_solver3d import Solver3D
from ehd_young_laplace import solve_shape_for_volume
mm=1e-3; V=4000.
RES=os.path.join(DATA, 'shape_steering.json')
db=json.load(open(RES)) if os.path.exists(RES) else {}
Vhemi=(2/3)*np.pi*R_DROP**3

def make_drop(shape):
    if shape=='hemi':
        return SphericalCapDroplet(R_DROP, np.pi/2, potential=V)
    if shape=='hemiprof':
        # CONTROL: the same hemisphere expressed as a numerical profile, so the
        # ProfileDroplet path (masking + full-great-circle extraction) can be
        # validated against the analytic SphericalCapDroplet result.
        th=np.linspace(0,np.pi/2,600)
        return ProfileDroplet(R_DROP*np.cos(th), R_DROP*np.sin(th), potential=V)
    th=int(shape)
    b,xr,zr=solve_shape_for_volume(np.deg2rad(th), Vhemi, GAMMA, RHO_L, G)
    zphys=zr.max()-zr                      # flip: apex on top, contact line at z=0
    return ProfileDroplet(xr, zphys, potential=V)

def run(shape, d, h=0.20):
    key=f"{shape}_d{d}_h{h}"
    if key in db: print("cached",key,flush=True); return
    drop=make_drop(shape)
    cond=[Disc(DISC_R,-DISC_T,0.0,potential=V), drop,
          RoundedPin(d*mm,0.,PIN_A,H_GAP,H_GAP+PIN_LEN,R_TIP,potential=0.)]
    t0=time.time()
    s3=Solver3D((-14*mm,14*mm),(0.,12*mm),(-4*mm,18*mm),h*mm,cond,ybc='neumann',robin=True)
    N=int(s3.unknown.sum()); s3.solve()
    s,xg,zg,E=s3.great_circle_field(drop,n_s=1401,off=0.15*mm)
    ia=int(np.argmin(np.abs(xg))); sap=float(s[ia])
    sp,Ep = s3.peak_robust(s, E, sap)
    db[key]=dict(shape=shape,d=d,h=h,N=N,perim_mm=float(s[-1]/mm),s_apex_mm=sap/mm,
                 Eapex=float(E[ia]),Emax=float(Ep),speak_mm=float(sp/mm),
                 ds_mm=float(abs(sp-sap)/mm), ds_norm=float(abs(sp-sap)/s[-1]),
                 t=round(time.time()-t0,1))
    json.dump(db,open(RES,'w'),indent=2)
    r=db[key]; print(f"{key}: perim={r['perim_mm']:.2f} Eapex={r['Eapex']:.3e} Emax={r['Emax']:.3e} "
                     f"ds={r['ds_mm']:.3f}mm ds/L={r['ds_norm']*100:.2f}% N={N} {r['t']}s",flush=True)

if __name__=='__main__':
    for a in sys.argv[1:]:
        sh,d=a.split(':'); run(sh,int(d))
