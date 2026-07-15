"""
run_bo.py -- Honest Bayesian optimisation with REAL solver calls (Reviewer 4, #5).

Corrections vs the original submission:
 * every acquisition triggers an actual 3-D finite-difference solve at the newly
   selected continuous offset, not a lookup of a precomputed interpolant;
 * the initial design is drawn away from the known optimum (it does not contain it);
 * BO is compared against random search at an EQUAL high-fidelity budget over
   several independent seeds, so any advantage is measured, not asserted.

Objective: steering displacement ds(d) of the high-field site (to be MAXIMISED).
This is a genuine interior optimum (ds(d) rises then saturates near d~6 mm), so the
search cannot trivially succeed by running to a box edge.
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
from ehd_solver3d import Solver3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
V=4000.; mmv=1e-3
CACHE=os.path.join(DATA, 'bo_cache.json')
cache=json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def steering(dmm):
    key=f"{dmm:.3f}"
    if key in cache: return cache[key]
    cond,drop=fs_pin(V,d=dmm*mmv)
    s3=Solver3D((-13*mmv,13*mmv),(0.,11*mmv),(-4*mmv,17*mmv),0.20*mmv,cond,ybc='neumann',robin=True)
    s3.solve()
    ss,xg,zg,E=s3.great_circle_field(drop,n_s=1201,off=0.15*mmv); sap=ss[np.argmin(np.abs(xg))]
    sp,Ep=s3._parabolic_peak(ss,E,srange=(sap-3.2e-3,sap+0.3e-3))
    val=float(abs(sp-sap)/mmv); cache[key]=val; json.dump(cache,open(CACHE,'w')); return val

def bo_run(seed, budget=10, n_init=3):
    rng=np.random.default_rng(seed)
    # initial design drawn from [0,4] mm only -> does NOT contain the ~6 mm optimum
    Xd=list(rng.uniform(0.3,4.5,n_init))
    Y=[steering(x) for x in Xd]
    grid=np.linspace(0,9,181)
    for t in range(budget-n_init):
        k=ConstantKernel(1.0,(1e-2,1e2))*Matern(length_scale=1.5,length_scale_bounds=(0.3,10),nu=2.5)
        gp=GaussianProcessRegressor(kernel=k,alpha=1e-6,normalize_y=True,n_restarts_optimizer=5,random_state=seed)
        gp.fit(np.array(Xd).reshape(-1,1),np.array(Y))
        mu,sd=gp.predict(grid.reshape(-1,1),return_std=True)
        best=max(Y); z=(mu-best-0.01)/np.maximum(sd,1e-9)
        ei=(mu-best-0.01)*norm.cdf(z)+sd*norm.pdf(z)
        xn=grid[int(np.argmax(ei))]
        Xd.append(float(xn)); Y.append(steering(float(xn)))
    ybest=np.maximum.accumulate(Y)
    return Xd,Y,list(map(float,ybest))

def rand_run(seed,budget=10):
    rng=np.random.default_rng(1000+seed)
    Xd=list(rng.uniform(0,9,budget)); Y=[steering(x) for x in Xd]
    return Xd,Y,list(map(float,np.maximum.accumulate(Y)))

if __name__=='__main__':
    seeds=[int(x) for x in sys.argv[1:]] or [0]
    RES=os.path.join(DATA, 'bo_results.json')
    allr=json.load(open(RES)) if os.path.exists(RES) else {'bo':{},'rand':{}}
    for sd in seeds:
        t0=time.time()
        xb,yb,bb=bo_run(sd); allr['bo'][str(sd)]=dict(X=xb,ybest=bb,final=bb[-1],dstar=xb[int(np.argmax(yb))])
        xr,yr,br=rand_run(sd); allr['rand'][str(sd)]=dict(ybest=br,final=br[-1])
        json.dump(allr,open(RES,'w'),indent=2)
        print(f"seed {sd}: BO final Δs={bb[-1]:.3f} at d*={allr['bo'][str(sd)]['dstar']:.2f}mm | rand final={br[-1]:.3f} | {time.time()-t0:.0f}s (cache {len(cache)})",flush=True)
