"""
run_ml_pareto.py -- Surrogate benchmark with geometry-level validation (R4.5) and
Pareto analysis with interval dominance (R4.6), on the corrected free-space data.

Key methodological corrections vs the original submission:
 * The exact V-linearity E = c(d)*V is factored out analytically: surrogates model
   the geometry-dependent coefficient c(d)=E_n,max/V, one value per solved geometry,
   rather than 60 voltage-scaled duplicates of 10 geometries.
 * Generalisation is measured by leave-one-GEOMETRY-out CV (each offset withheld
   entirely), reporting per-geometry error, not random CV over voltage-scaled points.
 * Pareto dominance is evaluated on unrounded objectives with an uncertainty
   tolerance (interval / epsilon-dominance) derived from the mesh study.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

import numpy as np, json, sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
try:
    from xgboost import XGBRegressor
    HAVE_XGB=True
except Exception:
    HAVE_XGB=False

master=json.load(open(os.path.join(DATA, 'master.json')))
ET=master['ET']; V=4000.0
sweep=master['sweep']
d=np.array([r['d'] for r in sweep],float)                 # offsets (mm)
Emax=np.array([r['Emax'] for r in sweep])                 # steered-peak field at 4 kV
c=Emax/V                                                  # geometry coefficient c(d)=E/V
Vcross=np.array([r['Vcross_kV'] for r in sweep])
ds=np.array([r['ds'] for r in sweep])

# ---------- surrogate benchmark: leave-one-geometry-out on c(d) ----------
X=d.reshape(-1,1)
models={
 'Linear':LinearRegression(),
 'RandomForest':RandomForestRegressor(n_estimators=400,max_depth=6,random_state=42),
 'GPR-Matern52':GaussianProcessRegressor(
     kernel=ConstantKernel(1.0,(1e-3,1e3))*Matern(length_scale=2.0,length_scale_bounds=(0.3,20.0),nu=2.5),
     alpha=1e-10, normalize_y=True, n_restarts_optimizer=15, random_state=0),
}
if HAVE_XGB:
    models['XGBoost']=XGBRegressor(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,random_state=42)

def loo_geometry(model):
    preds=np.zeros_like(c); 
    for i in range(len(d)):
        tr=np.arange(len(d))!=i
        m=clone(model)
        m.fit(X[tr],c[tr]); preds[i]=m.predict(X[i:i+1])[0]
    return preds

results={}
for name,m in models.items():
    p=loo_geometry(m)
    results[name]=dict(R2=float(r2_score(c,p)),
                       MAE_pct=float(100*mean_absolute_error(c,p)/c.mean()),
                       max_err_pct=float(100*np.max(np.abs(p-c))/c.mean()))
# ---------- Pareto with interval dominance ----------
# objectives: maximise ds, minimise Vcross. mesh uncertainty on the 3-D field ~8%
# -> propagate to Vcross (prop. to 1/Emax) and to ds (~0.05 mm detection + mesh)
unc_V = 0.08*Vcross      # ~8% mesh uncertainty
unc_ds = np.full_like(ds, 0.06) + 0.03*ds
def dominated(i,j,tolV,tolds):
    # j dominates i if j is >= on ds and <= on Vcross beyond tolerance, strictly better on one
    better_ds = ds[j] >= ds[i]-tolds[i]
    better_V  = Vcross[j] <= Vcross[i]+tolV[i]
    strict = (ds[j] > ds[i]+tolds[i]) or (Vcross[j] < Vcross[i]-tolV[i])
    return better_ds and better_V and strict
def pareto(tolV,tolds):
    front=[]
    for i in range(len(d)):
        if not any(dominated(i,j,tolV,tolds) for j in range(len(d)) if j!=i):
            front.append(int(d[i]))
    return front
front_strict=pareto(np.zeros_like(Vcross),np.zeros_like(ds))
front_interval=pareto(unc_V,unc_ds)
# maximin optimum (min-max normalised)
dn=(ds-ds.min())/(ds.max()-ds.min())
vn=(Vcross-Vcross.min())/(Vcross.max()-Vcross.min())
U=np.minimum(dn,1-vn)
dstar=int(d[np.argmax(U)])

out=dict(surrogates=results,
         pareto_strict=front_strict, pareto_interval=front_interval,
         maximin_dstar=dstar, U=list(map(float,U)),
         steering_peak_d=int(d[np.argmax(ds)]), steering_peak_ds=float(ds.max()))
json.dump(out, open(os.path.join(DATA, 'ml_pareto.json'),'w'), indent=2)
print("SURROGATE (leave-one-geometry-out on c(d)=E/V):")
for k,v in results.items(): print(f"  {k:14s} R2={v['R2']:+.4f}  MAE={v['MAE_pct']:.2f}%  max_err={v['max_err_pct']:.2f}%")
print(f"\nPareto (strict dominance):   d = {front_strict}")
print(f"Pareto (interval dominance): d = {front_interval}   <- accounts for mesh uncertainty")
print(f"steering peaks at d={out['steering_peak_d']} (Δs={out['steering_peak_ds']:.3f} mm)")
print(f"maximin balanced optimum: d* = {dstar} mm")
