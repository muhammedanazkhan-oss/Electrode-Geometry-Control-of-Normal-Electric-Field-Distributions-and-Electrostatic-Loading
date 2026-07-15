import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
import numpy as np, sys, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ehd_config import *
from ehd_axisym import AxisymSolver
from ehd_solver3d import Solver3D
mmv=1e-3; V=4000.
M=json.load(open(os.path.join(DATA, 'ALL.json'))); E=M['electrostatics']; ET=E['ET']; ER=E['ER']
plt.rcParams.update({'font.size':11,'figure.dpi':130}); OUT = FIGS

# ---- FIG E: field-magnitude maps PP & PT (axisym) ----
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
for a,kind,ttl in [(ax[0],'PP','(a) Parallel-plate'),(ax[1],'PT','(b) Pin-plate (on-axis)')]:
    cond,drop=(fs_pp(V) if kind=='PP' else fs_pin(V,d=0.0))
    s=AxisymSolver(20*mmv,22*mmv,0.06*mmv,cond,z_min=-6*mmv,robin=True); s.solve()
    R,Z=np.meshgrid(s.r,s.z,indexing='ij')
    Er=np.gradient(-s.phi,s.r,axis=0); Ez=np.gradient(-s.phi,s.z,axis=1)
    Emag=np.sqrt(Er**2+Ez**2); Emag[s.dirichlet]=np.nan
    pc=a.pcolormesh(R/mmv,Z/mmv,np.log10(np.clip(Emag,1e3,None)),shading='auto',cmap='viridis')
    a.set_xlim(0,15); a.set_ylim(-3,15); a.set_aspect('equal')
    a.set_xlabel('Radius $r$ (mm)'); a.set_ylabel('Height $z$ (mm)'); a.set_title(ttl)
    plt.colorbar(pc,ax=a,label='$\\log_{10}|E|$ (V m$^{-1}$)')
plt.tight_layout(); plt.savefig(f'{OUT}/figE_fieldmaps.png',bbox_inches='tight'); plt.close()

# ---- FIG F: double-pin perimeter profile (two maxima) ----
cond,drop=fs_double_pin(Vp=20e3,x_off=6*mmv)
s3=Solver3D((-16*mmv,16*mmv),(0.,14*mmv),(-4*mmv,18*mmv),0.20*mmv,cond,ybc='neumann',robin=True); s3.solve()
ss,xg,zg,En=s3.great_circle_field(drop,n_s=1601,off=0.15*mmv)
fig,ax=plt.subplots(figsize=(6.5,4.4))
ax.plot(ss/mmv,np.abs(En),c='tab:red',lw=1.8)
ax.axhline(ET,ls='--',c='k',lw=1.2,label='$E_T$ (Taylor)')
dd=M['double_pin']['Vp20.0|h0.2|x6.0']
ax.plot([dd['s_left_mm'],dd['s_right_mm']],[dd['E_left'],dd['E_right']],'v',ms=10,c='crimson',label='Field maxima')
ax.set_xlabel('Arc length $s$ (mm)'); ax.set_ylabel('$|E_n|$ (V m$^{-1}$)')
ax.set_title('Double-pin ($\\pm$20 kV): two symmetric field maxima (below $E_T$ in air)')
ax.legend(fontsize=9); plt.tight_layout(); plt.savefig(f'{OUT}/figF_doublepin.png',bbox_inches='tight'); plt.close()

# ---- FIG G: surrogate leave-one-geometry-out fit c(d)=E/V ----
sw=E['sweep']; d=np.array([r['d'] for r in sw]); c=np.array([r['Emax'] for r in sw])/V
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,ConstantKernel
k=ConstantKernel(1.0,(1e-3,1e3))*Matern(length_scale=2.0,length_scale_bounds=(0.3,20),nu=2.5)
gp=GaussianProcessRegressor(kernel=k,alpha=1e-10,normalize_y=True,n_restarts_optimizer=15,random_state=0)
gp.fit(d.reshape(-1,1),c); dg=np.linspace(0,9,120); mu,sd=gp.predict(dg.reshape(-1,1),return_std=True)
fig,ax=plt.subplots(figsize=(6.3,4.4))
ax.fill_between(dg,mu-2*sd,mu+2*sd,alpha=0.25,label='GPR 95% band')
ax.plot(dg,mu,c='tab:blue',label='GPR mean'); ax.plot(d,c,'ko',ms=6,label='FD ground truth')
ax.set_xlabel('Pin offset $d$ (mm)'); ax.set_ylabel('Field coefficient $c(d)=E_{n,max}/V$ (m$^{-1}$)')
gr=M['ml_pareto']['surrogates']['GPR-Matern52']
ax.set_title(f"Surrogate (leave-one-geometry-out $R^2$={gr['R2']:.3f})"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figG_surrogate.png',bbox_inches='tight'); plt.close()

# ---- FIG H: BO vs random convergence (mean over seeds) ----
bo=json.load(open(os.path.join(DATA, 'bo_results.json')))
seeds=sorted(bo['bo'].keys())
BO=np.array([bo['bo'][s]['ybest'] for s in seeds]); RD=np.array([bo['rand'][s]['ybest'] for s in seeds])
it=np.arange(1,BO.shape[1]+1)
fig,ax=plt.subplots(figsize=(6.3,4.4))
ax.plot(it,BO.mean(0),'-o',c='tab:blue',label='Bayesian optimisation')
ax.fill_between(it,BO.mean(0)-BO.std(0),BO.mean(0)+BO.std(0),alpha=0.2,color='tab:blue')
ax.plot(it,RD.mean(0),'-s',c='tab:orange',label='Random search')
ax.fill_between(it,RD.mean(0)-RD.std(0),RD.mean(0)+RD.std(0),alpha=0.2,color='tab:orange')
ax.set_xlabel('High-fidelity solver evaluations'); ax.set_ylabel('Best steering $\\Delta s$ found (mm)')
ax.set_title(f"BO vs random ({len(seeds)} seeds; no significant speed-up in 1-D)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figH_bo.png',bbox_inches='tight'); plt.close()
print("figures E-H written")
