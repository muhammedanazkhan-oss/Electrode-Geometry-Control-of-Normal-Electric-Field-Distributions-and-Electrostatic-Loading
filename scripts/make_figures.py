"""Generate the corrected publication figures from the computed data.
Fixes: capitalised axis labels (R3.1); Fig 4 shaded zones incl. the Taylor band
(R1.5 'where is the yellow zone'); Fig 10 labelled apex electric field not stress;
consistent subscripted notation; comparison basis stated."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

import numpy as np, sys, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ehd_config import *
from ehd_axisym import AxisymSolver
from ehd_solver3d import Solver3D
mmv=1e-3; V=4000.
M=json.load(open(os.path.join(DATA, 'ALL.json')))
E=M['electrostatics']; ET=E['ET']; ER=E['ER']
plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':0.3,'figure.dpi':130})
OUT = FIGS

# ---- helper solves ----
def axi(kind,d=0.0,Vv=V,h=0.06):
    cond,drop=(fs_pp(Vv) if kind=='PP' else fs_pin(Vv,d=d*mmv))
    s=AxisymSolver(30*mmv,30*mmv,h*mmv,cond,z_min=-10*mmv,robin=True); s.solve()
    return s,drop

# ======== FIG A: perimeter normal-field profiles, PP & PT, six voltages ========
volts=[500,1000,2000,4000,6000,8000]
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for a,kind,ttl in [(ax[0],'PP','(a) Parallel-plate'),(ax[1],'PT','(b) Pin-plate (on-axis)')]:
    s,drop=axi(kind,h=0.06)
    ss,xx,zz,En=s.surface_normal_field(drop,off=0.15*mmv)
    for Vv in volts:
        a.plot(ss/mmv,(En/V*Vv),label=f'{Vv} V')
    a.axhline(ET,ls='--',c='k',lw=1.2,label='$E_T$ (Taylor)')
    a.axhline(ER,ls='-.',c='grey',lw=1.0,label='$E_R$ (Rayleigh ref.)')
    a.set_title(ttl); a.set_xlabel('Arc length $s$ (mm)'); a.set_ylabel('Normal field $E_n$ (V m$^{-1}$)')
    a.set_ylim(0,3.2e6)
ax[1].legend(fontsize=7,ncol=2)
plt.tight_layout(); plt.savefig(f'{OUT}/figA_perimeter.png',bbox_inches='tight'); plt.close()

# ======== FIG B: apex/steered field vs voltage with Taylor crossing + shaded zones ========
fig,ax=plt.subplots(figsize=(6.5,4.6))
vv=np.linspace(0,12000,50)
ppc=E['PP']['Eapex']/V; ptc=E['PT0']['Emax']/V
ax.plot(vv/1e3,ppc*vv,'-o',ms=3,label='Parallel-plate apex')
ax.plot(vv/1e3,ptc*vv,'-s',ms=3,label='Pin-plate steered peak')
ax.axhline(ET,ls='--',c='k',lw=1.2)
ax.axhline(ER,ls='-.',c='grey',lw=1.0)
# shaded zones (R1.5: make the Taylor band explicit and label both)
ax.axhspan(ET,ER,color='gold',alpha=0.25,label='Above $E_T$ (Taylor-crossing)')
ax.axhspan(ER,3.4e6,color='orange',alpha=0.18,label='Above $E_R$ (Rayleigh ref.)')
for c,vc,lab in [(ppc,E['PP']['Vcross_kV'],'PP'),(ptc,E['PT0']['Vcross_kV'],'PT')]:
    ax.plot(vc,ET,'*',ms=14,c='crimson')
ax.text(E['PP']['Vcross_kV'],ET*1.03,f"  $V$={E['PP']['Vcross_kV']:.1f} kV",fontsize=8)
ax.text(E['PT0']['Vcross_kV'],ET*0.86,f"$V$={E['PT0']['Vcross_kV']:.1f} kV",fontsize=8)
ax.set_xlabel('Applied voltage $V$ (kV)'); ax.set_ylabel('Peak normal field $E_n$ (V m$^{-1}$)')
ax.set_ylim(0,3.4e6); ax.set_xlim(0,12); ax.legend(fontsize=8,loc='upper left')
ax.set_title('Taylor-field-crossing estimate (shaded: Taylor band gold, Rayleigh-ref orange)')
plt.tight_layout(); plt.savefig(f'{OUT}/figB_crossing.png',bbox_inches='tight'); plt.close()

# ======== FIG C: parametric offset study (4 panels) ========
sw=E['sweep']; d=np.array([r['d'] for r in sw]); ds=np.array([r['ds'] for r in sw])
Eap=np.array([r['Eapex'] for r in sw]); Emx=np.array([r['Emax'] for r in sw]); Vc=np.array([r['Vcross_kV'] for r in sw])
fig,ax=plt.subplots(2,2,figsize=(10,7.5))
# (a) perimeter profiles for a few offsets (compute)
for dd in [0,3,6,9]:
    cond,drop=fs_pin(V,d=dd*mmv)
    s3=Solver3D((-14*mmv,14*mmv),(0.,12*mmv),(-4*mmv,18*mmv),0.20*mmv,cond,ybc='neumann',robin=True); s3.solve()
    ss,xg,zg,En=s3.great_circle_field(drop,n_s=1201,off=0.15*mmv)
    ax[0,0].plot(ss/mmv,En,label=f'$d$={dd} mm')
ax[0,0].axhline(ET,ls='--',c='k',lw=1); ax[0,0].set_xlabel('Arc length $s$ (mm)')
ax[0,0].set_ylabel('Normal field $E_n$ (V m$^{-1}$)'); ax[0,0].set_title('(a) Perimeter profiles ($V$=4 kV)'); ax[0,0].legend(fontsize=8)
ax[0,1].plot(d,ds,'-o',c='tab:green'); ax[0,1].set_xlabel('Pin offset $d$ (mm)')
ax[0,1].set_ylabel('Steering $\\Delta s$ (mm)'); ax[0,1].set_title('(b) Steering of high-field site')
ax[1,0].plot(d,Eap,'-o',label='Apex $E_{apex}$'); ax[1,0].plot(d,Emx,'-s',label='Steered peak $E_{n,max}$')
ax[1,0].set_xlabel('Pin offset $d$ (mm)'); ax[1,0].set_ylabel('Field at 4 kV (V m$^{-1}$)'); ax[1,0].set_title('(c) Apex & steered-peak field'); ax[1,0].legend(fontsize=9)
ax[1,1].plot(d,Vc,'-o',c='tab:red'); ax[1,1].set_xlabel('Pin offset $d$ (mm)')
ax[1,1].set_ylabel('$V_{cross}$ (kV)'); ax[1,1].set_title('(d) Taylor-field-crossing voltage')
plt.tight_layout(); plt.savefig(f'{OUT}/figC_offset.png',bbox_inches='tight'); plt.close()

# ======== FIG D: Pareto with interval dominance ========
fig,ax=plt.subplots(figsize=(6.3,4.6))
unc=0.08*Vc
ax.errorbar(Vc,ds,xerr=unc,fmt='o',ms=7,capsize=3,c='tab:blue')
for i in range(len(d)): ax.annotate(f'{int(d[i])}',(Vc[i],ds[i]),fontsize=8,xytext=(3,3),textcoords='offset points')
strict=M['ml_pareto']['pareto_strict']
sel=[i for i in range(len(d)) if int(d[i]) in strict]
ax.plot(Vc[sel],ds[sel],'-',c='crimson',lw=2,label='Pareto front (strict)')
ax.axvspan(0,0,color='gold',alpha=0.2)
ax.set_xlabel('$V_{cross}$ (kV)  — minimise'); ax.set_ylabel('Steering $\\Delta s$ (mm) — maximise')
ax.set_title('Steering–voltage trade-off (error bars: 8% mesh uncertainty)'); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figD_pareto.png',bbox_inches='tight'); plt.close()

print("figures A-D written")
