"""FD vs independent FEM: full E_n(s) profile and peak-location comparison (rev#12)."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'data')
FIGS = os.path.join(_ROOT, 'figures')
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

import numpy as np, sys, json, time
from ehd_config import *
from ehd_solver3d import Solver3D
from skfem import MeshTet, Basis, ElementTetP1, condense, solve as fsolve
from skfem.models.poisson import laplace
mm=1e-3; V=4000.; OFF=0.6*mm  # matched extraction offset for both methods (>= FEM cell)

def fem_profile(cond,drop,box,h,n_s=601):
    x=np.arange(box[0][0],box[0][1]+h/2,h);y=np.arange(box[1][0],box[1][1]+h/2,h);z=np.arange(box[2][0],box[2][1]+h/2,h)
    m=MeshTet.init_tensor(x,y,z); e=ElementTetP1(); basis=Basis(m,e); A=laplace.assemble(basis); P=m.p
    u=np.zeros(basis.N); dm=np.zeros(basis.N,bool)
    for c in cond:
        ins=c.f(P[0],P[1],P[2])<1e-9; u[ins]=c.potential; dm|=ins
    onb=((np.abs(P[0]-x[0])<1e-9)|(np.abs(P[0]-x[-1])<1e-9)|(np.abs(P[1]-y[-1])<1e-9)|(np.abs(P[2]-z[0])<1e-9)|(np.abs(P[2]-z[-1])<1e-9))
    onb&=~dm; u[onb]=0.; dm|=onb
    u=fsolve(*condense(A,np.zeros(basis.N),x=u,D=np.where(dm)[0]))
    itp=basis.interpolator(u)
    s,xs,zs,nx,nz=drop.surface_points(n_s)
    p1=itp(np.vstack([xs+OFF*nx,np.zeros_like(xs),zs+OFF*nz]))
    p2=itp(np.vstack([xs+2*OFF*nx,np.zeros_like(xs),zs+2*OFF*nz]))
    En=-(4*(p1-drop.potential)-(p2-drop.potential))/(2*OFF)
    return s,xs,En,basis.N

def fd_profile(cond,drop,n_s=601):
    s3=Solver3D((-14*mm,14*mm),(0.,14*mm),(-4*mm,17*mm),0.20*mm,cond,ybc='neumann',robin=True); s3.solve()
    s,xs,zs,nx,nz=drop.surface_points(n_s); y=np.zeros_like(xs)
    p1=s3.interp(xs+OFF*nx,y,zs+OFF*nz); p2=s3.interp(xs+2*OFF*nx,y,zs+2*OFF*nz)
    En=-(4*(p1-drop.potential)-(p2-drop.potential))/(2*OFF)
    return s,xs,En

case=sys.argv[1]
if case=='d6':
    c1,d1=fs_pin(V,d=6*mm); box=((-14*mm,14*mm),(0.,14*mm),(-4*mm,17*mm))
    sf,xf,Ef,N=fem_profile(c1,d1,box,0.5*mm)
    c2,d2=fs_pin(V,d=6*mm); sd,xd,Ed=fd_profile(c2,d2)
    sap=sd[np.argmin(np.abs(xd))]
    pf,_=Solver3D._parabolic_peak(sf,Ef,srange=(sap-3.2e-3,sap+0.3e-3))
    pd,_=Solver3D._parabolic_peak(sd,Ed,srange=(sap-3.2e-3,sap+0.3e-3))
    out=dict(case='d6',N_fem=int(N),ds_fem_mm=float(abs(pf-sap)/mm),ds_fd_mm=float(abs(pd-sap)/mm),
             speak_fem_mm=float(pf/mm),speak_fd_mm=float(pd/mm),
             profiles=dict(s=(sd/mm).tolist(),fd=(Ed/np.nanmax(Ed)).tolist(),fem=(Ef/np.nanmax(Ef)).tolist()))
else:
    c1,d1=fs_double_pin(Vp=20e3,x_off=6*mm); box=((-16*mm,16*mm),(0.,14*mm),(-4*mm,17*mm))
    sf,xf,Ef,N=fem_profile(c1,d1,box,0.5*mm)
    c2,d2=fs_double_pin(Vp=20e3,x_off=6*mm); sd,xd,Ed=fd_profile(c2,d2)
    sm=sd[np.argmin(np.abs(xd))]
    def two(s,E):
        l=Solver3D._parabolic_peak(s[s<sm],np.abs(E[s<sm]))[0]; r=Solver3D._parabolic_peak(s[s>=sm],np.abs(E[s>=sm]))[0]; return l/mm,r/mm
    lf,rf=two(sf,Ef); ld,rd=two(sd,Ed)
    out=dict(case='double',N_fem=int(N),peaks_fem_mm=[float(lf),float(rf)],peaks_fd_mm=[float(ld),float(rd)],
             profiles=dict(s=(sd/mm).tolist(),fd=(np.abs(Ed)/np.nanmax(np.abs(Ed))).tolist(),fem=(np.abs(Ef)/np.nanmax(np.abs(Ef))).tolist()))
json.dump(out,open(fos.path.join(DATA, 'femcmp_{case}.json'),'w'),indent=2)
print({k:v for k,v in out.items() if k!='profiles'},flush=True)
