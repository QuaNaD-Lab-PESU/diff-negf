import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from negf_numpy import (negf_transmission, analytic_barrier_T,
                        build_potential_single_barrier, HBAR2_OVER_2M0)

plt.rcParams.update({
    "font.family":"serif", "font.size":11,
    "axes.edgecolor":"#33475b","axes.labelcolor":"#1b2a41",
    "text.color":"#1b2a41","xtick.color":"#33475b","ytick.color":"#33475b",
    "axes.linewidth":0.9,"figure.dpi":150})
NAVY="#1F3A5F"; RUST="#B5482A"; STEEL="#5B7DA6"; GREY="#8a8a8a"

m_r=0.067

# ---------- Fig 1: single barrier ----------
a=0.05; V0=0.3; L=3.0
n_b=int(round(L/a)); n_lead=int(round(6.0/a))
U=np.concatenate([np.zeros(n_lead),np.full(n_b,V0),np.zeros(n_lead)])
x=np.arange(len(U))*a
E=np.linspace(0.02,0.6,400)
Tn=negf_transmission(E,U,a,m_r); Ta=analytic_barrier_T(E,V0,n_b*a,m_r)
fig,ax=plt.subplots(1,2,figsize=(9.2,3.4))
ax[0].plot(x,U,color=NAVY,lw=2); ax[0].fill_between(x,U,color=NAVY,alpha=0.12)
ax[0].set_xlabel("position  x  (nm)"); ax[0].set_ylabel("potential  U(x)  (eV)")
ax[0].set_title("(a)  the device: a 3 nm barrier",fontsize=11,loc="left")
ax[0].set_ylim(-0.02,0.36)
ax[1].plot(E,Ta,color=GREY,lw=4,alpha=0.55,label="analytic (exact)")
ax[1].plot(E,Tn,color=RUST,lw=1.6,label="NEGF solver")
ax[1].axvline(V0,ls=":",color=STEEL,lw=1); ax[1].text(V0+0.005,0.05,"barrier top",color=STEEL,fontsize=8,rotation=90,va="bottom")
ax[1].set_xlabel("electron energy  E  (eV)"); ax[1].set_ylabel("transmission  T(E)")
ax[1].set_title("(b)  NEGF vs. exact analytic result",fontsize=11,loc="left")
ax[1].legend(frameon=False,fontsize=9,loc="lower right"); ax[1].set_ylim(-0.02,1.02)
plt.tight_layout(); plt.savefig("figs/fig_barrier.png",bbox_inches="tight"); plt.close()

# ---------- Fig 2: convergence ----------
E2=np.linspace(0.02,0.6,250); avals=[0.2,0.1,0.05,0.025,0.0125]; errs=[]
for aa in avals:
    nb=int(round(L/aa)); nl=int(round(6.0/aa))
    Uu=np.concatenate([np.zeros(nl),np.full(nb,V0),np.zeros(nl)])
    Tnn=negf_transmission(E2,Uu,aa,m_r); Taa=analytic_barrier_T(E2,V0,nb*aa,m_r)
    errs.append(np.abs(Tnn-Taa).max())
errs=np.array(errs); avals=np.array(avals)
fig,ax=plt.subplots(figsize=(5.0,3.6))
ax.loglog(avals,errs,'o-',color=NAVY,lw=1.8,ms=7,label="max |T$_{NEGF}$ - T$_{exact}$|")
ref=errs[0]*(avals/avals[0])**2
ax.loglog(avals,ref,'--',color=RUST,lw=1.4,label="slope 2  (O($a^2$))")
ax.set_xlabel("grid spacing  a  (nm)"); ax.set_ylabel("maximum transmission error")
ax.set_title("Stage 1: the solver converges to the exact answer",fontsize=10.5,loc="left")
ax.legend(frameon=False,fontsize=9); ax.grid(True,which="both",ls=":",alpha=0.4)
plt.tight_layout(); plt.savefig("figs/fig_convergence.png",bbox_inches="tight"); plt.close()

# ---------- Fig 3: double-barrier RTD ----------
a=0.05
def seg(nm): return int(round(nm/a))
U=np.concatenate([np.zeros(seg(4)),np.full(seg(1.0),0.3),np.zeros(seg(4.0)),
                  np.full(seg(1.0),0.3),np.zeros(seg(4))])
x=np.arange(len(U))*a
E=np.linspace(0.005,0.4,900)
T=negf_transmission(E,U,a,m_r)
fig,ax=plt.subplots(figsize=(6.4,3.6))
ax.semilogy(E,T,color=NAVY,lw=1.6)
ax.set_xlabel("electron energy  E  (eV)"); ax.set_ylabel("transmission  T(E)  (log)")
ax.set_title("A double-barrier device: sharp resonant tunnelling peaks",fontsize=10.5,loc="left")
ax.set_ylim(1e-6,2)
axin=ax.inset_axes([0.60,0.13,0.36,0.34])
axin.plot(x,U,color=NAVY,lw=1.4); axin.fill_between(x,U,color=NAVY,alpha=0.12)
axin.set_xticks([]); axin.set_yticks([]); axin.set_title("U(x)",fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig_rtd.png",bbox_inches="tight"); plt.close()

# ---------- Fig 4: gradient check (autodiff vs finite diff) ----------
import torch
from negf_torch import negf_T
torch.set_default_dtype(torch.float64)
a=0.1; N=121; x=np.arange(N)*a
U_np=(0.25*np.exp(-((x-4.0)**2)/(2*0.8**2))+0.20*np.exp(-((x-8.0)**2)/(2*0.8**2)))
E=np.linspace(0.02,0.5,60); E_t=torch.tensor(E)
U_t=torch.tensor(U_np,requires_grad=True)
T_t=negf_T(U_t,E_t,a,m_r); T_t.sum().backward()
g_auto=U_t.grad.numpy().copy()
eps=1e-6; g_fd=np.zeros(N)
for j in range(N):
    Up=U_np.copy();Up[j]+=eps; Um=U_np.copy();Um[j]-=eps
    g_fd[j]=(negf_transmission(E,Up,a,m_r).sum()-negf_transmission(E,Um,a,m_r).sum())/(2*eps)
fig,ax=plt.subplots(1,2,figsize=(9.2,3.4))
ax[0].plot(x,U_np,color=NAVY,lw=2); ax[0].fill_between(x,U_np,color=NAVY,alpha=0.12)
ax[0].set_xlabel("position x (nm)"); ax[0].set_ylabel("U(x) (eV)")
ax[0].set_title("(a)  a two-hump test potential",fontsize=11,loc="left")
ax[1].plot(x,g_fd,color=GREY,lw=4,alpha=0.55,label="finite differences")
ax[1].plot(x,g_auto,color=RUST,lw=1.5,label="autodiff (free)")
ax[1].set_xlabel("position x (nm)"); ax[1].set_ylabel(r"sensitivity  $\partial(\Sigma_E T)/\partial U(x)$")
ax[1].set_title("(b)  gradients agree to 1e-8",fontsize=11,loc="left")
ax[1].legend(frameon=False,fontsize=9)
plt.tight_layout(); plt.savefig("figs/fig_grad.png",bbox_inches="tight"); plt.close()

# ---------- Fig 5: FNO ----------
d=np.load("fno_pred.npz"); Eg=d["E"]; Xte=d["Xte"]; Tte=d["Tte"]; pred=d["pred"]
err=np.abs(pred-Tte); worst=np.argsort(err.max(1)); 
sh=[worst[len(worst)//2], worst[-3]]  # a typical and a hard one
fig,ax=plt.subplots(1,2,figsize=(9.2,3.4))
for k,idx in enumerate(sh):
    lab_t="NEGF (truth)" if k==0 else None; lab_p="FNO surrogate" if k==0 else None
    ax[0].plot(Eg,Tte[idx],color=NAVY,lw=3,alpha=0.5,label=lab_t)
    ax[0].plot(Eg,pred[idx],color=RUST,lw=1.4,ls="--",label=lab_p)
ax[0].set_xlabel("energy E (eV)"); ax[0].set_ylabel("transmission T(E)")
ax[0].set_title("(a)  surrogate vs. truth (two test devices)",fontsize=11,loc="left")
ax[0].legend(frameon=False,fontsize=9)
ax[1].scatter(Tte.ravel(),pred.ravel(),s=3,color=STEEL,alpha=0.25)
ax[1].plot([0,1],[0,1],color=NAVY,lw=1.2)
ax[1].set_xlabel("true T"); ax[1].set_ylabel("predicted T")
ax[1].set_title("(b)  parity, held-out set (rel. L2 $\\approx$ 3.9%)",fontsize=11,loc="left")
ax[1].set_xlim(0,1); ax[1].set_ylim(0,1)
plt.tight_layout(); plt.savefig("figs/fig_fno.png",bbox_inches="tight"); plt.close()

# ---------- Fig 6: inverse design ----------
r=np.load(os.path.join(_DATA, "inverse_result.npz"))
xI,EI,UI,Ttar,Tfin=r["x"],r["E"],r["U"],r["T_target"],r["T_final"]
fig,ax=plt.subplots(1,2,figsize=(9.2,3.4))
ax[0].plot(EI,Ttar,color=GREY,lw=4,alpha=0.55,label="target T(E)")
ax[0].plot(EI,Tfin,color=RUST,lw=1.6,label="achieved (designed device)")
ax[0].set_xlabel("energy E (eV)"); ax[0].set_ylabel("transmission T(E)")
ax[0].set_title("(a)  we asked for a peak at 0.16 eV ...",fontsize=11,loc="left")
ax[0].legend(frameon=False,fontsize=9)
ax[1].plot(xI,UI,color=NAVY,lw=2); ax[1].fill_between(xI,UI,color=NAVY,alpha=0.12)
ax[1].set_xlabel("position x (nm)"); ax[1].set_ylabel("U(x) (eV)")
ax[1].set_title("(b)  ... and it invented a double barrier",fontsize=11,loc="left")
plt.tight_layout(); plt.savefig("figs/fig_inverse.png",bbox_inches="tight"); plt.close()

print("figures written:")
import os
for f in sorted(os.listdir("figs")): print("  figs/"+f, os.path.getsize("figs/"+f)//1024,"KB")
