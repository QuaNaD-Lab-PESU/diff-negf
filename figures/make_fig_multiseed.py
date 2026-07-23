"""Multi-seed figure: pass-band successes vs. the infeasible stop-band target,
and the superlattice discovered for the pass-band specification.
Reads the checkpointed sweep state written by inverse_design/multiseed_sweep.py."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
_FIG  = os.path.join(_HERE, '..', 'figs')
os.makedirs(_FIG, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":10,
    "axes.edgecolor":"#33475b","axes.labelcolor":"#1b2a41","text.color":"#1b2a41",
    "xtick.color":"#33475b","ytick.color":"#33475b","axes.linewidth":0.9,"figure.dpi":150})
NAVY="#1F3A5F"; RUST="#B5482A"; STEEL="#5B7DA6"; GREY="#9aa7b4"
E=np.linspace(0.02,0.45,90)
def t_pass(E): return 0.95/(1.0+((E-0.21)/0.030)**8)+0.02
def t_stop(E): return 0.85-0.83*np.exp(-((E-0.20)**2)/(2*0.015**2))
z=np.load(os.path.join(_DATA, "multiseed_sweep_state.npz"),allow_pickle=True); done=dict(z["done"].item())
# best seeds
bp=min(range(5),key=lambda s: done[f"passband_{s}"]["mse"])
bs=min(range(5),key=lambda s: done[f"stopband_{s}"]["mse"])
Tp,Up=done[f"passband_{bp}"]["T"],done[f"passband_{bp}"]["U"]
Ts=done[f"stopband_{bs}"]["T"]
x=np.arange(81)*0.1
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
ax[0].plot(E,t_pass(E),color=GREY,lw=3.2,alpha=0.55,label="pass-band target")
ax[0].plot(E,Tp,color=RUST,lw=1.5,label="achieved (5/5 seeds)")
ax[0].plot(E,t_stop(E),color=GREY,lw=1.6,ls=":",label="stop-band target")
ax[0].plot(E,Ts,color=STEEL,lw=1.4,ls="--",label="best attempt (0/5)")
ax[0].set_xlabel("energy $E$ (eV)"); ax[0].set_ylabel("transmission $T(E)$")
ax[0].set_title("(a) feasible vs. infeasible targets",fontsize=9.5,loc="left")
ax[0].legend(frameon=False,fontsize=6.8,loc="center right"); ax[0].set_ylim(-0.03,1.03)
ax[1].plot(x,Up,color=NAVY,lw=1.8); ax[1].fill_between(x,Up,color=NAVY,alpha=0.12)
ax[1].set_xlabel("position $x$ (nm)"); ax[1].set_ylabel("$U(x)$ (eV)")
ax[1].set_title("(b) pass-band device: a discovered superlattice",fontsize=9.5,loc="left")
plt.tight_layout(); plt.savefig(os.path.join(_FIG, "fig_sweep.png"),bbox_inches="tight"); plt.close()
# original flat-init resonance MSE for the table
r=np.load(os.path.join(_DATA, "inverse_result.npz"))
mse0=float(np.mean((r["T_final"]-r["T_target"])**2))
print(f"flat-init resonance MSE={mse0:.3e}; passband best seed={bp}, stopband best={bs}")
print("figs/fig_sweep.png written")
