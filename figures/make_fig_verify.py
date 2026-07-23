import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
_FIG  = os.path.join(_HERE, '..', 'figs')
os.makedirs(_FIG, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family":"serif","font.size":10,
    "axes.edgecolor":"#33475b","axes.labelcolor":"#1b2a41","text.color":"#1b2a41",
    "xtick.color":"#33475b","ytick.color":"#33475b","axes.linewidth":0.9,"figure.dpi":150})
NAVY="#1F3A5F"; RUST="#B5482A"; GREY="#9aa7b4"
d=np.load(os.path.join(_DATA, "verification_results.npz"))
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
# (a) Breit-Wigner
ax[0].plot(d["E1"],d["T_bw_sym"],color=GREY,lw=3.5,alpha=0.55,label="Breit–Wigner (exact)")
ax[0].plot(d["E1"],d["T_negf_sym"],color=NAVY,lw=1.3,label=r"NEGF, $\Gamma_L{=}\Gamma_R$")
ax[0].plot(d["E1"],d["T_bw_asym"],color=GREY,lw=3.5,alpha=0.55)
ax[0].plot(d["E1"],d["T_negf_asym"],color=RUST,lw=1.3,label=r"NEGF, $\Gamma_L{\neq}\Gamma_R$")
ax[0].set_xlabel(r"$E-\varepsilon$ (units of $\Gamma_L$)")
ax[0].set_ylabel("transmission $T(E)$")
ax[0].set_title("(a) single resonant level",fontsize=9.5,loc="left")
ax[0].legend(frameon=False,fontsize=7.2,loc="upper right")
ax[0].set_ylim(-0.02,1.05)
# (b) NEGF vs TMM double barrier
ax[1].semilogy(d["E2"],d["T_tmm_db"],color=GREY,lw=3.5,alpha=0.55,label="transfer matrix")
ax[1].semilogy(d["E2"],d["T_negf_db"],color=NAVY,lw=1.2,label="NEGF")
ax[1].set_xlabel("$E$ (eV)")
ax[1].set_ylabel("$T(E)$")
ax[1].set_title("(b) double barrier: NEGF vs. independent method",fontsize=9.5,loc="left")
ax[1].legend(frameon=False,fontsize=7.2,loc="lower right")
ax[1].set_ylim(1e-6,2)
plt.tight_layout(); plt.savefig(os.path.join(_FIG, "fig_verify.png"),bbox_inches="tight"); plt.close()
print("figs/fig_verify.png written")
