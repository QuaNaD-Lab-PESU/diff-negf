"""Surrogate study, step 3: absolute worst-case errors + regenerate paper FNO figure
from the retrained weights (so figure and numbers match)."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
_FIG  = os.path.join(_HERE, '..', 'figs')
os.makedirs(_FIG, exist_ok=True)
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
exec(open(os.path.join(_HERE, "train_fno_mlp.py")).read().split("def train")[0])  # rebuild classes/splits
fno = FNO1d(); fno.load_state_dict(torch.load(os.path.join(_DATA, "fno_weights.pt"))); fno.eval()
mlp = nn.Sequential(nn.Linear(64,493), nn.ReLU(), nn.Linear(493,493),
                    nn.ReLU(), nn.Linear(493,64))
mlp.load_state_dict(torch.load(os.path.join(_DATA, "mlp_weights.pt"))); mlp.eval()

print(f"{'split':12s} | FNO maxabs | MLP maxabs | worst-rel device ||T||")
with torch.no_grad():
    for k,(X,Tt) in splits.items():
        Pf, Pm = fno(X), mlp(X)
        mf = (Pf-Tt).abs().max().item(); mm = (Pm-Tt).abs().max().item()
        per = torch.linalg.norm(Pf-Tt,dim=1)/torch.linalg.norm(Tt,dim=1)
        iw = per.argmax().item()
        print(f"{k:12s} |   {mf:.3f}    |   {mm:.3f}    | {torch.linalg.norm(Tt[iw]).item():.3f} (mean T={Tt[iw].mean().item():.4f})")

# regenerate paper FNO figure from retrained model
plt.rcParams.update({"font.family":"serif","font.size":11,
    "axes.edgecolor":"#33475b","axes.labelcolor":"#1b2a41","text.color":"#1b2a41",
    "xtick.color":"#33475b","ytick.color":"#33475b","axes.linewidth":0.9,"figure.dpi":150})
NAVY="#1F3A5F"; RUST="#B5482A"; STEEL="#5B7DA6"
d = np.load(os.path.join(_DATA, "fno_data.npz")); Eg = d["E"]
Xte, Tte = splits["in-family"]
with torch.no_grad(): pred = fno(Xte).numpy()
Tte_n = Tte.numpy()
err = np.abs(pred-Tte_n)
order = np.argsort(err.max(1)); sh = [order[len(order)//2], order[-3]]
fig,ax = plt.subplots(1,2,figsize=(9.2,3.4))
for k,idx in enumerate(sh):
    lt = "NEGF (truth)" if k==0 else None; lp = "FNO surrogate" if k==0 else None
    ax[0].plot(Eg,Tte_n[idx],color=NAVY,lw=3,alpha=0.5,label=lt)
    ax[0].plot(Eg,pred[idx],color=RUST,lw=1.4,ls="--",label=lp)
ax[0].set_xlabel("energy E (eV)"); ax[0].set_ylabel("transmission T(E)")
ax[0].set_title("(a)  surrogate vs. truth (two test devices)",fontsize=11,loc="left")
ax[0].legend(frameon=False,fontsize=9)
ax[1].scatter(Tte_n.ravel(),pred.ravel(),s=3,color=STEEL,alpha=0.25)
ax[1].plot([0,1],[0,1],color=NAVY,lw=1.2)
ax[1].set_xlabel("true T"); ax[1].set_ylabel("predicted T")
ax[1].set_title("(b)  parity, held-out set (rel. L2 $\\approx$ 3.7%)",fontsize=11,loc="left")
ax[1].set_xlim(0,1); ax[1].set_ylim(0,1)
plt.tight_layout(); plt.savefig(os.path.join(_FIG, "fig_fno.png"),bbox_inches="tight"); plt.close()
print("figs/fig_fno.png regenerated from retrained weights")
