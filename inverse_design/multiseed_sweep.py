"""Multi-seed inverse-design study. Three targets of increasing
difficulty x 5 random initialisations, all confirmed by the classical solver.
Checkpointed: safe to invoke repeatedly until all runs complete."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np, torch, time, os
from negf_torch import negf_T
from negf_numpy import negf_transmission
torch.set_default_dtype(torch.float64)

m_r, a, N, Umax, edge = 0.067, 0.1, 81, 0.45, 6
E = np.linspace(0.02, 0.45, 90); E_t = torch.tensor(E)
mask = torch.ones(N); mask[:edge]=0; mask[-edge:]=0

# ---------- targets ----------
def t_res(E):   # (i) Lorentzian resonance at 0.16 eV (paper's original target)
    return 0.97/(1.0+((E-0.16)/0.014)**2) + 0.01
def t_pass(E):  # (ii) flat-top pass-band centred 0.21 eV, half-width ~30 meV
    return 0.95/(1.0+((E-0.21)/0.030)**8) + 0.02
def t_stop(E):  # (iii) notch (stop-band) at 0.20 eV in a transmitting background
    return 0.85 - 0.83*np.exp(-((E-0.20)**2)/(2*0.015**2))
TARGETS = {"resonance": t_res, "passband": t_pass, "stopband": t_stop}
SEEDS = [0,1,2,3,4]

STATE = os.path.join(_DATA, "multiseed_sweep_state.npz")
done = {}
if os.path.exists(STATE):
    z = np.load(STATE, allow_pickle=True)
    done = dict(z["done"].item())
print(f"resume: {len(done)}/15 runs complete")

t0 = time.time(); BUDGET = 225.0
for tname, tf in TARGETS.items():
    Tt = torch.tensor(tf(E))
    for sd in SEEDS:
        key = f"{tname}_{sd}"
        if key in done: continue
        if time.time()-t0 > BUDGET:
            print("time budget reached; checkpointing"); break
        torch.manual_seed(sd)
        theta = (torch.randn(N)*0.3 - 1.5).requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=0.04)
        for it in range(750):
            opt.zero_grad()
            U = Umax*torch.sigmoid(theta)*mask
            T = negf_T(U, E_t, a, m_r)
            loss = torch.mean((T-Tt)**2) + 2e-3*torch.mean((U[1:]-U[:-1])**2)
            loss.backward(); opt.step()
        U_f = (Umax*torch.sigmoid(theta)*mask).detach().numpy()
        T_cl = negf_transmission(E, U_f, a, m_r)     # classical confirmation
        mse = float(np.mean((T_cl - tf(E))**2))
        done[key] = {"U": U_f, "T": T_cl, "mse": mse}
        print(f"  {key}: classical MSE={mse:.4e}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        continue
    break
np.savez(STATE, done=np.array(done, dtype=object))
print(f"checkpoint saved: {len(done)}/15 complete")
