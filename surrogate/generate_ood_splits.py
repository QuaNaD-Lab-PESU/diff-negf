"""GAP 3 stage 1: generate three designated OOD splits.
In-family generator (fno_data.npz): 1-3 bumps, w 0.4-1.6 nm, h 0.08-0.40 eV.
OOD-tall:   h 0.40-0.50 eV (beyond training heights)
OOD-narrow: w 0.20-0.40 nm (sharper than training)
OOD-4bump:  4 bumps (more structure than training)"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np
from negf_numpy import negf_transmission
rng = np.random.default_rng(42)
d = np.load(os.path.join(_DATA, "fno_data.npz"))
a, m_r = float(d["a"]), float(d["m_r"])
xg, E = d["x"], d["E"]; N = len(xg); edge = 6

def make(nb_lo, nb_hi, w_lo, w_hi, h_lo, h_hi, M):
    Us = np.zeros((M, N))
    for i in range(M):
        U = np.zeros(N)
        for _ in range(rng.integers(nb_lo, nb_hi+1)):
            c = rng.uniform(xg[edge], xg[-edge])
            w = rng.uniform(w_lo, w_hi); h = rng.uniform(h_lo, h_hi)
            U += h*np.exp(-((xg-c)**2)/(2*w*w))
        U[:edge] = 0; U[-edge:] = 0
        Us[i] = np.clip(U, 0, 0.55)
    Ts = np.stack([negf_transmission(E, u, a, m_r) for u in Us])
    return Us, Ts

M = 150
U_tall, T_tall     = make(1,3, 0.4,1.6, 0.40,0.50, M)
U_narrow, T_narrow = make(1,3, 0.20,0.40, 0.08,0.40, M)
U_4b, T_4b         = make(4,4, 0.4,1.6, 0.08,0.40, M)
np.savez(os.path.join(_DATA, "gap3_ood.npz"), U_tall=U_tall,T_tall=T_tall,
         U_narrow=U_narrow,T_narrow=T_narrow,U_4b=U_4b,T_4b=T_4b)
for nm,(u,t) in [("tall",(U_tall,T_tall)),("narrow",(U_narrow,T_narrow)),("4bump",(U_4b,T_4b))]:
    print(f"{nm}: U max={u.max():.3f} eV, T range [{t.min():.2e},{t.max():.3f}]")
print("saved gap3_ood.npz")
