import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np, torch
from negf_torch import negf_T
from negf_numpy import negf_transmission
torch.set_default_dtype(torch.float64); torch.manual_seed(0); np.random.seed(0)

m_r, a = 0.067, 0.1
N = 81
x = np.arange(N) * a
E = np.linspace(0.02, 0.45, 90); E_t = torch.tensor(E)
E0, gamma, A = 0.16, 0.014, 0.97
T_target = A / (1.0 + ((E - E0)/gamma)**2) + 0.01
T_target_t = torch.tensor(T_target)

Umax = 0.45
theta = torch.full((N,), -1.5, requires_grad=True)
edge = 6
mask = torch.ones(N); mask[:edge]=0; mask[-edge:]=0
opt = torch.optim.Adam([theta], lr=0.04)
lam = 2e-3
for it in range(750):
    opt.zero_grad()
    U = Umax*torch.sigmoid(theta)*mask
    T = negf_T(U, E_t, a, m_r)
    dl = torch.mean((T-T_target_t)**2)
    sm = torch.mean((U[1:]-U[:-1])**2)
    (dl + lam*sm).backward(); opt.step()
    if it%150==0 or it==749:
        print(f"iter {it:4d} data={dl.item():.3e} smooth={sm.item():.3e}", flush=True)
U_final = (Umax*torch.sigmoid(theta)*mask).detach().numpy()
T_tf = negf_T(torch.tensor(U_final), E_t, a, m_r).detach().numpy()
T_cl = negf_transmission(E, U_final, a, m_r)
print("CONFIRM max|T_diff-T_classical| =", f"{np.abs(T_tf-T_cl).max():.3e}", flush=True)
pk=np.argmax(T_cl)
print("peak E=%.3f T=%.3f (target E0=%.3f A=%.2f)"%(E[pk],T_cl[pk],E0,A), flush=True)
nb=int(np.sum((U_final[1:-1]>U_final[:-2])&(U_final[1:-1]>U_final[2:])))
print("barriers(local maxima in U):", nb, flush=True)
np.savez(os.path.join(_DATA, "inverse_result.npz"), x=x, E=E, U=U_final, T_target=T_target, T_final=T_cl)
print("DONE", flush=True)
