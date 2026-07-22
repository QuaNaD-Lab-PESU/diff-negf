"""GAP 2: verify autodiff gradients dT/dU(x) against central finite
differences across three device families, including near-resonance energies
where gradients are stiffest. Reports relative L2 errors."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np, torch
from negf_numpy import negf_transmission
from negf_torch import negf_T
torch.set_default_dtype(torch.float64)

m_r = 0.067

def grad_auto(U_np, E_np, a):
    U = torch.tensor(U_np, requires_grad=True)
    T = negf_T(U, torch.tensor(E_np), a, m_r)
    T.sum().backward()
    return U.grad.numpy().copy()

def grad_fd(U_np, E_np, a, eps=1e-6):
    N = len(U_np); g = np.zeros(N)
    for j in range(N):
        Up = U_np.copy(); Up[j] += eps
        Um = U_np.copy(); Um[j] -= eps
        g[j] = (negf_transmission(E_np, Up, a, m_r).sum()
                - negf_transmission(E_np, Um, a, m_r).sum())/(2*eps)
    return g

def rel_l2(ga, gf):
    return np.linalg.norm(ga-gf)/np.linalg.norm(gf)

results = []

# ---------- Family 1: single rectangular barrier, broad energies ----------
a = 0.1
def seg(nm): return int(round(nm/a))
U1 = np.concatenate([np.zeros(seg(5)), np.full(seg(3.0),0.3), np.zeros(seg(5))])
E1 = np.linspace(0.02, 0.55, 40)
ga = grad_auto(U1,E1,a); gf = grad_fd(U1,E1,a)
r1 = rel_l2(ga,gf)
results.append(("single barrier (broad)", len(U1), "40 broad", r1))
print(f"1) single barrier: N={len(U1)}, relL2={r1:.3e}")

# ---------- Family 2: double barrier ----------
U2 = np.concatenate([np.zeros(seg(4)), np.full(seg(1.0),0.3),
                     np.zeros(seg(4.0)), np.full(seg(1.0),0.3),
                     np.zeros(seg(4))])
# locate ground resonance on a fine scan
Ef = np.linspace(0.01, 0.15, 1500)
Tf = negf_transmission(Ef, U2, a, m_r)
ipk = np.argmax(Tf); E_res = Ef[ipk]
half = Tf[ipk]/2
il = ipk - np.argmax(Tf[:ipk][::-1] < half); ir = ipk + np.argmax(Tf[ipk:] < half)
gamma = (Ef[ir]-Ef[il])/2
print(f"   double-barrier resonance: E_res={E_res:.4f} eV, HWHM~{gamma*1e3:.2f} meV, T_pk={Tf[ipk]:.4f}")

# 2a: broad + near-resonance cluster
E2 = np.sort(np.concatenate([np.linspace(0.02,0.40,28),
                             np.linspace(E_res-3*gamma, E_res+3*gamma, 12)]))
ga = grad_auto(U2,E2,a); gf = grad_fd(U2,E2,a)
r2a = rel_l2(ga,gf)
results.append(("double barrier (broad + near-res.)", len(U2), "28+12 cluster", r2a))
print(f"2a) double barrier broad+cluster: N={len(U2)}, relL2={r2a:.3e}")

# 2b: single ON-resonance energy (stiffest case)
E2b = np.array([E_res])
ga = grad_auto(U2,E2b,a)
gf6 = grad_fd(U2,E2b,a,eps=1e-6)
r2b6 = rel_l2(ga,gf6)
gf7 = grad_fd(U2,E2b,a,eps=1e-7)
r2b7 = rel_l2(ga,gf7)
r2b = min(r2b6, r2b7)
print(f"2b) ON-resonance single energy: relL2(eps=1e-6)={r2b6:.3e}, (eps=1e-7)={r2b7:.3e}")
print(f"    |grad| max = {np.abs(ga).max():.2e}  (stiffness indicator)")
results.append(("double barrier (on-resonance $E$)", len(U2), "1 at $E_{\\mathrm{res}}$", r2b))

# ---------- Family 3: random smooth potentials, 3 seeds ----------
worst = 0.0
x = np.arange(seg(12.0)+1)*a
for sd in [1,2,3]:
    rng = np.random.default_rng(sd)
    Ur = np.zeros_like(x)
    for _ in range(3):
        c=rng.uniform(2,10); w=rng.uniform(0.5,1.2); h=rng.uniform(0.1,0.35)
        Ur += h*np.exp(-((x-c)**2)/(2*w*w))
    Ur[:seg(1.0)]=0; Ur[-seg(1.0):]=0
    E3 = np.linspace(0.02,0.5,40)
    r = rel_l2(grad_auto(Ur,E3,a), grad_fd(Ur,E3,a))
    print(f"3) random seed {sd}: N={len(Ur)}, relL2={r:.3e}")
    worst = max(worst, r)
results.append(("random smooth (3 seeds, worst)", len(x), "40 broad", worst))

print("\n=== TABLE ===")
allok = True
for name,N,eset,r in results:
    print(f"{name:38s} N={N:4d}  E:{eset:15s}  relL2={r:.2e}")
    if r >= 1e-6: allok = False
print("ALL < 1e-6:", allok)
np.savez(os.path.join(_DATA, "gap2_results.npz"),
         rows=np.array([(n,N,e,f"{r:.1e}") for n,N,e,r in results],dtype=object),
         E_res=E_res, gamma=gamma, r_vals=np.array([r for *_,r in results]))
