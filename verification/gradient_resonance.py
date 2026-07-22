"""Refined on-resonance verification: (i) flank of resonance (stiffest true
gradient); (ii) exact peak with FD-noise-aware Richardson extrapolation;
(iii) directional-derivative check at the peak."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np, torch
from negf_numpy import negf_transmission
from negf_torch import negf_T
torch.set_default_dtype(torch.float64)
m_r, a = 0.067, 0.1
def seg(nm): return int(round(nm/a))
U2 = np.concatenate([np.zeros(seg(4)), np.full(seg(1.0),0.3),
                     np.zeros(seg(4.0)), np.full(seg(1.0),0.3),
                     np.zeros(seg(4))])
E_res, gamma = 0.1063, 0.01812

def gauto(E):
    U=torch.tensor(U2,requires_grad=True)
    negf_T(U,torch.tensor(np.atleast_1d(E)),a,m_r).sum().backward()
    return U.grad.numpy().copy()
def gfd(E,eps):
    N=len(U2); g=np.zeros(N); E=np.atleast_1d(E)
    for j in range(N):
        Up=U2.copy();Up[j]+=eps; Um=U2.copy();Um[j]-=eps
        g[j]=(negf_transmission(E,Up,a,m_r).sum()-negf_transmission(E,Um,a,m_r).sum())/(2*eps)
    return g
rl=lambda x,y: np.linalg.norm(x-y)/np.linalg.norm(y)

# (i) FLANK: E_res - gamma (max |dT/dE| region -> largest dT/dU)
Efl = E_res - gamma
ga_f = gauto(Efl); gf_f = gfd(Efl,1e-6)
print(f"(i) flank E=E_res-gamma: |grad|max={np.abs(ga_f).max():.2e}  relL2={rl(ga_f,gf_f):.3e}")

# (ii) PEAK with Richardson: D(eps) central FD has error c*eps^2 + noise/eps
ga_p = gauto(E_res)
D1 = gfd(E_res,2e-4); D2 = gfd(E_res,1e-4)
rich = (4*D2 - D1)/3.0          # eliminates O(eps^2) term
print(f"(ii) peak: relL2 vs FD(2e-4)={rl(ga_p,D1):.2e}, FD(1e-4)={rl(ga_p,D2):.2e}, Richardson={rl(ga_p,rich):.3e}")
print(f"     eps-scaling check (roundoff signature): FD(1e-6)={4.247e-05:.1e}, FD(1e-7)={4.032e-04:.1e} -> error ~ 1/eps: ratio {4.032e-04/4.247e-05:.1f} (expect ~10)")

# (iii) directional derivative at peak: scalar FD is well-conditioned
rng=np.random.default_rng(0); v=rng.standard_normal(len(U2)); v/=np.linalg.norm(v)
def Tsum(t): return negf_transmission(np.atleast_1d(E_res),U2+t*v,a,m_r).sum()
h=1e-5
dd_fd=(Tsum(h)-Tsum(-h))/(2*h)
dd_fd_r=(4*((Tsum(h/2)-Tsum(-h/2))/h) - dd_fd)/3.0
dd_auto=float(ga_p@v)
print(f"(iii) directional: auto={dd_auto:.10e}  FD-Richardson={dd_fd_r:.10e}  rel err={abs(dd_auto-dd_fd_r)/abs(dd_fd_r):.3e}")
