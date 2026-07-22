"""GAP 1: (a) Breit-Wigner closed-form benchmark of the NEGF recipe on a
single resonant level; (b) algorithmically independent transfer-matrix
cross-check of the lattice NEGF solver on double-barrier and random-potential
devices."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np
from negf_numpy import negf_transmission, HBAR2_OVER_2M0

# ---------- (a) Breit-Wigner single-level benchmark ----------
def single_level_T_negf(E, eps, GL, GR):
    """The 4-step NEGF recipe on a 1-site device, wide-band self-energies."""
    T = np.zeros_like(E)
    for i, e in enumerate(E):
        SigL, SigR = -1j*GL/2, -1j*GR/2
        G = 1.0/(e - eps - SigL - SigR)
        T[i] = (-2*SigL.imag) * abs(G)**2 * (-2*SigR.imag)
    return T

def breit_wigner(E, eps, GL, GR):
    return GL*GR/((E-eps)**2 + ((GL+GR)/2)**2)

E1 = np.linspace(-4, 4, 801)
errs = []
for GL, GR in [(1.0,1.0),(1.0,0.3),(0.5,2.0)]:
    errs.append(np.abs(single_level_T_negf(E1,0.0,GL,GR)
                       - breit_wigner(E1,0.0,GL,GR)).max())
bw_err = max(errs)
print(f"(a) Breit-Wigner: max |T_NEGF - T_BW| over 3 (GL,GR) cases = {bw_err:.3e}")

# ---------- (b) transfer-matrix method on the identical lattice ----------
def tmm_transmission(E_grid, U, a, m_r):
    """Transmission by 2x2 transfer-matrix products on the same tight-binding
    chain: psi_{n+1} = [(2t0+U_n-E)/t0] psi_n - psi_{n-1}. Lead mode matching
    at both ends. Algorithmically independent of the Green's-function route."""
    t0 = HBAR2_OVER_2M0/(m_r*a*a)
    N = len(U)
    T = np.zeros_like(E_grid, dtype=float)
    for i, E in enumerate(E_grid):
        ca = 1.0 - E/(2*t0)
        if abs(ca) >= 1.0:          # outside lead band
            T[i] = 0.0; continue
        ka = np.arccos(ca)
        # accumulate M over one lead site (U=0) then the N device sites
        M = np.eye(2, dtype=complex)
        for Un in np.concatenate([[0.0], U]):
            Mn = np.array([[(2*t0+Un-E)/t0, -1.0],[1.0, 0.0]], dtype=complex)
            M = Mn @ M
        # v_{-1} = (psi_0, psi_{-1}); left lead: psi_n = e^{ikna}+r e^{-ikna}
        # right lead: psi_n = t e^{ikna}; after M: v_N = (psi_{N+1}, psi_N)
        # unknowns (r, t):  M @ [1+r, e^{-ika}+r e^{ika}] = [t e^{ik(N+1)a}, t e^{ikNa}]
        A = np.array([
            [M[0,0] + M[0,1]*np.exp(1j*ka), -np.exp(1j*ka*(N+1))],
            [M[1,0] + M[1,1]*np.exp(1j*ka), -np.exp(1j*ka*N)],
        ], dtype=complex)
        b = -np.array([M[0,0] + M[0,1]*np.exp(-1j*ka),
                       M[1,0] + M[1,1]*np.exp(-1j*ka)], dtype=complex)
        r, t = np.linalg.solve(A, b)
        T[i] = abs(t)**2
    return T

m_r, a = 0.067, 0.05
def seg(nm): return int(round(nm/a))
# device 1: double barrier (the paper's Fig-RTD structure)
U_db = np.concatenate([np.zeros(seg(4)), np.full(seg(1.0),0.3),
                       np.zeros(seg(4.0)), np.full(seg(1.0),0.3),
                       np.zeros(seg(4))])
E2 = np.linspace(0.005, 0.40, 900)
T_negf_db = negf_transmission(E2, U_db, a, m_r)
T_tmm_db  = tmm_transmission(E2, U_db, a, m_r)
dev_db = np.abs(T_negf_db - T_tmm_db).max()
print(f"(b) double barrier: max |T_NEGF - T_TMM| = {dev_db:.3e}  "
      f"(N={len(U_db)}, {len(E2)} energies)")

# device 2: random smooth potential (out-of-family shape)
rng = np.random.default_rng(7)
x = np.arange(seg(12.0)+1)*a
U_rnd = np.zeros_like(x)
for _ in range(3):
    c = rng.uniform(2.0, 10.0); w = rng.uniform(0.5, 1.2); h = rng.uniform(0.1, 0.35)
    U_rnd += h*np.exp(-((x-c)**2)/(2*w*w))
U_rnd[:seg(1.0)] = 0; U_rnd[-seg(1.0):] = 0
T_negf_r = negf_transmission(E2, U_rnd, a, m_r)
T_tmm_r  = tmm_transmission(E2, U_rnd, a, m_r)
dev_r = np.abs(T_negf_r - T_tmm_r).max()
print(f"(b) random 3-bump: max |T_NEGF - T_TMM| = {dev_r:.3e}  (N={len(U_rnd)})")

np.savez(os.path.join(_DATA, "gap1_results.npz"),
         E1=E1, bw_err=bw_err,
         E2=E2, U_db=U_db, T_negf_db=T_negf_db, T_tmm_db=T_tmm_db,
         dev_db=dev_db, dev_r=dev_r,
         T_bw_sym=breit_wigner(E1,0,1,1), T_negf_sym=single_level_T_negf(E1,0,1,1),
         T_bw_asym=breit_wigner(E1,0,1,0.3), T_negf_asym=single_level_T_negf(E1,0,1,0.3))
print("saved gap1_results.npz")
