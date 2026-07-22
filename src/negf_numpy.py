"""
Stage 1: A trustworthy classical 1D coherent-NEGF solver (NumPy).
We validate transmission through a single rectangular barrier against the
exact analytical formula.

Units: energies in eV, lengths in nm.
Constant hbar^2/(2 m0) = 0.0380998 eV*nm^2
"""
import numpy as np

HBAR2_OVER_2M0 = 0.0380998  # eV * nm^2


def build_potential_single_barrier(N, a, V0, L_barrier, xc=None):
    """Return U(x) on an N-site grid: flat leads (U=0) with a rectangular
    barrier of height V0 and width L_barrier centered at xc."""
    x = np.arange(N) * a
    if xc is None:
        xc = x[-1] / 2.0
    U = np.zeros(N)
    inside = np.abs(x - xc) <= (L_barrier / 2.0)
    U[inside] = V0
    return x, U


def lead_self_energy(E, t0, U_lead):
    """Retarded surface self-energy of a semi-infinite 1D lead attached to a
    contact site. Sigma = -t0 * exp(i k a), with the decaying (retarded) branch."""
    # cos(k a) = 1 - (E - U_lead)/(2 t0)
    ca = 1.0 - (E - U_lead) / (2.0 * t0)
    ka = np.arccos(ca + 0j)               # complex-safe (handles evanescent E)
    sig = -t0 * np.exp(1j * ka)
    # enforce retarded branch: Im(Sigma) <= 0  -> Gamma = -2 Im(Sigma) >= 0
    if sig.imag > 0:
        sig = np.conj(sig)
    return sig


def negf_transmission(E_grid, U, a, m_r, U_left=0.0, U_right=0.0):
    """Coherent NEGF transmission T(E) for potential profile U on grid spacing a.
    m_r = m*/m0 (effective mass ratio)."""
    t0 = HBAR2_OVER_2M0 / (m_r * a * a)     # hopping (eV)
    N = len(U)
    T = np.zeros_like(E_grid, dtype=float)
    # device Hamiltonian: tridiagonal, diag = 2 t0 + U_i, offdiag = -t0
    H = np.diag(2.0 * t0 + U) \
        + np.diag(-t0 * np.ones(N - 1), 1) \
        + np.diag(-t0 * np.ones(N - 1), -1)
    I = np.eye(N, dtype=complex)
    for i, E in enumerate(E_grid):
        sigL = lead_self_energy(E, t0, U_left)
        sigR = lead_self_energy(E, t0, U_right)
        Sig = np.zeros((N, N), dtype=complex)
        Sig[0, 0] = sigL
        Sig[N - 1, N - 1] = sigR
        G = np.linalg.inv(E * I - H - Sig)
        gamL = -2.0 * sigL.imag
        gamR = -2.0 * sigR.imag
        T[i] = gamL * abs(G[0, N - 1]) ** 2 * gamR
    return T


def analytic_barrier_T(E_grid, V0, L, m_r):
    """Exact transmission through a rectangular barrier (parabolic band)."""
    pref = m_r / HBAR2_OVER_2M0
    T = np.zeros_like(E_grid, dtype=float)
    for i, E in enumerate(E_grid):
        if E <= 0:
            T[i] = 0.0
            continue
        if E < V0:
            kappa = np.sqrt(pref * (V0 - E))
            denom = 1.0 + (V0 ** 2 * np.sinh(kappa * L) ** 2) / (4.0 * E * (V0 - E))
        elif E > V0:
            kp = np.sqrt(pref * (E - V0))
            denom = 1.0 + (V0 ** 2 * np.sin(kp * L) ** 2) / (4.0 * E * (E - V0))
        else:  # E == V0
            kp = np.sqrt(pref * E)
            denom = 1.0 + (pref * V0 * L ** 2) / 4.0  # limit form
        T[i] = 1.0 / denom
    return T


if __name__ == "__main__":
    m_r = 0.067          # GaAs
    a = 0.1              # nm grid
    V0 = 0.3             # eV
    L = 3.0              # nm barrier width
    # device: 6 nm lead + 3 nm barrier + 6 nm lead
    Ltot = 15.0
    N = int(round(Ltot / a)) + 1
    x, U = build_potential_single_barrier(N, a, V0, L)
    E = np.linspace(0.005, 0.6, 400)
    T_negf = negf_transmission(E, U, a, m_r)
    T_ana = analytic_barrier_T(E, V0, L, m_r)
    # compare below and above barrier
    err = np.abs(T_negf - T_ana)
    print(f"N sites = {N},  t0 = {HBAR2_OVER_2M0/(m_r*a*a):.3f} eV")
    print(f"max|T_negf - T_analytic| over full range = {err.max():.3e}")
    mask = E < 0.28
    print(f"max abs error for E<0.28 eV (tunnelling) = {np.abs(T_negf-T_ana)[mask].max():.3e}")
    # a couple of sample points
    for e0 in [0.05, 0.15, 0.25, 0.35, 0.5]:
        j = np.argmin(np.abs(E - e0))
        print(f"  E={E[j]:.3f}  NEGF={T_negf[j]:.5e}  analytic={T_ana[j]:.5e}")
