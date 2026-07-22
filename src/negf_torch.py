"""
Stage 2: Differentiable NEGF in PyTorch.
Same physics as the NumPy solver, but written so autograd can hand us
dT/dU for every grid point for free. We verify against finite differences.
"""
import torch

HBAR2_OVER_2M0 = 0.0380998  # eV nm^2


def sigma_lead(E, t0, U_lead):
    """Retarded lead self-energy for a batch of energies E (tensor)."""
    ca = 1.0 - (E - U_lead) / (2.0 * t0)          # cos(k a), real
    s = torch.sqrt(torch.clamp(1.0 - ca * ca, min=0.0))
    exp_ika = torch.complex(ca, s)                # ca + i sqrt(1-ca^2)  (retarded)
    return -t0 * exp_ika


def negf_T(U, E_grid, a, m_r, U_left=0.0, U_right=0.0):
    """Differentiable transmission T(E) for potential U (tensor, shape [N]).
    Returns tensor of shape [nE]. Gradients flow through U."""
    t0 = HBAR2_OVER_2M0 / (m_r * a * a)
    N = U.shape[0]
    nE = E_grid.shape[0]
    dev = U.device
    # tridiagonal device Hamiltonian
    H = torch.diag(2.0 * t0 + U.to(torch.complex128))
    off = -t0 * torch.ones(N - 1, dtype=torch.complex128, device=dev)
    H = H + torch.diag(off, 1) + torch.diag(off, -1)
    I = torch.eye(N, dtype=torch.complex128, device=dev)
    sigL = sigma_lead(E_grid, t0, U_left)          # [nE]
    sigR = sigma_lead(E_grid, t0, U_right)          # [nE]
    gamL = -2.0 * sigL.imag
    gamR = -2.0 * sigR.imag
    Ecx = E_grid.to(torch.complex128)
    # Build A = E I - H - Sigma for all energies:  [nE, N, N]
    A = Ecx.view(nE, 1, 1) * I.unsqueeze(0) - H.unsqueeze(0)
    A[:, 0, 0] = A[:, 0, 0] - sigL
    A[:, N - 1, N - 1] = A[:, N - 1, N - 1] - sigR
    G = torch.linalg.inv(A)                          # [nE, N, N]
    G1N = G[:, 0, N - 1]
    T = gamL * (G1N.abs() ** 2) * gamR
    return T.real


if __name__ == "__main__":
    import numpy as np
    from negf_numpy import negf_transmission

    torch.set_default_dtype(torch.float64)
    m_r, a = 0.067, 0.1
    # random-ish smooth barrier
    N = 121
    x = np.arange(N) * a
    U_np = 0.25 * np.exp(-((x - 4.0) ** 2) / (2 * 0.8 ** 2)) \
         + 0.20 * np.exp(-((x - 8.0) ** 2) / (2 * 0.8 ** 2))
    E = np.linspace(0.02, 0.5, 60)

    # 1) torch matches numpy
    U_t = torch.tensor(U_np, requires_grad=True)
    E_t = torch.tensor(E)
    T_t = negf_T(U_t, E_t, a, m_r)
    T_np = negf_transmission(E, U_np, a, m_r)
    print("max|T_torch - T_numpy| =", np.abs(T_t.detach().numpy() - T_np).max())

    # 2) gradient check: d(sum T)/dU  vs finite differences
    loss = T_t.sum()
    loss.backward()
    g_auto = U_t.grad.detach().numpy().copy()

    eps = 1e-6
    g_fd = np.zeros(N)
    for j in range(N):
        Up = U_np.copy(); Up[j] += eps
        Um = U_np.copy(); Um[j] -= eps
        Tp = negf_transmission(E, Up, a, m_r).sum()
        Tm = negf_transmission(E, Um, a, m_r).sum()
        g_fd[j] = (Tp - Tm) / (2 * eps)
    rel = np.linalg.norm(g_auto - g_fd) / np.linalg.norm(g_fd)
    print("relative L2 error (autodiff vs finite-diff gradient) =", rel)
    print("sample: site 40  auto=%.6e  fd=%.6e" % (g_auto[40], g_fd[40]))
