"""Optional external cross-check with the Kwant package (paper Sec. IV-A).

Builds the SAME tight-binding chain (on-site 2*t0 + U_i, hopping -t0, ideal
1D leads) in Kwant and compares its S-matrix transmission against this
repository's NEGF solver on the paper's double-barrier structure.

Measured: max |T_Kwant - T_NEGF| = 1.17e-12 over 900 energies on the N = 280
double barrier (Kwant 1.5.0, NumPy 1.26.4, SciPy 1.17.1, OpenBLAS, CPython
3.12). The residual is roundoff-dominated, so the last digits may shift with a
different LAPACK build; the script prints the value it measures.

Install note: Kwant 1.5.0 ships pregenerated Cython sources that do not compile
against NumPy >= 2.0 (`PyArray_Descr has no member named subarray`). Build it
in an environment pinned to NumPy 1.x:

    python -m venv kwenv
    kwenv/bin/pip install "numpy<2" scipy cython tinyarray
    kwenv/bin/pip install --no-build-isolation kwant
    kwenv/bin/python verification/kwant_crosscheck.py

Google Colab also works (`!pip install kwant`).
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
import numpy as np
import kwant

from negf_numpy import negf_transmission, HBAR2_OVER_2M0

m_r, a = 0.067, 0.05
t0 = HBAR2_OVER_2M0 / (m_r * a * a)

def seg(nm): return int(round(nm / a))
U = np.concatenate([np.zeros(seg(4)), np.full(seg(1.0), 0.3),
                    np.zeros(seg(4.0)), np.full(seg(1.0), 0.3),
                    np.zeros(seg(4))])
N = len(U)

# ---- build the identical chain in Kwant ----
lat = kwant.lattice.chain(a, norbs=1)
syst = kwant.Builder()
for i in range(N):
    syst[lat(i)] = 2.0 * t0 + U[i]
for i in range(N - 1):
    syst[lat(i), lat(i + 1)] = -t0

lead = kwant.Builder(kwant.TranslationalSymmetry((-a,)))
lead[lat(0)] = 2.0 * t0          # U = 0 in the leads
lead[lat(0), lat(1)] = -t0
syst.attach_lead(lead)
syst.attach_lead(lead.reversed())
fsyst = syst.finalized()

E_grid = np.linspace(0.005, 0.40, 900)
T_kwant = np.array([kwant.smatrix(fsyst, E).transmission(1, 0) for E in E_grid])
T_negf = negf_transmission(E_grid, U, a, m_r)

dev = np.abs(T_kwant - T_negf).max()
print(f"double barrier, N={N}, {len(E_grid)} energies")
print(f"max |T_Kwant - T_NEGF| = {dev:.3e}")
assert dev < 1e-8, "unexpected disagreement -- investigate before citing"
print("PASS: Kwant and NEGF solver agree.")
