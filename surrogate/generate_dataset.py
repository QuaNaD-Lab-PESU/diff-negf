import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np
from negf_numpy import negf_transmission
np.random.seed(1)

m_r, a = 0.067, 0.15
N = 64
x = np.arange(N)*a
E = np.linspace(0.02, 0.45, 64)
edge = 6

def random_U():
    U = np.zeros(N)
    K = np.random.randint(1,4)          # 1-3 bumps
    for _ in range(K):
        c = np.random.uniform(x[edge], x[-edge])
        w = np.random.uniform(0.4, 1.6)
        h = np.random.uniform(0.08, 0.40)
        U += h*np.exp(-((x-c)**2)/(2*w*w))
    U[:edge]=0; U[-edge:]=0
    return np.clip(U, 0, 0.5)

M = 1000
Us = np.zeros((M,N)); Ts = np.zeros((M,64))
for i in range(M):
    U = random_U()
    Us[i]=U
    Ts[i]=negf_transmission(E, U, a, m_r)
    if i%250==0: print("gen",i,flush=True)
np.savez(os.path.join(_DATA, "fno_data.npz"), x=x, E=E, U=Us, T=Ts, a=a, m_r=m_r)
print("saved", Us.shape, Ts.shape, "T range", Ts.min(), Ts.max())
