"""Multi-seed analysis: success statistics per target (classical-solver-judged),
solution character, failure-mode diagnosis, and the paper figure."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np
z = np.load(os.path.join(_DATA, "multiseed_sweep_state.npz"), allow_pickle=True); done = dict(z["done"].item())
E = np.linspace(0.02, 0.45, 90)
def t_res(E): return 0.97/(1.0+((E-0.16)/0.014)**2)+0.01
def t_pass(E): return 0.95/(1.0+((E-0.21)/0.030)**8)+0.02
def t_stop(E): return 0.85-0.83*np.exp(-((E-0.20)**2)/(2*0.015**2))

print("=== RESONANCE: success = |E_pk-0.160|<=7 meV AND T_pk>=0.9 ===")
stats={}
for tname in ["resonance","passband","stopband"]:
    mses=[]; succ=0; det=[]
    for sd in range(5):
        r=done[f"{tname}_{sd}"]; T=r["T"]; U=r["U"]; mses.append(r["mse"])
        nb=int(np.sum((U[1:-1]>U[:-2])&(U[1:-1]>U[2:])&(U[1:-1]>0.05)))
        if tname=="resonance":
            pk=np.argmax(T); ok=(abs(E[pk]-0.16)<=0.007) and (T[pk]>=0.9)
            det.append(f"s{sd}: Epk={E[pk]:.3f} Tpk={T[pk]:.3f} barriers={nb} {'OK' if ok else 'FAIL'}")
        elif tname=="passband":
            inb=np.abs(E-0.21)<=0.025; outb=np.abs(E-0.21)>=0.08
            ok=(T[inb].mean()>=0.7) and (T[outb].mean()<=0.15)
            det.append(f"s{sd}: in-band<T>={T[inb].mean():.3f} out<T>={T[outb].mean():.3f} barriers={nb} {'OK' if ok else 'FAIL'}")
        else:
            notch=np.abs(E-0.20)<=0.008; hi=(E>=0.30)
            ok=(T[notch].min()<=0.3) and (T[hi].mean()>=0.5)
            det.append(f"s{sd}: T@notch={T[notch].min():.3f} high-side<T>={T[hi].mean():.3f} barriers={nb} {'OK' if ok else 'FAIL'}")
        succ+=ok
    stats[tname]=(succ, np.min(mses), np.median(mses), np.max(mses))
    for d in det: print(" ",d)
    print(f"  -> {tname}: success {succ}/5, MSE best/med/worst = {np.min(mses):.3e}/{np.median(mses):.3e}/{np.max(mses):.3e}\n")

# spread across seeds (solution-family question): pairwise U distance, resonance
Us=np.stack([done[f"resonance_{s}"]["U"] for s in range(5)])
dU=max(np.linalg.norm(Us[i]-Us[j])/np.linalg.norm(Us[i]) for i in range(5) for j in range(i+1,5))
print(f"resonance: max pairwise relative ||dU|| across seeds = {dU:.2f}")

# stopband failure diagnosis: what does the best run actually look like?
best_sb=min(range(5), key=lambda s: done[f"stopband_{s}"]["mse"])
Tsb=done[f"stopband_{best_sb}"]["T"]
notch=np.abs(E-0.20)<=0.008
print(f"stopband best seed {best_sb}: T@notch={Tsb[notch].min():.3f}, "
      f"T high-side mean={Tsb[E>=0.30].mean():.3f}, T low-side mean={Tsb[E<=0.12].mean():.3f}")

np.savez(os.path.join(_DATA, "multiseed_stats.npz"), stats=np.array(stats,dtype=object), best_sb=best_sb, dU=dU)
