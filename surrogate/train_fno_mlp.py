"""GAP 3 stage 2: train FNO + parameter-matched MLP identically on the
in-family training split; evaluate aggregate and per-spectrum worst-case
relative L2 on in-family test and the three OOD splits."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
_DATA = os.path.join(_HERE, '..', 'data')
import numpy as np, torch, torch.nn as nn
torch.set_default_dtype(torch.float64); torch.manual_seed(0); np.random.seed(0)
d = np.load(os.path.join(_DATA, "fno_data.npz")); U, T = d["U"], d["T"]; N = U.shape[1]
o = np.load(os.path.join(_DATA, "gap3_ood.npz"))
Xtr,Ttr = torch.tensor(U[:800]), torch.tensor(T[:800])
splits = {"in-family": (torch.tensor(U[800:]), torch.tensor(T[800:])),
          "OOD-tall": (torch.tensor(o["U_tall"]), torch.tensor(o["T_tall"])),
          "OOD-narrow": (torch.tensor(o["U_narrow"]), torch.tensor(o["T_narrow"])),
          "OOD-4bump": (torch.tensor(o["U_4b"]), torch.tensor(o["T_4b"]))}

class SpectralConv1d(nn.Module):
    def __init__(s, ci, co, modes):
        super().__init__(); s.modes=modes
        s.w=nn.Parameter((1.0/(ci*co))*torch.randn(ci,co,modes,dtype=torch.cfloat).to(torch.complex128))
    def forward(s,x):
        xf=torch.fft.rfft(x,dim=-1)
        of=torch.zeros(x.shape[0],s.w.shape[1],xf.shape[-1],dtype=torch.complex128)
        of[...,:s.modes]=torch.einsum('bim,iom->bom',xf[...,:s.modes],s.w)
        return torch.fft.irfft(of,n=x.shape[-1],dim=-1)

class FNO1d(nn.Module):
    def __init__(s, modes=16, width=48):
        super().__init__()
        s.fc0=nn.Linear(2,width)
        s.c=nn.ModuleList([SpectralConv1d(width,width,modes) for _ in range(4)])
        s.w=nn.ModuleList([nn.Conv1d(width,width,1) for _ in range(4)])
        s.fc1=nn.Linear(width,64); s.fc2=nn.Linear(64,1)
        s.register_buffer("gx",torch.linspace(0,1,N).view(1,N,1))
    def forward(s,u):
        B=u.shape[0]
        h=s.fc0(torch.cat([u.unsqueeze(-1),s.gx.expand(B,N,1)],-1)).permute(0,2,1)
        for cc,ww in zip(s.c,s.w): h=torch.relu(cc(h)+ww(h))
        return s.fc2(torch.relu(s.fc1(h.permute(0,2,1)))).squeeze(-1)

def nparams(m): return sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters())
fno = FNO1d(); pf = nparams(fno)

# MLP with matched parameter count: 64 -> H -> H -> 64
def mlp_params(H): return 64*H+H + H*H+H + H*64+64
H = 64
while mlp_params(H+1) <= pf: H += 1
mlp = nn.Sequential(nn.Linear(64,H), nn.ReLU(), nn.Linear(H,H), nn.ReLU(), nn.Linear(H,64))
print(f"FNO params={pf:,}  MLP(H={H}) params={nparams(mlp):,}")

def train(net, tag, epochs=250):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    for ep in range(epochs):
        perm = torch.randperm(800); net.train()
        for i in range(0,800,64):
            idx=perm[i:i+64]; opt.zero_grad()
            loss=torch.mean((net(Xtr[idx])-Ttr[idx])**2)
            loss.backward(); opt.step()
        sch.step()
        if ep%80==0: print(f"  {tag} ep{ep} loss={loss.item():.2e}", flush=True)
    return net

train(fno, "FNO"); torch.save(fno.state_dict(), os.path.join(_DATA, "fno_weights.pt"))
train(mlp, "MLP"); torch.save(mlp.state_dict(), os.path.join(_DATA, "mlp_weights.pt"))

def metrics(net):
    out={}
    net.eval()
    with torch.no_grad():
        for k,(X,Tt) in splits.items():
            P=net(X)
            agg=(torch.norm(P-Tt)/torch.norm(Tt)).item()
            per=torch.linalg.norm(P-Tt,dim=1)/torch.linalg.norm(Tt,dim=1)
            out[k]=(agg, per.max().item(), per.median().item())
    return out

mf, mm = metrics(fno), metrics(mlp)
print(f"\n{'split':12s} | FNO agg / worst / med  | MLP agg / worst / med")
for k in splits:
    a1,w1,d1 = mf[k]; a2,w2,d2 = mm[k]
    print(f"{k:12s} | {a1*100:5.1f}% {w1*100:6.1f}% {d1*100:5.1f}% | {a2*100:5.1f}% {w2*100:6.1f}% {d2*100:5.1f}%")
np.savez(os.path.join(_DATA, "gap3_results.npz"),
         fno={k:v for k,v in mf.items()}, mlp={k:v for k,v in mm.items()},
         pf=pf, pm=nparams(mlp), H=H, allow_pickle=True)
