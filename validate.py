import numpy as np, time
from sim import *
R=50; N=1000
cfg={"ER":dict(er_p=0.01),"BA":dict(ba_m=3),"WS":dict(ws_k=4,ws_beta=0.1)}
tcc={("ER",1):"659±434",("BA",1):"308±175",("WS",1):"1905±1188",
     ("ER",2):"14±2",("BA",2):"17±3",("WS",2):"313±122",("ER",3):"11±2",("BA",3):"14±3",("WS",3):"497±260"}
t0=time.time()
for net in ["ER","BA","WS"]:
    graphs=[make_graph(net,N,2025+r+{"ER":0,"BA":1000,"WS":2000}[net],**cfg[net]) for r in range(R)]
    print(net,"mean N_gc",np.mean([len(g[0])-1 for g in graphs]),"mean k",np.mean([g[1].size/(len(g[0])-1) for g in graphs]))
    for a in [0.5,1,2,3]:
        model = VOTER if a==1 else NL
        res=[simulate(g[0],g[1],model,float(a),0.0,0.0,0,6000,3000,r,False) for r,g in enumerate(graphs)]
        tc=np.array([x[0] for x in res]); ok=tc>=0
        print(f"  {net} alpha={a}: P(cons)={ok.mean():.2f}  T={tc[ok].mean() if ok.any() else float('nan'):.0f}±{tc[ok].std() if ok.any() else float('nan'):.0f}   (TCC: {tcc.get((net,a),'n/a')})")
print("time",time.time()-t0)
