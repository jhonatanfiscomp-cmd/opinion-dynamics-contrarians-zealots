import numpy as np, pandas as pd, time, sys, itertools
from functools import lru_cache
from multiprocessing import Pool
from sim import *

K=6.0
@lru_cache(maxsize=None)
def graphs(net,N,R):
    off={"ER":0,"BA":10000,"WS":20000}[net]
    return [make_graph(net,N,2026+off+r,k_mean=K) for r in range(R)]

def task(a):
    exp,net,model,N,R,pc,pz,zmode,alpha,max_mcs,burn=a
    gs=graphs(net,N,R)
    out=[]
    for r,g in enumerate(gs):
        tc,n1,ma,m2,_=simulate(g[0],g[1],model,alpha,pc,pz,zmode,max_mcs,burn,1000+r,False)
        out.append((tc,n1,ma,m2,len(g[0])-1))
    tc=np.array([o[0] for o in out]); ok=tc>=0
    n1=np.array([o[1] for o in out]); Ns=np.array([o[4] for o in out])
    ma=np.array([o[2] for o in out]); m2=np.array([o[3] for o in out])
    return dict(exp=exp,net=net,model=model,N=N,R=R,pc=pc,pz=pz,zmode=zmode,alpha=alpha,max_mcs=max_mcs,
        p_cons=ok.mean(), T_mean=tc[ok].mean() if ok.any() else np.nan, T_std=tc[ok].std() if ok.any() else np.nan,
        T_med=np.median(tc[ok]) if ok.any() else np.nan,
        T1_mean=tc[ok&(n1==Ns)].mean() if (ok&(n1==Ns)).any() else np.nan,
        p_cons1=(ok&(n1==Ns)).mean(), m_abs=ma.mean(), m_abs_sem=ma.std()/np.sqrt(R), m2=m2.mean(), m_abs_sd=ma.std())

def build(which):
    tasks=[]
    nets=["ER","BA","WS"]; models=[VOTER,NL,MR,SZNAJD]
    if "A" in which:   # baseline consensus time, N scaling
        for net,m in itertools.product(nets,models):
            for N in [250,500,1000,2000]:
                tasks.append(("A",net,m,N,100 if N<=1000 else 50,0.0,0.0,0,2.0,60*N if m==VOTER else 30*N,0))
    if "B" in which:   # contrarians -> stationary |m|
        pcs=[0,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4]
        for net,m,pc in itertools.product(nets,models,pcs):
            tasks.append(("B",net,m,1000,50,pc,0.0,0,2.0,3000,1500))
    if "C" in which:   # symmetric zealots
        pzs=[0,0.01,0.02,0.05,0.1,0.2,0.3]
        for net,m,pz in itertools.product(nets,models,pzs):
            tasks.append(("C",net,m,1000,50,0.0,pz,0,2.0,3000,1500))
    if "D" in which:   # one-sided zealots -> consensus time
        pzs=[0,0.01,0.02,0.05,0.1]
        for net,m,pz in itertools.product(nets,models,pzs):
            tasks.append(("D",net,m,1000,50,0.0,pz,1,2.0,20000,0))
    if "E" in which:   # contrarians, finite-size dependence
        for net,m,N,pc in itertools.product(nets,models,[250,500,2000],[0.005,0.01,0.02,0.1,0.2]):
            tasks.append(("E",net,m,N,30,pc,0.0,0,2.0,3000,1500))
    return tasks

if __name__=="__main__":
    which=sys.argv[1]
    tasks=build(which); t0=time.time()
    with Pool(2) as p: rows=p.map(task,tasks,chunksize=1)
    pd.DataFrame(rows).to_csv(f"results/exp{which}.csv",index=False)
    print(which,len(rows),"tasks",round(time.time()-t0),"s")
