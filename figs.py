import numpy as np, pandas as pd, networkx as nx, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim import *
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":.25,"grid.linewidth":.5,"axes.labelsize":9.5,"legend.fontsize":8,"figure.dpi":150})
MN={0:"Voter",1:"Non-linear voter ($\\alpha$=2)",2:"Majority rule ($r$=3)",3:"Sznajd"}
COL={0:"#2a78d6",1:"#eb6834",2:"#1baf7a",3:"#6250d6"}
MK={0:"o",1:"s",2:"^",3:"D"}; LS={0:"-",1:"--",2:"-.",3:":"}
NETS=["ER","BA","WS"]; NETLAB={"ER":"Erdős–Rényi","BA":"Barabási–Albert","WS":"Watts–Strogatz"}
def style(m): return dict(color=COL[m],marker=MK[m],ls=LS[m],lw=1.6,ms=4.5,mec="white",mew=.6)
def panels(w=7.2,h=2.5):
    fig,ax=plt.subplots(1,3,figsize=(w,h),sharey=True); return fig,ax
def save(fig,name): fig.tight_layout(); fig.savefig(f"figs/{name}.png",dpi=300); plt.close(fig)
def legend(fig,ax):
    h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.5,-.02)); fig.subplots_adjust(bottom=.28)

# Fig 1 : consensus time vs N
A=pd.read_csv("results/expA.csv")
fig,ax=panels()
for a,net in zip(ax,NETS):
    for m in range(4):
        d=A[(A.net==net)&(A.model==m)].sort_values("N")
        a.errorbar(d.N,d.T_mean,yerr=d.T_std/np.sqrt(d.R),label=MN[m],capsize=2,**style(m))
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("$N$"); a.set_title(NETLAB[net],fontsize=9.5)
    a.set_xticks([250,500,1000,2000]); a.set_xticklabels(["250","500","1000","2000"]); a.minorticks_off()
ax[0].set_ylabel("Consensus time $T_c$ (MCS)")
legend(fig,ax); fig.tight_layout(rect=(0,.12,1,1)); fig.savefig("figs/fig1_consensus_time.png",dpi=300); plt.close(fig)

# Fig 2 : contrarians
B=pd.read_csv("results/expB.csv")
fig,ax=panels()
for a,net in zip(ax,NETS):
    for m in range(4):
        d=B[(B.net==net)&(B.model==m)].sort_values("pc")
        a.errorbar(d.pc,d.m_abs,yerr=d.m_abs_sem,capsize=1.5,label=MN[m],**style(m))
    a.set_xlabel("Contrarian fraction $p_c$"); a.set_title(NETLAB[net],fontsize=9.5); a.set_ylim(0,1.05)
ax[0].set_ylabel("Stationary $\\langle |m| \\rangle$")
legend(fig,ax); fig.tight_layout(rect=(0,.12,1,1)); fig.savefig("figs/fig2_contrarians.png",dpi=300); plt.close(fig)

# Fig 3 : symmetric zealots
try:
    C=pd.read_csv("results/expC.csv")
    fig,ax=panels()
    for a,net in zip(ax,NETS):
        for m in range(4):
            d=C[(C.net==net)&(C.model==m)].sort_values("pz")
            a.errorbar(d.pz,d.m_abs,yerr=d.m_abs_sem,capsize=1.5,label=MN[m],**style(m))
        a.set_xlabel("Zealot fraction $p_z$"); a.set_title(NETLAB[net],fontsize=9.5); a.set_ylim(0,1.05)
    ax[0].set_ylabel("Stationary $\\langle |m| \\rangle$")
    legend(fig,ax); fig.tight_layout(rect=(0,.12,1,1)); fig.savefig("figs/fig3_zealots_symmetric.png",dpi=300); plt.close(fig)
except FileNotFoundError: print("no C yet")

# Fig 4 : one-sided zealots, probability of consensus on zealot opinion
D=pd.read_csv("results/expD.csv")
fig,ax=panels()
for a,net in zip(ax,NETS):
    for m in range(4):
        d=D[(D.net==net)&(D.model==m)].sort_values("pz")
        a.plot(d.pz,d.p_cons1,label=MN[m],**style(m))
    a.set_xlabel("Zealot fraction $p_z$"); a.set_title(NETLAB[net],fontsize=9.5); a.set_ylim(0,1.05)
ax[0].set_ylabel("$P$(consensus on 1)")
legend(fig,ax); fig.tight_layout(rect=(0,.12,1,1)); fig.savefig("figs/fig4_zealots_onesided.png",dpi=300); plt.close(fig)

# Fig 5 : example trajectories, ER, p_c = 0.05
g=make_graph("ER",1000,777,k_mean=6.0)
fig,ax=plt.subplots(1,4,figsize=(7.2,2.3),sharey=True)
for a,m in zip(ax,range(4)):
    r=simulate(g[0],g[1],m,2.0,0.05,0.0,0,1500,750,5,True)[4]
    a.plot(np.arange(len(r)),r,color=COL[m],lw=.9); a.set_title(MN[m].split(" (")[0],fontsize=9); a.set_xlabel("MCS"); a.set_ylim(-1.05,1.05)
ax[0].set_ylabel("Magnetisation $m(t)$"); fig.tight_layout(); fig.savefig("figs/fig5_trajectories.png",dpi=300); plt.close(fig)

# Tables
rows=[]
for net in NETS:
    ks=[];k2=[];kmax=[];cl=[];Ns=[]
    for r in range(20):
        off={"ER":0,"BA":10000,"WS":20000}[net]
        ip,ix=make_graph(net,1000,2026+off+r,k_mean=6.0)
        n=len(ip)-1; deg=np.diff(ip); ks.append(deg.mean()); k2.append((deg**2).mean()/deg.mean()); kmax.append(deg.max()); Ns.append(n)
        G=nx.Graph(); 
        for i in range(n):
            for j in ix[ip[i]:ip[i+1]]: 
                if i<j: G.add_edge(i,int(j))
        cl.append(nx.average_clustering(G))
    rows.append(dict(net=net,N=np.mean(Ns),k=np.mean(ks),k2k=np.mean(k2),kmax=np.mean(kmax),C=np.mean(cl)))
pd.DataFrame(rows).to_csv("results/table_networks.csv",index=False)
# thresholds
th=[]
for net in NETS:
    for m in range(4):
        d=B[(B.net==net)&(B.model==m)].sort_values("pc")
        x=d.pc.values;y=d.m_abs.values
        idx=np.where(y<0.5)[0]
        if len(idx)==0: pcs=np.nan
        elif idx[0]==0: pcs=0
        else:
            i=idx[0]; pcs=x[i-1]+(0.5-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1])
        th.append(dict(net=net,model=m,pc_half=pcs))
pd.DataFrame(th).to_csv("results/table_thresholds.csv",index=False)
print(pd.DataFrame(rows).round(3)); print(pd.DataFrame(th).pivot(index="model",columns="net",values="pc_half").round(3))
