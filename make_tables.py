"""Rebuilds results/tables.json (Tables 1-4 of the manuscript) from the CSV files.

Usage (from the project folder, after running figs.py):  python make_tables.py [output.json]
Inputs : results/expA.csv, expB.csv, expE.csv, table_networks.csv, table_thresholds.csv
Output : results/tables.json  (keys t1..t4, each a list of rows of strings)
"""
import sys, json
import numpy as np, pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/tables.json"
NETS = ["ER", "BA", "WS"]
MODEL_LABEL = {0: "Voter", 1: "Non-linear voter (α = 2)", 2: "Majority rule (r = 3)", 3: "Sznajd"}

A = pd.read_csv("results/expA.csv")
B = pd.read_csv("results/expB.csv")
E = pd.read_csv("results/expE.csv")
net = pd.read_csv("results/table_networks.csv").set_index("net")
thr = pd.read_csv("results/table_thresholds.csv")


def fmt_t(mean, std):
    d = 1 if mean < 50 else 0          # one decimal for short times, none for long ones
    return f"{mean:.{d}f} ± {std:.{d}f}"


# Table 1: network structure (mean over 20 networks, N = 1000)
t1 = []
for n in NETS:
    r = net.loc[n]
    t1.append([n, f"{round(r.N):d}", f"{r.k:.2f}", f"{r.k * r.k2k:.1f}", f"{round(r.kmax):d}", f"{r.C:.3f}"])

# Table 2: consensus time at N = 1000 and exponent of T ~ N^z (N = 250-2000)
t2 = []
for m in range(4):
    row = [MODEL_LABEL[m]]
    for n in NETS:
        s = A[(A.net == n) & (A.model == m) & (A.pc == 0) & (A.pz == 0)].sort_values("N")
        z = np.polyfit(np.log(s.N), np.log(s.T_mean), 1)[0]
        r1000 = s[s.N == 1000].iloc[0]
        row += [fmt_t(r1000.T_mean, r1000.T_std), f"{z:.2f}"]
    t2.append(row)

# Table 3: contrarian fraction at which <|m|> crosses 0.5 (N = 1000)
t3 = []
for m in range(4):
    row = [MODEL_LABEL[m]]
    for n in NETS:
        v = thr[(thr.net == n) & (thr.model == m)].pc_half.iloc[0]
        row.append("< 0.005" if v < 0.005 else f"{v:.2f}")
    t3.append(row)

# Table 4: <|m|> at pc = 0.005 for N = 250, 500, 1000, 2000
t4 = []
for m in range(4):
    for i, n in enumerate(NETS):
        vals = []
        for N in (250, 500, 1000, 2000):
            src = B if N == 1000 else E
            s = src[(src.net == n) & (src.model == m) & (src.N == N) & (src.pc == 0.005)]
            vals.append(f"{s.m_abs.iloc[0]:.2f}")
        t4.append([MODEL_LABEL[m] if i == 0 else "", n] + vals)

json.dump(dict(t1=t1, t2=t2, t3=t3, t4=t4), open(OUT, "w"), ensure_ascii=False, indent=1)
print("wrote", OUT)
