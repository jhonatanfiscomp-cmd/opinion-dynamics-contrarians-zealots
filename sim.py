"""Fast opinion-dynamics simulator on complex networks (numba, CSR adjacency).

Binary opinions s in {0,1}. One Monte Carlo step (MCS) = N elementary updates,
with a uniformly random focal node (random sequential update).

Base mechanisms (model id)
  0  VOTER   focal node copies one random neighbour
  1  NL      non-linear voter (Yang et al. 2011): P(1) = n1^a / (n1^a + n0^a)
  2  MR      majority rule (Galam 2002) on a group of size 3 = focal node + 2
             random neighbours; every member adopts the group majority
  3  SZNAJD  focal node i and a random neighbour j; if s_i == s_j, all
             neighbours of i and of j adopt their common opinion

Perturbations
  contrarians (fraction pc, quenched): when influenced, adopt the OPPOSITE of
      the opinion the normal rule would give (for MR: the group minority,
      as in Galam 2004).
  zealots (fraction pz, quenched): never change opinion. `zealot_mode`
      0 = symmetric (half fixed at 0, half at 1), 1 = all fixed at opinion 1.
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
from numba import njit

VOTER, NL, MR, SZNAJD = 0, 1, 2, 3
MODEL_NAMES = {VOTER: "Voter", NL: "Non-linear voter", MR: "Majority rule", SZNAJD: "Sznajd"}


# ----------------------------------------------------------------- networks
def make_graph(kind, N, seed, k_mean=6.0, ws_beta=0.1, er_p=None, ba_m=None, ws_k=None):
    """Return CSR (indptr, indices) of the giant component."""
    if kind == "ER":
        p = er_p if er_p is not None else k_mean / (N - 1)
        G = nx.fast_gnp_random_graph(N, p, seed=seed)
    elif kind == "BA":
        m = ba_m if ba_m is not None else int(round(k_mean / 2))
        G = nx.barabasi_albert_graph(N, m, seed=seed)
    elif kind == "WS":
        k = ws_k if ws_k is not None else int(round(k_mean))
        G = nx.watts_strogatz_graph(N, k, ws_beta, seed=seed)
    else:
        raise ValueError(kind)
    if not nx.is_connected(G):
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
    A = sp.csr_matrix(nx.to_scipy_sparse_array(G, format="csr"))
    A.sort_indices()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64)


# --------------------------------------------------------------- simulation
@njit(cache=True)
def _set(s, m, target, zeal, n1):
    if zeal[m]:
        return n1
    n1 += target - s[m]
    s[m] = target
    return n1


@njit(cache=True)
def simulate(indptr, indices, model, alpha, pc, pz, zealot_mode,
             max_mcs, burn, seed, record):
    """Run one realisation.

    Returns (t_cons, final_n1, mean|m|, mean m^2, traj)
      t_cons  : MCS at which consensus was reached, -1 if never
      mean|m| : time average of |m| over MCS in [burn, max_mcs) (absorbing
                consensus counted as |m| = 1 for the remaining time)
      traj    : magnetisation m(t) per MCS (only if record, else length 0)
    """
    np.random.seed(seed)
    N = indptr.shape[0] - 1
    s = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s[i] = np.random.randint(0, 2)

    # quenched roles: random permutation -> first contrarians, then zealots
    perm = np.random.permutation(N)
    nc = int(N * pc)
    nz = int(N * pz)
    contr = np.zeros(N, dtype=np.bool_)
    zeal = np.zeros(N, dtype=np.bool_)
    for q in range(nc):
        contr[perm[q]] = True
    for q in range(nz):
        v = perm[nc + q]
        zeal[v] = True
        if zealot_mode == 1:
            s[v] = 1
        else:
            s[v] = q % 2

    n1 = 0
    for i in range(N):
        n1 += s[i]

    if record:
        traj = np.zeros(max_mcs + 1)
    else:
        traj = np.zeros(0)

    acc_abs = 0.0
    acc_sq = 0.0
    nacc = 0
    t_cons = -1

    for t in range(max_mcs):
        m = 2.0 * n1 / N - 1.0
        if record:
            traj[t] = m
        if t >= burn:
            acc_abs += abs(m)
            acc_sq += m * m
            nacc += 1
        if n1 == 0 or n1 == N:
            t_cons = t
            # absorbing: fill the rest
            if record:
                for u in range(t, max_mcs + 1):
                    traj[u] = m
            rem = max_mcs - max(t, burn)
            if rem > 0:
                acc_abs += abs(m) * rem
                acc_sq += m * m * rem
                nacc += rem
            break

        for _ in range(N):
            i = np.random.randint(0, N)
            lo = indptr[i]
            k = indptr[i + 1] - lo
            if k == 0:
                continue

            if model == 0:  # voter
                if zeal[i]:
                    continue
                j = indices[lo + np.random.randint(0, k)]
                tgt = s[j]
                if contr[i]:
                    tgt = 1 - tgt
                n1 = _set(s, i, tgt, zeal, n1)

            elif model == 1:  # non-linear voter
                if zeal[i]:
                    continue
                c1 = 0
                for e in range(lo, lo + k):
                    c1 += s[indices[e]]
                c0 = k - c1
                if c0 == 0:
                    p1 = 1.0
                elif c1 == 0:
                    p1 = 0.0
                else:
                    a = c1 ** alpha
                    b = c0 ** alpha
                    p1 = a / (a + b)
                tgt = 1 if np.random.random() < p1 else 0
                if contr[i]:
                    tgt = 1 - tgt
                n1 = _set(s, i, tgt, zeal, n1)

            elif model == 2:  # majority rule, group of 3
                a = indices[lo + np.random.randint(0, k)]
                if k >= 2:
                    b = a
                    while b == a:
                        b = indices[lo + np.random.randint(0, k)]
                else:
                    b = a
                maj = 1 if (s[i] + s[a] + s[b]) >= 2 else 0
                # members (b == a only when degree 1; avoid double update)
                t0 = 1 - maj if contr[i] else maj
                n1 = _set(s, i, t0, zeal, n1)
                t1 = 1 - maj if contr[a] else maj
                n1 = _set(s, a, t1, zeal, n1)
                if b != a:
                    t2 = 1 - maj if contr[b] else maj
                    n1 = _set(s, b, t2, zeal, n1)

            else:  # Sznajd
                j = indices[lo + np.random.randint(0, k)]
                if s[i] == s[j]:
                    o = s[i]
                    for e in range(lo, lo + k):
                        v = indices[e]
                        if v != j:
                            tg = 1 - o if contr[v] else o
                            n1 = _set(s, v, tg, zeal, n1)
                    lj = indptr[j]
                    for e in range(lj, indptr[j + 1]):
                        v = indices[e]
                        if v != i:
                            tg = 1 - o if contr[v] else o
                            n1 = _set(s, v, tg, zeal, n1)

    else:
        pass

    if record and t_cons < 0:
        traj[max_mcs] = 2.0 * n1 / N - 1.0

    if nacc == 0:
        nacc = 1
    return t_cons, n1, acc_abs / nacc, acc_sq / nacc, traj
