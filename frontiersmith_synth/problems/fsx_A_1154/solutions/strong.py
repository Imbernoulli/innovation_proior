# TIER: strong
"""
Exploits the innovation hook directly: the adversary here is NOT all of R^n, it is the
published K-vector family x^(1)..x^(K) sitting right in the input. Instead of a leverage
score computed over the whole space (L_G^+), invert leverage scoring onto the family's own
span:

    relevance_e = sum_k  w_e * (x_k[u]-x_k[v])^2  /  (x_k^T L_G x_k)

An edge that is zero on every published test vector (e.g. every edge of a decoy region the
family never touches) gets relevance 0 and is never selected, no matter how large its GLOBAL
effective resistance is. Among the top-s edges by this restricted relevance, a small linear
program then redistributes the (dim-only, w' in [0, w_e]) weight budget to minimize the WORST
relative error across all K targets simultaneously -- a max-min reallocation across the
family's spectral support, not a single greedy pass.
"""
import sys
import numpy as np
from scipy.optimize import linprog


def main():
    data = sys.stdin.read().split()
    pos = 0
    n = int(data[pos]); m = int(data[pos + 1]); s = int(data[pos + 2]); K = int(data[pos + 3])
    pos += 4
    edges = []
    for _ in range(m):
        u = int(data[pos]); v = int(data[pos + 1]); w = float(data[pos + 2]); pos += 3
        edges.append((u, v, w))
    xs = []
    for _k in range(K):
        x = [float(t) for t in data[pos:pos + n]]
        pos += n
        xs.append(x)

    def quad_form(edge_list):
        totals = [0.0] * K
        for (u, v, w) in edge_list:
            du, dv = u - 1, v - 1
            for ki in range(K):
                diff = xs[ki][du] - xs[ki][dv]
                totals[ki] += w * diff * diff
        return totals

    targets = quad_form(edges)
    targets = [max(t, 1e-12) for t in targets]

    # per-edge contribution to each test vector, and restricted relevance
    contrib = np.zeros((m, K), dtype=np.float64)
    for i, (u, v, w) in enumerate(edges):
        du, dv = u - 1, v - 1
        for ki in range(K):
            diff = xs[ki][du] - xs[ki][dv]
            contrib[i, ki] = w * diff * diff

    relevance = np.array([sum(contrib[i, ki] / targets[ki] for ki in range(K)) for i in range(m)])

    order = sorted(range(m), key=lambda i: -relevance[i])
    sel_idx = [i for i in order[:s] if relevance[i] > 0.0]
    if not sel_idx:
        sel_idx = order[:s]  # degenerate fallback (shouldn't happen on real instances)

    Ssz = len(sel_idx)
    orig_w = np.array([edges[i][2] for i in sel_idx])
    # A[k, e] = coefficient multiplying w'_e in the k-th quadratic form
    A = np.zeros((K, Ssz), dtype=np.float64)
    for col, i in enumerate(sel_idx):
        u, v, _w = edges[i]
        du, dv = u - 1, v - 1
        for ki in range(K):
            diff = xs[ki][du] - xs[ki][dv]
            A[ki, col] = diff * diff

    t_arr = np.array(targets)

    weights = orig_w.copy()  # fallback: keep original weight of every selected edge
    try:
        nvar = Ssz + 1
        c = np.zeros(nvar); c[-1] = 1.0
        rows = []
        rhs = []
        for ki in range(K):
            row = np.zeros(nvar)
            row[:Ssz] = A[ki]
            row[-1] = -t_arr[ki]
            rows.append(row); rhs.append(t_arr[ki])
            row2 = np.zeros(nvar)
            row2[:Ssz] = -A[ki]
            row2[-1] = -t_arr[ki]
            rows.append(row2); rhs.append(-t_arr[ki])
        A_ub = np.array(rows); b_ub = np.array(rhs)
        bounds = [(0.0, float(orig_w[e])) for e in range(Ssz)] + [(0.0, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            weights = np.clip(res.x[:Ssz], 0.0, orig_w)
    except Exception:
        pass  # keep the fallback (original weight on the relevance-selected support)

    out_edges = []
    for col, i in enumerate(sel_idx):
        u, v, _w = edges[i]
        out_edges.append((u, v, float(weights[col])))

    out = [str(len(out_edges))]
    for (u, v, w) in out_edges:
        out.append(f"{u} {v} {w:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
