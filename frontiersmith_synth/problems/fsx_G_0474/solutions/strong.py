# TIER: strong
"""Precision-matrix (partial-correlation) skeleton + forcing-anchored orientation.

Skeleton: the inverse covariance (precision) matrix is, up to scaling, the matrix
of partial correlations; an off-diagonal entry near zero means the two channels
are conditionally independent given all the others.  Thresholding |partial corr|
recovers the conditional-independence graph -- the true skeleton plus the
"moralization" edges between co-parents -- far cleaner than marginal correlation
because it drops indirect-path false edges.

Orientation: the published FORCING nodes are known roots.  We layer the skeleton
by breadth-first distance from the forcing set, orienting every skeleton edge from
the lower layer to the higher layer (out of / away from the forcings).  Because
the disturbances are heteroscedastic, the classic "source has smallest marginal
variance" rule is unreliable, so it is used only as a last-resort tie-breaker for
edges the forcing layering leaves flat (same layer, or a component the forcings
never reach).

Leftover moralization edges, occasional mis-layered pairs, and forcing-unreachable
components keep it short of perfect -- especially on the larger, denser,
data-poorer meshes."""
import sys, json
from collections import deque
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["data"], dtype=np.float64)
    d = int(inst["n_nodes"])
    forcing = [int(f) for f in inst.get("forcing_nodes", []) if 0 <= int(f) < d]

    # ---- skeleton via thresholded partial correlations ----
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    cov = cov + 1e-6 * np.eye(d)
    prec = np.linalg.pinv(cov)
    dp = np.sqrt(np.abs(np.diag(prec))) + 1e-12
    pcorr = -prec / np.outer(dp, dp)

    thr = 0.25
    skel = [[] for _ in range(d)]
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            if abs(pcorr[i, j]) > thr:
                skel[i].append(j)
                skel[j].append(i)
                pairs.append((i, j))

    # ---- BFS layering from the forcing set ----
    INF = 10 ** 9
    layer = [INF] * d
    dq = deque()
    for f in forcing:
        layer[f] = 0
        dq.append(f)
    while dq:
        u = dq.popleft()
        for v in skel[u]:
            if layer[v] > layer[u] + 1:
                layer[v] = layer[u] + 1
                dq.append(v)

    # variance-rank fallback (ascending marginal variance ~ earlier in topo order)
    var = X.var(axis=0)
    vorder = np.argsort(var, kind="mergesort")
    vrank = np.empty(d, dtype=np.int64)
    for pos, node in enumerate(vorder):
        vrank[node] = pos

    edges = []
    for (i, j) in pairs:
        li, lj = layer[i], layer[j]
        if li != lj:
            a, b = (i, j) if li < lj else (j, i)
        elif vrank[i] != vrank[j]:
            a, b = (i, j) if vrank[i] < vrank[j] else (j, i)
        else:
            a, b = i, j
        edges.append([a, b])
    print(json.dumps({"edges": edges}))


main()
