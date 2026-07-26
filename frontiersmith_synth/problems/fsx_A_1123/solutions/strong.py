# TIER: strong
# The insight: treat anchor placement as a COVERING problem in effective-
# resistance space, not (only) an additive-gain accumulation problem, and
# recognize that neither view alone dominates -- so run BOTH constructions
# and let joint local refinement pick the winner.
#   1) Compute the pseudoinverse of the Laplacian to get pairwise effective
#      resistance R(i,j).
#   2) Farthest-point-cover initialization: start from the highest-degree
#      node, then repeatedly add whichever remaining node is FARTHEST (in
#      resistance) from the current anchor set -- this guarantees every
#      pendant branch/tendril gets touched before any branch gets a SECOND
#      anchor, unlike degree/marginal-gain search which can spend the whole
#      budget on one dense region.
#   2') Also build the sequential marginal-gain (additive-gain) anchor set,
#      the same one the `greedy` tier uses -- it is often strong on generic,
#      unstructured graphs, so discarding it would throw away real signal.
#   3) One round of joint local refinement on EACH of the two candidate sets:
#      try swapping each anchor against the (few) still-farthest non-anchor
#      nodes, evaluating the TRUE lambda_min(L_g) for each trial, and commit
#      the single best improving swap. This is a coordinated, non-additive
#      move (it can trade one anchor for another, not just append), which is
#      exactly what breaks out of the non-submodular trap. Report whichever
#      of the two refined sets scores higher.
import sys, json
import numpy as np


def laplacian(n, edges):
    L = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    return L


def lambda_min_grounded(L, anchors):
    n = L.shape[0]
    keep = [i for i in range(n) if i not in anchors]
    if not keep:
        return 0.0
    Lg = L[np.ix_(keep, keep)]
    return float(np.linalg.eigvalsh(Lg)[0])


def resist(Lp, i, j):
    return float(Lp[i, i] + Lp[j, j] - 2.0 * Lp[i, j])


def farthest_point_init(Lp, n, k, start):
    anchors = [start]
    while len(anchors) < k:
        best_v, best_d = -1, -1.0
        for v in range(n):
            if v in anchors:
                continue
            d = min(resist(Lp, v, s) for s in anchors)
            if d > best_d:
                best_d, best_v = d, v
        anchors.append(best_v)
    return anchors


def swap_pool(Lp, n, anchors, pool_size):
    dists = []
    for v in range(n):
        if v in anchors:
            continue
        d = min(resist(Lp, v, s) for s in anchors)
        dists.append((d, v))
    dists.sort(reverse=True)
    return [v for _, v in dists[:pool_size]]


def marginal_gain_init(L, n, k):
    anchors = []
    for _ in range(k):
        best_v, best_obj = -1, -1.0
        for v in range(n):
            if v in anchors:
                continue
            obj = lambda_min_grounded(L, anchors + [v])
            if obj > best_obj:
                best_obj, best_v = obj, v
        anchors.append(best_v)
    return anchors


def refine_one_round(L, Lp, n, anchors, pool_size):
    anchors = list(anchors)
    cur = lambda_min_grounded(L, anchors)
    pool = swap_pool(Lp, n, anchors, pool_size)
    best_gain, best_ai, best_v, best_obj = 1e-12, -1, -1, cur
    for ai in range(len(anchors)):
        saved = anchors[ai]
        for v in pool:
            if v in anchors:
                continue
            anchors[ai] = v
            obj = lambda_min_grounded(L, anchors)
            if obj - cur > best_gain:
                best_gain, best_ai, best_v, best_obj = obj - cur, ai, v, obj
        anchors[ai] = saved
    if best_ai >= 0:
        anchors[best_ai] = best_v
        cur = best_obj
    return anchors, cur


def main():
    inst = json.load(sys.stdin)
    n, k, edges = inst["n"], inst["k"], inst["edges"]
    L = laplacian(n, edges)
    Lp = np.linalg.pinv(L, hermitian=True)
    pool_size = min(20, n - k)

    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    start = sorted(range(n), key=lambda i: (-deg[i], i))[0]

    cover_init = farthest_point_init(Lp, n, k, start)
    cover_anchors, cover_obj = refine_one_round(L, Lp, n, cover_init, pool_size)

    gain_init = marginal_gain_init(L, n, k)
    gain_anchors, gain_obj = refine_one_round(L, Lp, n, gain_init, pool_size)

    if cover_obj >= gain_obj:
        anchors = cover_anchors
    else:
        anchors = gain_anchors

    print(json.dumps({"anchors": anchors}))


if __name__ == "__main__":
    main()
