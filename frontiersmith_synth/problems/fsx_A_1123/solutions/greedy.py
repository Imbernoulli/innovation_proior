# TIER: greedy
# The classical "additive-gain" leader-selection recipe: build the anchor set
# ONE node at a time, always taking whichever remaining node gives the
# largest immediate lambda_min(L_g) if added right now (recomputing the true
# eigenvalue at every step -- this is NOT a cheap degree heuristic, it is the
# textbook sequential marginal-gain greedy). Provably gets stuck wherever the
# objective is non-submodular, and has no notion of "cover the far branches
# jointly" -- it only ever asks "what's best to add given what I already
# fixed", never revisits earlier picks.
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


def main():
    inst = json.load(sys.stdin)
    n, k, edges = inst["n"], inst["k"], inst["edges"]
    L = laplacian(n, edges)

    anchors = []
    for _ in range(k):
        best_v, best_obj = -1, -1.0
        for v in range(n):
            if v in anchors:
                continue
            trial = anchors + [v]
            obj = lambda_min_grounded(L, trial)
            if obj > best_obj:
                best_obj, best_v = obj, v
        anchors.append(best_v)

    print(json.dumps({"anchors": anchors}))


if __name__ == "__main__":
    main()
