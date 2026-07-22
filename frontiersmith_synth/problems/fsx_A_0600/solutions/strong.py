# TIER: strong
# Insight: the marginal reduction of the spectral radius from cutting edge (i,j) with
# weight w is, to first order, proportional to  w * x_i * x_j  where x is the leading
# (Perron) eigenvector of the current adjacency matrix -- NOT to w or to degree.
# So we greedily cut the edge of maximum eigen-leverage, and RECOMPUTE the Perron
# vector after every cut (the leverage landscape shifts as the core is dismantled).
import sys, math

ITERS = 500

def perron(n, elist):
    if n == 0:
        return [0.0] * 0
    x = [1.0] * n
    s = math.sqrt(float(n))
    x = [v / s for v in x]
    for _ in range(ITERS):
        y = [0.0] * n
        for (i, j, w) in elist:
            y[i] += w * x[j]
            y[j] += w * x[i]
        nrm = 0.0
        for v in y:
            nrm += v * v
        nrm = math.sqrt(nrm)
        if nrm <= 0.0:
            return x
        inv = 1.0 / nrm
        x = [v * inv for v in y]
    return x

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it)) - 1; v = int(next(it)) - 1; w = int(next(it))
        edges.append((u, v, w))

    remaining = set(range(m))
    removed = []
    for _ in range(min(k, m)):
        cur = [edges[i] for i in remaining]
        x = perron(n, cur)
        best_idx = -1; best_L = -1.0
        for idx in remaining:
            i, j, w = edges[idx]
            L = w * x[i] * x[j]
            if L > best_L:
                best_L = L; best_idx = idx
        if best_idx < 0:
            break
        removed.append(best_idx)
        remaining.discard(best_idx)

    print(" ".join(str(i + 1) for i in removed))

main()
