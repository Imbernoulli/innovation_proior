# TIER: strong
# INSIGHT: the unknown is not five separate response curves -- it is ONE
# shared sparse linear operator A that every impulse obeys.  Pool the
# (x[t] -> x[t+1]) transition pairs from ALL 5 training impulses into a single
# regression problem per target node, and instead of fitting a dense row
# (which spreads noise into every unexcited direction and generalises badly,
# as the correlation-graph recipe shows), search for a SMALL support: greedily
# add the predictor node whose column best explains the remaining residual
# (orthogonal matching pursuit), stopping after only a few terms.  Because the
# true network has average degree ~2.5, a tightly capped support size is both
# far cheaper to estimate AND far more stable than a full dense fit -- five
# impulses vastly overdetermine a handful of unknowns per row even though the
# same data is hopelessly underdetermined for a dense row.  Symmetrising the
# two independent per-row estimates of each candidate pipe (row i's estimate
# of edge (i,j) and row j's estimate of the same edge) gives a further
# consistency check "for free" from the shared-operator structure.
import sys
import numpy as np

KMAX = 3
EDGE_THRESH = 1e-4


def read_input():
    data = sys.stdin.read().split()
    pos = 0
    t = int(data[pos]); pos += 1
    n = int(data[pos]); pos += 1
    S = int(data[pos]); pos += 1
    T = int(data[pos]); pos += 1
    src = [int(data[pos + i]) for i in range(S)]; pos += S
    blocks = []
    for _ in range(S):
        rows = []
        for _ in range(T + 1):
            row = [float(data[pos + k]) for k in range(n)]
            pos += n
            rows.append(row)
        blocks.append(np.array(rows))
    return n, S, T, src, blocks


def pool_XY(blocks):
    X, Y = [], []
    for rows in blocks:
        for tt in range(len(rows) - 1):
            X.append(rows[tt])
            Y.append(rows[tt + 1])
    return np.array(X), np.array(Y)


def omp_row(X, y, n, kmax):
    support = []
    resid = y.copy()
    for _ in range(kmax):
        best_j, best_score = None, -1.0
        for j in range(n):
            if j in support:
                continue
            col = X[:, j]
            denom = float(col @ col) + 1e-9
            score = abs(float(col @ resid)) / (denom ** 0.5)
            if score > best_score:
                best_score, best_j = score, j
        support.append(best_j)
        Xs = X[:, support]
        coef, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        resid = y - Xs @ coef
    Xs = X[:, support]
    coef, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    return support, coef


def main():
    n, S, T, src, blocks = read_input()
    X, Y = pool_XY(blocks)

    Ahat = np.zeros((n, n))
    for i in range(n):
        support, coef = omp_row(X, Y[:, i], n, KMAX)
        for j, c in zip(support, coef):
            Ahat[i, j] = c

    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            w = (Ahat[i, j] + Ahat[j, i]) / 2.0
            if w > EDGE_THRESH:
                edges[(i, j)] = float(w)

    # feasibility repair: enforce per-node incident weight sum <= 1, clip to (0,1]
    edges = {k: min(v, 1.0) for k, v in edges.items() if v > 0}
    row_sum = [0.0] * n
    for (i, j), w in edges.items():
        row_sum[i] += w
        row_sum[j] += w
    mx = max(row_sum) if row_sum else 0.0
    if mx > 0.999:
        scale = 0.999 / mx
        edges = {k: v * scale for k, v in edges.items()}

    out = [str(len(edges))]
    for (i, j), w in edges.items():
        out.append("%d %d %.8f" % (i, j, w))
    print("\n".join(out))


if __name__ == "__main__":
    main()
