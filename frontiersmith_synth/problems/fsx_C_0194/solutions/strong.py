# TIER: strong
# Equal-variance top-down order recovery + regression pruning.
#
# For an equal-variance linear-Gaussian SCM the topological order is identifiable:
# a SOURCE station's reading has residual variance == the disturbance variance
# (the global minimum); any station regressed on a set that CONTAINS all its
# parents also collapses to that minimum, while a station still missing a parent
# has strictly larger residual variance. So we grow the order by repeatedly
# picking the remaining station whose residual variance -- when regressed on the
# already-ordered stations -- is smallest. Then each station is regressed on its
# predecessors and only the coefficients above a magnitude threshold are kept as
# directed parent edges. This tracks the true propagation graph closely, with
# finite-sample slack that leaves headroom on the denser, sample-starved nets.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
X = inst["samples"]
m = len(X)

# centered columns
mean = [0.0] * n
for row in X:
    for k in range(n):
        mean[k] += row[k]
mean = [s / m for s in mean]
col = [[row[k] - mean[k] for row in X] for k in range(n)]
var = [sum(v * v for v in col[k]) / m for k in range(n)]


def solve(A, b):
    k = len(b)
    M = [row[:] + [b[r]] for r, row in enumerate(A)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-15:
            M[piv][c] += 1e-12
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(c + 1, k):
            f = M[r][c] / pv
            if f:
                for cc in range(c, k + 1):
                    M[r][cc] -= f * M[c][cc]
    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        acc = M[i][k]
        for j in range(i + 1, k):
            acc -= M[i][j] * x[j]
        x[i] = acc / M[i][i]
    return x


def regress(target, preds):
    """Return (coeffs aligned with preds, residual variance)."""
    k = len(preds)
    y = col[target]
    if k == 0:
        return [], var[target]
    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k
    for a in range(k):
        ca = col[preds[a]]
        for b in range(a, k):
            cb = col[preds[b]]
            s = 0.0
            for t in range(m):
                s += ca[t] * cb[t]
            XtX[a][b] = s
            XtX[b][a] = s
        sy = 0.0
        for t in range(m):
            sy += ca[t] * y[t]
        Xty[a] = sy
    for a in range(k):
        XtX[a][a] += 1e-6
    beta = solve(XtX, Xty)
    ssr = 0.0
    for t in range(m):
        pr = 0.0
        for a in range(k):
            pr += beta[a] * col[preds[a]][t]
        d = y[t] - pr
        ssr += d * d
    return beta, ssr / m


# ---- top-down order recovery ----
ordered = []
remaining = list(range(n))
while remaining:
    best = None
    best_rv = float("inf")
    for v in remaining:
        _, rv = regress(v, ordered)
        if rv < best_rv - 1e-12:
            best_rv = rv
            best = v
    ordered.append(best)
    remaining.remove(best)

# ---- edge pruning by coefficient magnitude ----
thr = 0.20
edges = []
for idx, v in enumerate(ordered):
    preds = ordered[:idx]
    if not preds:
        continue
    beta, _ = regress(v, preds)
    for a, p in enumerate(preds):
        if abs(beta[a]) >= thr:
            edges.append([p, v])          # p -> v, respects the recovered order

print(json.dumps({"edges": edges}))
