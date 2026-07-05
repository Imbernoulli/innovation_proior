# TIER: strong
# Learning-to-rank via pointwise ridge regression on the historical log.
# Fit rel ~ w . x + b on the (features, graded-label) pairs in `train` by solving
# the ridge normal equations (X'X + lambda I) beta = X'y with a tiny Gaussian
# elimination (d+1 = 7 unknowns, pure stdlib).  Then score each candidate by the
# learned model and rank best-first.  Because it exploits ALL features -- including
# the session-specific tastes the greedy rating-only rule ignores -- it recovers the
# hidden ordering far better, while irreducible label noise keeps it below NDCG 1.0.
import sys, json

inst = json.load(sys.stdin)
d = inst["d"]
train = inst["train"]
items = inst["items"]
m = len(items)
LAM = 1e-2                       # ridge regularisation

p = d + 1                        # + bias term
# Accumulate A = X'X (+ridge) and g = X'y  with an augmented [1, x] design row.
A = [[0.0] * p for _ in range(p)]
g = [0.0] * p
for rec in train:
    x = rec["x"]
    y = float(rec["rel"])
    row = [1.0] + [float(v) for v in x]
    for a in range(p):
        g[a] += row[a] * y
        ra = row[a]
        Aa = A[a]
        for b in range(p):
            Aa[b] += ra * row[b]
for a in range(p):
    A[a][a] += LAM

# Solve A beta = g via Gaussian elimination with partial pivoting.
def solve(A, g):
    n = len(g)
    M = [A[i][:] + [g[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f != 0.0:
                for j in range(col, n + 1):
                    M[r][j] -= f * M[col][j]
    return [M[i][n] for i in range(n)]

beta = solve(A, g)

def pred(x):
    s = beta[0]
    for j in range(d):
        s += beta[j + 1] * float(x[j])
    return s

scores = [pred(items[i]) for i in range(m)]
order = sorted(range(m), key=lambda i: (scores[i], i), reverse=True)
print(json.dumps({"ranking": order}))
