# TIER: greedy
# Ordinary least squares (a plain squared / L2 loss) on the raw training data.
# It fits the signal but the squared loss chases the corrupted outliers, so the
# learned head is biased/noisy and generalizes worse than a robust fit.
import sys, json

try:
    import numpy as np
    _HAVE_NP = True
except Exception:
    _HAVE_NP = False

inst = json.load(sys.stdin)
d = inst["d"]
X = inst["X_train"]
y = inst["y_train"]
n = len(X)

if _HAVE_NP:
    A = np.array([row + [1.0] for row in X], dtype=float)   # design matrix + intercept
    b_vec = np.array(y, dtype=float)
    sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    w = sol[:d].tolist()
    b = float(sol[d])
else:
    # pure-python normal equations fallback (ridge-tiny for stability)
    m = d + 1
    AtA = [[0.0] * m for _ in range(m)]
    Atb = [0.0] * m
    for i in range(n):
        row = X[i] + [1.0]
        for a in range(m):
            Atb[a] += row[a] * y[i]
            for c in range(m):
                AtA[a][c] += row[a] * row[c]
    for a in range(m):
        AtA[a][a] += 1e-6
    # Gaussian elimination
    for col in range(m):
        p = max(range(col, m), key=lambda r: abs(AtA[r][col]))
        AtA[col], AtA[p] = AtA[p], AtA[col]
        Atb[col], Atb[p] = Atb[p], Atb[col]
        piv = AtA[col][col]
        for c in range(col, m):
            AtA[col][c] /= piv
        Atb[col] /= piv
        for r in range(m):
            if r != col and AtA[r][col] != 0.0:
                f = AtA[r][col]
                for c in range(col, m):
                    AtA[r][c] -= f * AtA[col][c]
                Atb[r] -= f * Atb[col]
    w = Atb[:d]
    b = Atb[d]

print(json.dumps({"w": list(w), "b": float(b)}))
