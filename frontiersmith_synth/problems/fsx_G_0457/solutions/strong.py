# TIER: strong
# Linear stacking (the validation-fitted cousin of the oracle).  Ridge-regress the
# validation outcomes on [1, p_1..p_k] to learn a joint fuser that re-weights AND
# de-biases the members at once, predict on test and clamp into [0,1].  Because
# thin validation histories make the raw fit noisy, it shrinks the stacked forecast
# toward a stable skill-weighted committee (more shrinkage when validation is thin
# relative to the number of members).  Deterministic; clearly beats the mean and the
# plain skill-weighting, yet stays below the test-fitted oracle (real headroom).
import sys, json

inst = json.load(sys.stdin)
k = inst["k"]
val_pred = inst["val_pred"]
val_y = inst["val_y"]
test_pred = inst["test_pred"]
nv = len(val_y)


def solve_ridge(A, b, ridge):
    n = len(A)
    m = len(A[0])
    ata = [[0.0] * m for _ in range(m)]
    atb = [0.0] * m
    for r in range(n):
        row = A[r]
        yr = b[r]
        for i in range(m):
            ri = row[i]
            atb[i] += ri * yr
            arow = ata[i]
            for j in range(i, m):
                arow[j] += ri * row[j]
    for i in range(m):
        for j in range(i):
            ata[i][j] = ata[j][i]
        ata[i][i] += ridge
    M = [ata[i][:] + [atb[i]] for i in range(m)]
    for col in range(m):
        piv = col
        best = abs(M[col][col])
        for r in range(col + 1, m):
            v = abs(M[r][col])
            if v > best:
                best = v
                piv = r
        if best < 1e-15:
            continue
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(m):
            if r == col:
                continue
            f = M[r][col] / pv
            if f == 0.0:
                continue
            Mr = M[r]
            Mc = M[col]
            for c in range(col, m + 1):
                Mr[c] -= f * Mc[c]
    w = [0.0] * m
    for i in range(m):
        d = M[i][i]
        w[i] = (M[i][m] / d) if abs(d) > 1e-15 else 0.0
    return w


# ---- linear stacker in raw-probability space (same functional class as the oracle) ----
Aval = [[1.0] + row for row in val_pred]
w = solve_ridge(Aval, [float(y) for y in val_y], ridge=1.0)

# ---- skill-weighted committee: a stable prior to shrink toward ----
bs = [0.0] * k
for i in range(nv):
    row = val_pred[i]
    y = val_y[i]
    for j in range(k):
        d = row[j] - y
        bs[j] += d * d
for j in range(k):
    bs[j] /= max(1, nv)
sw = []
for j in range(k):
    wj = 0.25 - bs[j]
    sw.append(wj if wj > 1e-3 else 1e-3)
tot = sum(sw)
sw = [x / tot for x in sw]

# trust the fitted stacker more when validation is plentiful vs #params
alpha = nv / (nv + 4.0 * (k + 1))

q = []
for row in test_pred:
    stacked = w[0]
    for j in range(k):
        stacked += w[j + 1] * row[j]
    prior = 0.0
    for j in range(k):
        prior += sw[j] * row[j]
    v = alpha * stacked + (1.0 - alpha) * prior
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    q.append(v)
print(json.dumps({"forecast": q}))
