# TIER: strong
# Harmonic + calendar regression. Fit an additive model by ordinary least
# squares (pure-python normal equations):
#   load(t) = c0 + c1*(t/n)                       (trend)
#           + sum_{k=1..3} daily harmonics(24h)   (double-peak profile)
#           + day-of-week dummies (weekday/weekend + weekly shape)
#           + low-frequency window harmonics       (slow "temperature" swing)
# then extrapolate the fitted model over the horizon. Captures every
# deterministic component so the residual is ~pure noise -> low MASE.
import sys, json, math

inst = json.load(sys.stdin)
y = inst["y"]; m = inst["period"]; H = inst["horizon"]
n = len(y)


def feats(t):
    f = [1.0, t / n]
    for k in (1, 2, 3):                       # daily harmonics
        f.append(math.cos(2 * math.pi * k * t / m))
        f.append(math.sin(2 * math.pi * k * t / m))
    dow = (t // m) % 7                          # day-of-week dummies (1..6)
    for d in range(1, 7):
        f.append(1.0 if dow == d else 0.0)
    for j in (1, 2, 3):                         # low-frequency window harmonics
        f.append(math.cos(2 * math.pi * j * t / n))
        f.append(math.sin(2 * math.pi * j * t / n))
    return f


X = [feats(t) for t in range(n)]
P = len(X[0])

# normal equations A = X^T X, g = X^T y
A = [[0.0] * P for _ in range(P)]
g = [0.0] * P
for row, yi in zip(X, y):
    for a in range(P):
        ra = row[a]
        if ra != 0.0:
            g[a] += ra * yi
            Aa = A[a]
            for b in range(a, P):
                Aa[b] += ra * row[b]
for a in range(P):                             # symmetrise
    for b in range(a):
        A[a][b] = A[b][a]

# ridge for numerical stability
for a in range(P):
    A[a][a] += 1e-6

# Gaussian elimination with partial pivoting
for col in range(P):
    piv = max(range(col, P), key=lambda r: abs(A[r][col]))
    if abs(A[piv][col]) < 1e-15:
        A[piv][col] += 1e-9
    if piv != col:
        A[col], A[piv] = A[piv], A[col]
        g[col], g[piv] = g[piv], g[col]
    pv = A[col][col]
    for r in range(col + 1, P):
        fct = A[r][col] / pv
        if fct == 0.0:
            continue
        for c in range(col, P):
            A[r][c] -= fct * A[col][c]
        g[r] -= fct * g[col]
coef = [0.0] * P
for i in range(P - 1, -1, -1):
    acc = g[i]
    for j in range(i + 1, P):
        acc -= A[i][j] * coef[j]
    coef[i] = acc / A[i][i]

fc = []
for i in range(H):
    row = feats(n + i)
    fc.append(sum(c * r for c, r in zip(coef, row)))
print(json.dumps({"forecast": fc}))
