# TIER: greedy
# Reduced harmonic regression: least-squares fit of a linear trend plus two
# daily harmonics only (no day-of-week calendar term, no slow temperature
# term). This denoises the daily profile and follows the trend -- beating the
# seasonal-naive baseline -- but leaves the weekend step and temperature swing
# unmodelled, so it trails the full calendar+harmonic model.
import sys, json, math

inst = json.load(sys.stdin)
y = inst["y"]; m = inst["period"]; H = inst["horizon"]
n = len(y)


def feats(t):
    f = [1.0, t / n]
    for k in (1, 2):
        f.append(math.cos(2 * math.pi * k * t / m))
        f.append(math.sin(2 * math.pi * k * t / m))
    return f


X = [feats(t) for t in range(n)]
P = len(X[0])
A = [[0.0] * P for _ in range(P)]
g = [0.0] * P
for row, yi in zip(X, y):
    for a in range(P):
        g[a] += row[a] * yi
        for b in range(a, P):
            A[a][b] += row[a] * row[b]
for a in range(P):
    for b in range(a):
        A[a][b] = A[b][a]
    A[a][a] += 1e-6

for col in range(P):
    piv = max(range(col, P), key=lambda r: abs(A[r][col]))
    if piv != col:
        A[col], A[piv] = A[piv], A[col]
        g[col], g[piv] = g[piv], g[col]
    pv = A[col][col]
    for r in range(col + 1, P):
        fct = A[r][col] / pv
        for c in range(col, P):
            A[r][c] -= fct * A[col][c]
        g[r] -= fct * g[col]
coef = [0.0] * P
for i in range(P - 1, -1, -1):
    acc = g[i]
    for j in range(i + 1, P):
        acc -= A[i][j] * coef[j]
    coef[i] = acc / A[i][i]

fc = [sum(c * r for c, r in zip(coef, feats(n + i))) for i in range(H)]
print(json.dumps({"forecast": fc}))
