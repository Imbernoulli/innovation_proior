# TIER: greedy
# Univariate filter: rank columns by |Pearson correlation| with the fraud label on
# the training log and keep a FIXED budget of the top 30 (or all, if fewer).  This
# throws out most pure-noise columns -- a big win over keeping everything -- but the
# fixed, generous budget still lets in spurious-correlation columns and never adapts
# the count per dataset, so it leaves accuracy on the table.
import sys, json, math

inst = json.load(sys.stdin)
X = inst["X_train"]
y = inst["y_train"]
F = inst["n_features"]
n = len(y)

my = sum(y) / n
scored = []
for j in range(F):
    mx = 0.0
    for i in range(n):
        mx += X[i][j]
    mx /= n
    sxy = sxx = syy = 0.0
    for i in range(n):
        dx = X[i][j] - mx
        dy = y[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    c = abs(sxy) / math.sqrt(sxx * syy) if sxx > 1e-12 and syy > 1e-12 else 0.0
    scored.append((c, j))

scored.sort(key=lambda t: (-t[0], t[1]))
k = min(30, F)
feats = [j for _, j in scored[:k]]
print(json.dumps({"features": feats}))
