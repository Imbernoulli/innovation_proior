# TIER: strong
# Rank columns by |correlation| with the fraud label (same univariate filter as the
# greedy baseline), THEN choose HOW MANY to keep by internally re-simulating the
# frozen nearest-centroid scorer under 5-fold cross-validation on the training log:
# for each candidate budget k in a grid, estimate held-out accuracy by CV and keep the
# best-scoring prefix.  Adapting the budget per dataset drops the spurious-correlation
# columns the fixed-budget greedy filter keeps, so it tracks the planted informative
# set far more closely -- but the finite log's CV noise keeps it below the oracle.
import sys, json, math

inst = json.load(sys.stdin)
X = inst["X_train"]
y = inst["y_train"]
F = inst["n_features"]
n = len(y)

# ---- univariate correlation ranking ----
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
rank = [j for _, j in scored]


# ---- cross-validated frozen-model accuracy for a given column subset ----
def cv_score(sel, folds=5):
    correct = 0
    total = 0
    for f in range(folds):
        val = [i for i in range(n) if i % folds == f]
        fit = [i for i in range(n) if i % folds != f]
        if not val or not fit:
            continue
        mu = {}
        sd = {}
        for j in sel:
            m = 0.0
            for i in fit:
                m += X[i][j]
            m /= len(fit)
            v = 0.0
            for i in fit:
                d = X[i][j] - m
                v += d * d
            v /= len(fit)
            mu[j] = m
            sd[j] = math.sqrt(v) if v > 1e-12 else 1.0
        cnt = {0: 0, 1: 0}
        sums = {0: {j: 0.0 for j in sel}, 1: {j: 0.0 for j in sel}}
        for i in fit:
            c = y[i]
            cnt[c] += 1
            for j in sel:
                sums[c][j] += (X[i][j] - mu[j]) / sd[j]
        cen = {0: {}, 1: {}}
        for c in (0, 1):
            for j in sel:
                cen[c][j] = sums[c][j] / cnt[c] if cnt[c] > 0 else 0.0
        maj = 0 if cnt[0] >= cnt[1] else 1
        for i in val:
            d0 = 0.0
            d1 = 0.0
            for j in sel:
                z = (X[i][j] - mu[j]) / sd[j]
                e0 = z - cen[0][j]
                e1 = z - cen[1][j]
                d0 += e0 * e0
                d1 += e1 * e1
            p = 0 if d0 < d1 else (1 if d1 < d0 else maj)
            if p == y[i]:
                correct += 1
            total += 1
    return correct / total if total else 0.0


grid = [g for g in (3, 5, 8, 12, 18, 25, 35, 50) if g <= F]
if not grid:
    grid = [min(F, 1)]
best_k = grid[0]
best_cv = -1.0
for k in grid:
    cv = cv_score(rank[:k])
    if cv > best_cv + 1e-12:
        best_cv = cv
        best_k = k

feats = rank[:best_k]
print(json.dumps({"features": feats}))
