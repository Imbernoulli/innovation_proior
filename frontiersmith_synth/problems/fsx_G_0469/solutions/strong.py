# TIER: strong
# Cross-validated recalibration. Fit a Platt (logistic) map as a robust low-variance
# base, and a pool-adjacent-violators isotonic map that can capture the non-affine,
# asymmetric miscalibration a single logistic cannot. The isotonic map is shrunk
# toward the Platt prior; the SHRINKAGE weight is chosen by held-out validation split
# (model selection), so the recalibrator falls back to pure Platt when the flexible
# fit does not generalize and leans on isotonic when it does. It therefore dominates
# either component alone, yet stays strictly below the test-fitted monotone oracle
# (which sees the held-out labels).
import sys, json, math


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def pav(y):
    vals, wts, cnts = [], [], []
    for yi in y:
        vals.append(float(yi)); wts.append(1.0); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-15:
            v2 = vals.pop(); w2 = wts.pop(); c2 = cnts.pop()
            v1 = vals.pop(); w1 = wts.pop(); c1 = cnts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2); cnts.append(c1 + c2)
    out = []
    for v, c in zip(vals, cnts):
        out.extend([v] * c)
    return out


def fit_platt(scores, labels):
    n = len(scores)
    x = [logit(s) for s in scores]
    y = [float(t) for t in labels]
    a, b = 1.0, 0.0
    for _ in range(400):
        ga = gb = 0.0
        for i in range(n):
            p = sigmoid(a * x[i] + b)
            e = p - y[i]
            ga += e * x[i]; gb += e
        a -= 0.1 * ga / n
        b -= 0.1 * gb / n
    return a, b


def build_iso(scores, labels, platt_ab, beta):
    """Isotonic map on (scores,labels) shrunk toward Platt with weight (1-beta);
       returns an apply(s) function via monotone linear-interpolated knots."""
    a, b = platt_ab
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    xs = [scores[i] for i in order]
    ys = [float(labels[i]) for i in order]
    fit = pav(ys)
    gfit = [beta * fit[k] + (1.0 - beta) * sigmoid(a * logit(xs[k]) + b) for k in range(n)]
    kx, kg = [], []
    for k in range(n):
        if kx and abs(xs[k] - kx[-1]) < 1e-12:
            kg[-1] = gfit[k]
        else:
            kx.append(xs[k]); kg.append(gfit[k])

    def apply(s):
        if s <= kx[0]:
            return kg[0]
        if s >= kx[-1]:
            return kg[-1]
        lo, hi = 0, len(kx) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if kx[mid] <= s:
                lo = mid
            else:
                hi = mid
        x0, x1 = kx[lo], kx[hi]
        g0, g1 = kg[lo], kg[hi]
        if x1 - x0 < 1e-12:
            return g0
        return g0 + (s - x0) / (x1 - x0) * (g1 - g0)

    return apply


inst = json.load(sys.stdin)
vs = inst["val_score"]
vy = inst["val_y"]
ts = inst["test_score"]
nv = len(vs)

# ---- deterministic held-out split for shrinkage-weight selection ----
idx = list(range(nv))
# fixed interleaved split (deterministic, no RNG): every 3rd point -> holdout
hold = [i for i in idx if i % 3 == 0]
train = [i for i in idx if i % 3 != 0]
tr_s = [vs[i] for i in train]; tr_y = [vy[i] for i in train]
ho_s = [vs[i] for i in hold]; ho_y = [vy[i] for i in hold]

ab_tr = fit_platt(tr_s, tr_y)
betas = [0.0, 0.25, 0.5, 0.75, 1.0]
best_beta = 0.0
best_bs = None
for beta in betas:
    ap = build_iso(tr_s, tr_y, ab_tr, beta)
    bs = 0.0
    for j in range(len(ho_s)):
        p = min(1.0, max(0.0, ap(ho_s[j])))
        d = p - ho_y[j]
        bs += d * d
    bs /= max(1, len(ho_s))
    if best_bs is None or bs < best_bs - 1e-12:
        best_bs = bs
        best_beta = beta

# ---- refit on ALL validation with the selected shrinkage weight ----
ab_full = fit_platt(vs, vy)
final_map = build_iso(vs, vy, ab_full, best_beta)

out = []
for s in ts:
    p = final_map(s)
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    out.append(p)
print(json.dumps({"prob": out}))
