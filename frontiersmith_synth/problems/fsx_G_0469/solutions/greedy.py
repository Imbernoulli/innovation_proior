# TIER: greedy
# Temperature scaling: fit a SINGLE scalar T and map q = sigmoid(logit(s)/T) on the
# validation history. This corrects global over-confidence (T>1) or under-confidence
# (T<1) -- the dominant miscalibration -- but has NO bias term, so it cannot undo a
# systematic wet/dry offset and cannot follow non-affine curvature. A real but partial
# fix: clearly better than the raw scores, clearly worse than a bias-aware calibrator.
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


inst = json.load(sys.stdin)
vs = inst["val_score"]
vy = inst["val_y"]
ts = inst["test_score"]
n = len(vs)

x = [logit(s) for s in vs]
y = [float(t) for t in vy]

# fit inverse-temperature a = 1/T by gradient descent on log-loss (bias fixed at 0)
a = 1.0
lr = 0.1
for _ in range(400):
    ga = 0.0
    for i in range(n):
        p = sigmoid(a * x[i])
        ga += (p - y[i]) * x[i]
    a -= lr * ga / n

out = []
for s in ts:
    p = sigmoid(a * logit(s))
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    out.append(p)
print(json.dumps({"prob": out}))
