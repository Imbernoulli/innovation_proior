# TIER: strong
# Robust regression head via Iteratively Reweighted Least Squares with a Huber
# loss: start from OLS, estimate a robust residual scale (MAD), then downweight
# points whose residual exceeds delta = 1.345 * scale (Huber weight = min(1, delta/|r|)),
# and refit weighted least squares.  Repeated a fixed number of iterations, this
# rejects the corrupted labels and recovers the clean head.  Because the clean
# validation noise floor is positive, the normalized score stays below 1.0.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
d = inst["d"]
X = np.array(inst["X_train"], dtype=float)
y = np.array(inst["y_train"], dtype=float)
n = X.shape[0]

A = np.hstack([X, np.ones((n, 1))])          # design matrix with intercept column


def wls(weights):
    W = np.sqrt(weights)
    Aw = A * W[:, None]
    yw = y * W
    sol, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    return sol


# initial OLS fit
sol = wls(np.ones(n))
for _ in range(30):
    resid = y - A @ sol
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = 1e-9
    delta = 1.345 * scale
    ar = np.abs(resid)
    w_huber = np.where(ar <= delta, 1.0, delta / np.maximum(ar, 1e-12))
    new_sol = wls(w_huber)
    if np.max(np.abs(new_sol - sol)) < 1e-8:
        sol = new_sol
        break
    sol = new_sol

w = sol[:d].tolist()
b = float(sol[d])
# guard against any non-finite escape
if not all(np.isfinite(w)) or not np.isfinite(b):
    w = [0.0] * d
    b = float(np.mean(y))

print(json.dumps({"w": list(w), "b": b}))
