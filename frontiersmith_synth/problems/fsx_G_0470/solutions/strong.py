# TIER: strong
# Supervised Fisher discriminant (LDA): learn a transform that maximises
# between-identity scatter over within-identity scatter, then keep the top
# discriminant directions. Uses the training labels, so it isolates the
# identity subspace the random rotation smeared across all coordinates.
import sys, json
import numpy as np
inst = json.load(sys.stdin)
X = np.asarray(inst["X_train"], float)
y = np.asarray(inst["y_train"], int)
d = inst["d"]
mu = X.mean(0)
Sw = np.zeros((d, d)); Sb = np.zeros((d, d))
for l in np.unique(y):
    Xl = X[y == l]; ml = Xl.mean(0)
    Sw += (Xl - ml).T @ (Xl - ml)
    Sb += len(Xl) * np.outer(ml - mu, ml - mu)
Sw += np.eye(d) * 1e-1
ev, V = np.linalg.eig(np.linalg.solve(Sw, Sb))
ev = ev.real; V = V.real
order = np.argsort(ev)[::-1]
k = min(8, d)
W = V[:, order[:k]].T
# scale each discriminant by sqrt of its eigenvalue (weight stronger directions)
w = np.sqrt(np.clip(ev[order[:k]], 0, None))
W = W * w[:, None]
print(json.dumps({"W": W.tolist()}))
