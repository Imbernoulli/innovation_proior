# TIER: greedy
# Unsupervised Mahalanobis whitening: W = Sigma^{-1/2} of the training features.
# Down-weights the high-variance nuisance directions and lifts the buried signal,
# but ignores the identity labels, so it only partially recovers the metric.
import sys, json
import numpy as np
inst = json.load(sys.stdin)
X = np.asarray(inst["X_train"], float)
d = inst["d"]
mu = X.mean(0)
Xc = X - mu
Sigma = Xc.T @ Xc / len(X) + np.eye(d) * 1e-3
ev, V = np.linalg.eigh(Sigma)
ev = np.clip(ev, 1e-9, None)
W = (V / np.sqrt(ev)) @ V.T
print(json.dumps({"W": W.tolist()}))
