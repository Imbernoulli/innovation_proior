# TIER: trivial
# Median split on the maximum-variance coordinate: two groups, ignores k and geometry.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
X = np.asarray(inst["points"], dtype=float)
j = int(np.argmax(X.var(axis=0)))
lab = (X[:, j] > np.median(X[:, j])).astype(int)
print(json.dumps({"labels": lab.tolist()}))
