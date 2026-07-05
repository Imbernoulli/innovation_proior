# TIER: greedy
# Plain k-means on standardized coordinates. Strong on convex blobs, blind to rings/moons.
import sys, json
import numpy as np
from sklearn.cluster import KMeans

inst = json.load(sys.stdin)
X = np.asarray(inst["points"], dtype=float)
k = int(inst["k"])
mu = X.mean(axis=0)
sd = X.std(axis=0)
sd[sd == 0] = 1.0
Xs = (X - mu) / sd
km = KMeans(n_clusters=k, n_init=10, random_state=0)
lab = km.fit_predict(Xs)
print(json.dumps({"labels": [int(v) for v in lab]}))
