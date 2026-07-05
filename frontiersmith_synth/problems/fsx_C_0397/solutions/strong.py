# TIER: strong
# Manifold-aware transferable design: standardize, then cluster on a k-nearest-neighbour
# affinity graph via spectral clustering. Unlike k-means/GMM this recovers non-convex station
# layouts (two crescents, concentric rings) while still handling convex blobs, so it transfers
# across the whole battery instead of overfitting one geometry. Deterministic: fixed seeds and
# the label-free 'discretize' assignment (no k-means randomness in the final step). Falls back
# to GMM then k-means if the graph degenerates.
import sys, json
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

inst = json.load(sys.stdin)
X = np.asarray(inst["points"], dtype=float)
k = int(inst["k"])
n = X.shape[0]
mu = X.mean(axis=0)
sd = X.std(axis=0)
sd[sd == 0] = 1.0
Xs = (X - mu) / sd

lab = None
try:
    nn = min(15, max(5, n // 12))
    lab = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                             n_neighbors=nn, assign_labels="discretize",
                             random_state=0).fit_predict(Xs)
    if len(np.unique(lab)) < 2:
        lab = None
except Exception:
    lab = None

if lab is None:
    try:
        lab = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                              random_state=0).fit_predict(Xs)
    except Exception:
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)

print(json.dumps({"labels": [int(v) for v in lab]}))
