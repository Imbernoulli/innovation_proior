# TIER: strong
"""Strong reducer: standardized (z-scored) PCA folded back into a single affine
map.  Per-gene centring and scaling neutralize loud technical-noise genes and
wildly-scaled housekeeping genes (which raw PCA chases); PCA on the standardized
data then locks onto the CORRELATED co-expression modules -- the true cell-type
factors -- while independent noise genes scatter into tiny trailing components.
Because the standardization is diagonal it folds exactly into z = W (x - b) with
W = V diag(1/sigma), b = mu, so the linear-projection contract is respected.
This recovers the cell-type geometry across clean / noisy / scaled / correlated
atlases, which the geometric mean rewards."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["X_train"], dtype=np.float64)
    d = int(inst["d_target"])

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma > 1e-9, sigma, 1.0)
    Z = (X - mu[None, :]) / sigma[None, :]
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[:d]                      # components in standardized space
    W = V / sigma[None, :]          # fold 1/sigma into the linear map
    print(json.dumps({"W": W.tolist(), "b": mu.tolist()}))


main()
