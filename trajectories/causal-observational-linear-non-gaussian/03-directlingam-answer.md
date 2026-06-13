**Problem (from step 2).** ICA-LiNGAM won the Erdos-Renyi scenarios but failed SF100 (F1 0.804, SHD
120, precision 0.702, seed-123 SHD 159) — the signature of a single *global, non-convex* FastICA fit on
a 100-dim unmixing under sub-Gaussian uniform noise landing in a bad optimum. Keep the non-Gaussianity;
drop the global ICA optimization; recover the causal order directly, one variable at a time.

**Key idea.** A DAG has an exogenous variable `x_j = e_j` (its own source). Detect it: regress every
other variable on `x_j` and test independence of the residuals from `x_j`. By Darmois–Skitovitch (which
needs the non-Gaussianity), `x_j` is exogenous **iff** it is independent of all its least-squares
residuals. Append the source, regress it out of the rest (the residuals are again a LiNGAM with the same
relative order), and recurse — `d−1` peeling rounds, no iterative search in parameter space.

**Why it works.** Least squares forces residuals *uncorrelated* with the regressor, so the test must
measure genuine independence. The pairwise likelihood ratio between the two directions reduces — joint
entropies cancel under a unit-determinant map — to a difference of **one-dimensional** differential
entropies, estimated by a fixed max-entropy approximation (log-cosh + skew contrasts). Scoring candidate
`i` by `M(i) = Σ_{j≠i} min(0, diff_MI(i,j))²` (counter-evidence to `i` being the source) and picking the
argmin is deterministic, needs no initialization/step-size, and — because every variable and residual is
standardized — is **scale-invariant**, fixing the ICA permutation's fragility. Order in hand, edges come
from `_BaseLiNGAM._estimate_adjacency_matrix` (adaptive lasso per node, BIC-selected).

**Hyperparameters.** Fixed contrast constants `k1=79.047`, `k2=7.4129`, `gamma=0.37457`; adaptive-lasso
`gamma=1.0` (via causal-learn). No tuning knobs and no random optimization in the ordering step (the seed
only seeds the `_BaseLiNGAM` regression utilities).

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    """
    import os
    import numpy as np
    from sklearn.utils import check_array
    from causallearn.search.FCMBased.lingam.base import _BaseLiNGAM

    X = check_array(X)
    seed = int(os.environ.get("SEED", "42"))
    n_features = X.shape[1]

    # Core DirectLiNGAM steps (from causallearn.search.FCMBased.lingam.direct_lingam)
    def _residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj

    def _entropy(u: np.ndarray) -> float:
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (1 + np.log(2 * np.pi)) / 2 -                k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2 -                k2 * (np.mean(u * np.exp((-u ** 2) / 2))) ** 2

    def _diff_mutual_info(
        xi_std: np.ndarray,
        xj_std: np.ndarray,
        ri_j: np.ndarray,
        rj_i: np.ndarray,
    ) -> float:
        return (_entropy(xj_std) + _entropy(ri_j / np.std(ri_j))) -                (_entropy(xi_std) + _entropy(rj_i / np.std(rj_i)))

    def _search_causal_order(X_work: np.ndarray, U: np.ndarray) -> int:
        if len(U) == 1:
            return int(U[0])
        M_list = []
        for i in U:
            M = 0.0
            for j in U:
                if i == j:
                    continue
                xi_std = (X_work[:, i] - np.mean(X_work[:, i])) / np.std(X_work[:, i])
                xj_std = (X_work[:, j] - np.mean(X_work[:, j])) / np.std(X_work[:, j])
                ri_j = _residual(xi_std, xj_std)
                rj_i = _residual(xj_std, xi_std)
                M += np.min([0.0, _diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)
        return int(U[np.argmax(M_list)])

    U = np.arange(n_features)
    K = []
    X_work = np.copy(X)
    for _ in range(n_features):
        m = _search_causal_order(X_work, U)
        for i in U:
            if i != m:
                X_work[:, i] = _residual(X_work[:, i], X_work[:, m])
        K.append(m)
        U = U[U != m]

    class _LocalDirectLiNGAM(_BaseLiNGAM):
        def fit(self, X):
            return self

    model = _LocalDirectLiNGAM(random_state=seed)
    model._causal_order = K
    model._estimate_adjacency_matrix(X)
    return model.adjacency_matrix_
```
