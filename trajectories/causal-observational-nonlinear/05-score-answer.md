**Problem.** CAM gets the nonlinear order right then prunes, but its order is recovered *greedily* (`O(d²)`
boosted residual-variance fits on growing predictor sets), which takes wrong, non-backtracked turns at
small `n` or with subtle mechanisms — the limiting factor on ER12-LowSample (precision-limited F1 0.564)
and ER20-Gauss recall (0.59). The remaining lever is to recover the order *directly and globally*.

**Key idea (SCORE, Rolland et al., ICML 2022, arXiv:2203.04413).** For a Gaussian-noise ANM, the
`j`-th diagonal entry of the Hessian of `log p`, `∂² log p/∂x_j²`, equals `-1/σ_j²` (a constant) plus a
term that varies with `x` *only through `x_j`'s children*. Hence **`j` is a leaf iff Var[∂² log p/∂x_j²]
= 0**. Estimate the score and its diagonal Jacobian non-parametrically via Stein identities with a
Gaussian kernel (`G = (K+η_G I)⁻¹∇K`, `H = -G² + (K+η_H I)⁻¹∇²K`), pick the min-variance column as the
leaf, remove it, recompute, repeat; reverse to get the topological order. Then prune edges with the same
nonlinear order-respecting selection CAM uses.

**Why it should clear CAM.** Leaf identification is a *direct* read of a distributional quantity (exact
in the population limit, no greedy commitment that compounds), and each step shrinks the problem cleanly —
the small-`n` robustness CAM's growing-predictor regressions lack. Reuses CAM's proven pruning, so the
only change is replacing the greedy order with the exact leaf-variance order.

**Faithfulness.** The leaf-detection core (median-distance bandwidth, `∇K`/`∇²K` forms, `G`, `-G²`
Hessian correction, per-column mean normalization, argmin-variance leaf with iterative column removal and
reversal) is transcribed from the reference implementation (re-expressed from PyTorch into numpy);
`η_G = η_H = 0.001` are the reference defaults. The pruning is the harness's gradient-boosted
feature-importance selection (cutoff `0.05`), substituting for the reference's CAM/GAM significance
pruning — the same pragmatic substitution the task's other baselines make.

**Bar (no leaderboard row).** Match or beat CAM everywhere; beat it most on ER12-LowSample (order
accuracy at `n=150`) and ER20-Gauss recall; parity expected on the already-near-perfect SF20-GP.

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    SCORE (Rolland et al., ICML 2022): recover the topological order by
    score-matching leaf detection -- a variable is a leaf iff the variance of
    the j-th diagonal Hessian entry of log p is zero -- then prune edges.
    """
    import os
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor

    n_samples, n_vars = X.shape
    seed = int(os.environ.get("SEED", "42"))

    eta_G = 0.001
    eta_H = 0.001

    # Use the data directly, as in the reference SCORE implementation.
    Xc = np.asarray(X, dtype=float)

    def stein_hess_diag(data):
        # Estimate the diagonal of the Hessian of log p_X at the sample points
        # via first- and second-order Stein identities (Rolland et al., 2022).
        m, p = data.shape
        X_diff = data[:, None, :] - data[None, :, :]          # (m, m, p)
        sqdist = np.sum(X_diff ** 2, axis=2)                   # (m, m)
        dist = np.sqrt(np.maximum(sqdist, 0.0))
        flat_dist = dist.ravel()
        mid = (flat_dist.size - 1) // 2
        s = np.partition(flat_dist, mid)[mid]                 # torch median convention
        K = np.exp(-sqdist / (2 * s ** 2)) / s                # Gaussian Gram matrix
        # nablaK[k, j] = -sum_i (data[k,j]-data[i,j]) K[i,k] / s^2
        nablaK = -np.einsum('kij,ik->kj', X_diff, K) / s ** 2  # (m, p)
        I = np.eye(m)
        G = np.linalg.solve(K + eta_G * I, nablaK)            # score estimate (m, p)
        # nabla2K[k, j] = sum_i (-1/s^2 + (data[k,j]-data[i,j])^2 / s^4) K[i,k]
        nabla2K = np.einsum('kij,ik->kj',
                            -1.0 / s ** 2 + X_diff ** 2 / s ** 4, K)  # (m, p)
        # diagonal Hessian of log p = -G^2 + (K + eta_H I)^{-1} nabla2K
        return -G ** 2 + np.linalg.solve(K + eta_H * I, nabla2K)

    # --- Step 1: topological order via score-matching leaf removal ---
    # Leaf = variable whose diagonal-Hessian column has minimum variance.
    data = Xc.copy()
    active = list(range(n_vars))
    order = []
    while data.shape[1] > 1:
        H = stein_hess_diag(data)
        H = H / H.mean(axis=0)                  # normalize columns before variance
        leaf = int(np.argmin(H.var(axis=0)))    # min-variance column is the leaf
        order.append(active[leaf])
        active.pop(leaf)
        data = np.delete(data, leaf, axis=1)
    order.append(active[0])
    order.reverse()                             # roots first

    # --- Step 2: edge selection along the order (CAM-style nonlinear pruning) ---
    B = np.zeros((n_vars, n_vars))
    for pos in range(1, len(order)):
        j = order[pos]
        candidate_parents = order[:pos]
        gbr = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed, subsample=0.8
        )
        gbr.fit(Xc[:, candidate_parents], Xc[:, j])
        importances = gbr.feature_importances_
        for k, p in enumerate(candidate_parents):
            if importances[k] > 0.05:
                B[j, p] = 1.0   # B[child, parent] = 1 means parent -> child

    return B
```
