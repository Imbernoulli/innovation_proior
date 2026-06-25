# SCORE (score-matching causal discovery), distilled

SCORE recovers the directed DAG of a nonlinear additive-noise model by reading
the *topological order* off the **score** of the data distribution — `s(x) = ∇ log p(x)` — instead of
searching for it with regressions. Its organizing fact is that, for Gaussian-noise ANMs, a variable is a
**leaf iff the variance of its diagonal Hessian of `log p` is zero**. Estimate that diagonal Hessian
non-parametrically with Stein identities, peel leaves one at a time to get the order, then prune edges.

## Problem it solves

Recover the fully directed nonlinear causal DAG from observational data, where the irreducibly causal
part is the topological order. Prior order-recovery methods (CAM's greedy residual-variance search,
RESIT's residual-independence tests) are `O(d^2)` regression searches: noisy at small `n`, slow, and
prone to non-backtracked greedy errors. SCORE replaces the search with a direct distributional
measurement and is competitive with the state of the art while substantially faster.

## Key ideas

1. **Leaf ⟺ constant diagonal Hessian.** Under `X_j = f_j(pa(j)) + N_j` with Gaussian `N_j`, the
   `j`-th diagonal entry of the Hessian of `log p`, `H_{jj}(x) = ∂^2 log p / ∂x_j^2`, equals the
   constant `-1/σ_j^2` (the own-noise term, since `f_j` has no self-dependence) plus a term that varies
   with `x` *only through `x_j`'s children*. Hence `j` is a leaf ⟺ `Var_x[H_{jj}(x)] = 0`. This is exact
   in the population.

2. **Order by iterative leaf removal.** Estimate the diagonal Hessian at the sample points; the
   min-variance column is the current leaf; append it (it goes last), remove that variable (the
   remainder is again an ANM), recompute, recurse; reverse to get the order roots-first. A *global
   measurement*, not a greedy fit-comparison.

3. **Stein score / Hessian estimation.** With a Gaussian kernel Gram matrix
   `K_{ij} = exp(-||x_i-x_j||^2/(2s^2))/s` (bandwidth `s` = median pairwise distance), the score is
   `G = (K + η_G I)^{-1} ∇K` and the diagonal Hessian is `-G^2 + (K + η_H I)^{-1} ∇^2 K`, where
   `∇K[k,j] = -Σ_i (x_{k,j}-x_{i,j}) K_{ik}/s^2` and
   `∇^2 K[k,j] = Σ_i (-1/s^2 + (x_{k,j}-x_{i,j})^2/s^4) K_{ik}`. The `-G^2` term corrects the score
   derivative into the genuine `∂^2 log p`. Ridge `η_G = η_H = 10^{-3}`.

4. **CAM-style pruning along the order.** A correct order already yields consistent intervention
   distributions, so edge selection is refinement: fit each node on its predecessors and keep a parent
   only if it clears a significance test (CAM pruning), lowering SHD.

## Why it works

The score's diagonal Hessian isolates exactly the leaf signal (own-noise term constant, only children
add variation), so the order is identifiable directly from the distribution; Stein identities estimate
it without a density. Removing one variable per step keeps each estimate well-posed, giving robustness
at small `n` that greedy growing-predictor regressions lack.

## Hyperparameters

- Ridge regularizers `η_G = η_H = 10^{-3}` (reference defaults).
- Kernel bandwidth `s` = median pairwise distance (no tuning).
- Leaf dispersion: variance of the (mean-normalized) diagonal-Hessian column (`median`/MAD optional).
- Pruning cutoff: small p-value (e.g. `10^{-3}`) for CAM significance pruning.

```python
import numpy as np


def stein_hessian_diagonal(X, eta_G, eta_H, s=None):
    """Diagonal of the Hessian of log p_X at the sample points, via first- and
    second-order Stein identities with a Gaussian kernel. Returns (n, d)."""
    n, d = X.shape
    X_diff = X[:, None, :] - X[None, :, :]                 # (n, n, d)
    sqdist = np.sum(X_diff ** 2, axis=2)                   # (n, n)
    if s is None:
        flat_dist = np.sqrt(np.maximum(sqdist, 0.0)).ravel()
        mid = (flat_dist.size - 1) // 2
        s = np.partition(flat_dist, mid)[mid]                 # torch median convention
    K = np.exp(-sqdist / (2 * s ** 2)) / s                 # Gaussian Gram matrix
    nablaK = -np.einsum('kij,ik->kj', X_diff, K) / s ** 2  # (n, d)
    I = np.eye(n)
    G = np.linalg.solve(K + eta_G * I, nablaK)             # score estimate (n, d)
    nabla2K = np.einsum('kij,ik->kj',
                        -1.0 / s ** 2 + X_diff ** 2 / s ** 4, K)   # (n, d)
    return -G ** 2 + np.linalg.solve(K + eta_H * I, nabla2K)       # diagonal Hessian


def compute_topological_order(X, eta_G=1e-3, eta_H=1e-3):
    """Iterative leaf detection: leaf = min-variance diagonal-Hessian column."""
    data = X.copy()
    active = list(range(X.shape[1]))
    order = []
    while data.shape[1] > 1:
        H = stein_hessian_diagonal(data, eta_G, eta_H)
        H = H / H.mean(axis=0)                  # scale-fair normalization
        leaf = int(np.argmin(H.var(axis=0)))    # constant column => leaf
        order.append(active[leaf])
        active.pop(leaf)
        data = np.delete(data, leaf, axis=1)    # removing a leaf leaves an ANM
    order.append(active[0])
    order.reverse()                             # roots first
    return order


def run_causal_discovery(X, eta_G=1e-3, eta_H=1e-3, cutoff=1e-3):
    """SCORE: score-matching order recovery + CAM-style pruning.
    Returns adjacency B with B[i, j] != 0 meaning j -> i."""
    d = X.shape[1]
    order = compute_topological_order(X, eta_G, eta_H)
    B = np.zeros((d, d))
    for pos in range(1, len(order)):
        j = order[pos]
        parents = order[:pos]
        pvals = gam_significance(X[:, j], X[:, parents])   # CAM significance pruning
        for k, p in enumerate(parents):
            if pvals[k] <= cutoff:
                B[j, p] = 1.0                              # B[child, parent]=1: parent->child
    return B
```
