CAM set the bar — SF20-GP F1 0.881 at SHD 7, ER20-Gauss F1 0.732 at precision 0.97, ER12-LowSample F1 0.564 — and vindicated decoupling order from edges: the recall collapse that sank NOTEARS-MLP is gone and the precision it bought is kept. So the architecture is right, and CAM's *only* remaining weakness is the stage I named as its residual one: the order search. ER12-LowSample is its soft spot, limited by precision (0.44–0.51): on 150 samples the greedy boosted residual-variance ordering places some nodes in the wrong relative order, which then lets spurious upstream edges through that the pruning cannot remove. And on ER20-Gauss the recall (0.59) lags, again an order issue — a node placed slightly too early loses true parents that come "after" it. The common thread is that CAM recovers the order *greedily and indirectly*: $O(d^2)$ boosted fits on growing predictor sets, noisy at small $n$, with wrong turns it never backtracks. The lever is to recover the order *directly and globally*, from a quantity that pins down leaves exactly rather than from comparing fitted residual variances.

I propose **SCORE** (Rolland et al., 2022): read the order off the data distribution's own geometry through score matching. The whole method falls out of one identity. Write the model $X_j = f_j(\mathrm{pa}(j)) + N_j$ with $N_j$ Gaussian and independent, so the joint log-density is $\log p(x) = \sum_j \log p_{N_j}(x_j - f_j(\mathrm{pa}(j)))$ and, for Gaussian noise, $\log p_{N_j}(z) = -z^2/(2\sigma_j^2) + \text{const}$. Take the *score* $s(x) = \nabla \log p(x)$; its $j$-th component depends on $x_j$, on $x_j$'s parents (through $f_j$), and on $x_j$'s children (through their mechanisms). Now take the *second* derivative along $x_j$ — the $j$-th diagonal entry of the Hessian of $\log p$,

$$H_{jj}(x) = \frac{\partial^2 \log p}{\partial x_j^2}.$$

For the Gaussian-noise term, $\partial^2/\partial x_j^2\,[-(x_j - f_j)^2/(2\sigma_j^2)] = -1/\sigma_j^2$, a *constant*, because $f_j$ does not depend on $x_j$ (no self-loops). The *only* $x$-dependence in $H_{jj}$ comes from $x_j$'s appearance inside its *children's* mechanisms. Therefore, if $j$ is a **leaf** — no children — then $H_{jj}(x) = -1/\sigma_j^2$ is constant in $x$, so its variance over the data is zero; and if $j$ is not a leaf, $H_{jj}$ varies with $x$ through the nonlinear child mechanisms, so its variance is strictly positive. That is the exact distributional characterization of a leaf I was missing: **a variable is a leaf iff $\mathrm{Var}[\partial^2 \log p/\partial x_j^2] = 0$** — a property of the score's Jacobian, read directly off the distribution, with no regression and no residual-variance comparison.

This gives a clean, *global*, backtracking-free order recovery. Estimate the diagonal of the Hessian of $\log p$ at the sample points, pick the variable with the **minimum empirical variance** of its diagonal-Hessian column as the current leaf, append it (it goes *last* in the topological order), then *remove that variable from the data* and recompute — because deleting a leaf leaves an ANM over the rest, whose next leaf is the second-to-last variable, and so on. After $d-1$ removals the last remaining variable is the root; reverse the leaf sequence for a topological order, roots first. This is strictly better-posed than CAM's greedy append: leaf identification is a *direct* read of a distributional quantity (exact in the population limit, no greedy commitment that compounds errors), and each step shrinks the problem cleanly — the small-$n$ robustness CAM's growing-predictor regressions lacked.

Everything reduces to estimating, from samples, the diagonal of the Hessian of $\log p$. I cannot differentiate a density I do not have, but I do not need it — only the score and the diagonal of its Jacobian, both estimable *non-parametrically* via Stein's identities with a kernel. First-order Stein gives the score, $G = (K + \eta_G I)^{-1}\nabla K$, where $K_{ij} = \exp(-\lVert x_i - x_j\rVert^2/(2s^2))/s$ is the Gaussian Gram matrix with bandwidth $s$ set to the median pairwise distance, $\nabla K$ its sample gradient ($\nabla K_{kj} = -\sum_i (x_{k,j}-x_{i,j})K_{ik}/s^2$), and $G$ the $(n\times d)$ matrix of estimated scores at the sample points. Second-order Stein gives the diagonal of the score's Jacobian,

$$H = -G^2 + (K + \eta_H I)^{-1}\nabla^2 K,$$

where $\nabla^2 K_{kj} = \sum_i (-1/s^2 + (x_{k,j}-x_{i,j})^2/s^4)K_{ik}$ and the $-G^2$ term is the correction that makes this the *diagonal Hessian of $\log p$* rather than just the derivative of the score. Both are ridge-regularized matrix solves ($\eta_G = \eta_H = 0.001$), $O(n^3)$ per evaluation — entirely practical at the $n = 150$–$2000$ and $d = 12$–$20$ here. Before taking the variance I normalize each column of $H$ by its mean so variables on different scales are compared fairly, then the leaf is the $\mathrm{argmin}$ of the column variances. This is the part with *no analogue* in any prior rung — none of them touch the score's Hessian, which is the object that pins leaves down exactly.

Once the order is in hand, the edge selection is the same regularized cleanup CAM already does well, and the order was the bottleneck, not the pruning, so I reuse it: for each node at position `pos`, fit a GradientBoostingRegressor on its predecessors and keep a parent only if its feature importance exceeds $0.05$. That is the minimal, principled delta from CAM — same decoupled architecture, same nonlinear order-respecting candidate generation, same pruning — with the order itself coming from the exact leaf-variance characterization instead of a greedy residual-variance search. The leaf-detection core (median-distance bandwidth, the $\nabla K$ and $\nabla^2 K$ forms, $G$, the $-G^2$ Hessian correction, per-column mean normalization, and the $\mathrm{argmin}$-variance leaf with iterative column removal and reversal) is transcribed from the reference, re-expressed from PyTorch into numpy; the only adaptation is the harness's gradient-boosted feature-importance pruning standing in for the reference's CAM/GAM significance test — the same pragmatic substitution the other baselines make. The claim is narrow and falsifiable: SCORE should match or beat CAM everywhere, and beat it most exactly where CAM's order search was weakest — ER12-LowSample order accuracy at $n=150$, and ER20-Gauss recall — with parity expected on the already-near-perfect SF20-GP.

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
