GraN-DAG confirmed that modeling the mechanisms nonlinearly is the right bet — it lifted ER20-Gauss to mean F1 0.396 (SHD 76), with seed 123 reaching F1 0.55 at precision 0.88 — but it also confirmed the weakness I expected. The per-seed variance is enormous (seed 456 collapses to F1 0.137, SHD 140 on that same scenario), on SF20-GP it is a wash with DirectLiNGAM (F1 0.273 vs 0.319) at far worse SHD (92 vs 60), and on ER12-LowSample it is the worst rung so far (F1 0.161). The pattern across all three is one thing: **precision** (0.11–0.50, unstable) while recall stays respectable. GraN-DAG over-connects because it has no explicit sparsity penalty and no significance pruning, and the order is learned only implicitly through the acyclicity constraint, so the augmented Lagrangian settles on acyclic-but-dense graphs. The next rung must keep nonlinear mechanism modeling but impose *real* edge selection so precision climbs and SHD falls — without the free-net seed lottery.

The continuous-constraint backbone is still right (it is the only thing that avoids both the combinatorial DAG search and the linear-Gaussian blindness), so I keep NOTEARS's $\mathrm{tr}\,e^{W\circ W} - d = 0$. The method I propose for this rung is a **two-stage hybrid: linear NOTEARS to get a skeleton and an order, then a gradient-boosted nonlinear refinement that does the edge selection** — there is no MLP anywhere in this fill, despite the name. Each stage answers a specific weakness of GraN-DAG, so let me walk them in the order the code runs.

Stage one is linear NOTEARS, the original continuous program. Standardize $X$, then minimize

$$\tfrac{0.5}{n}\,\lVert X - XW \rVert_F^2 + \lambda_1 \lVert W \rVert_1 \quad \text{s.t.} \quad h(W) = \mathrm{tr}\,e^{W\circ W} - d = 0,$$

with $\lambda_1 = 0.01$ and the augmented Lagrangian. The acyclicity value and gradient are computed from a *finite power-series* truncation of the matrix exponential (twelve terms), $\mathrm{expm}(M) = I + M + M^2/2! + \cdots$ with $M = W\circ W$, and $\nabla h = 2W \circ \mathrm{expm}(M)^{\!\top}$ — keeping `_h` and `_h_grad` self-contained and consistent with each other rather than relying on a library `expm`. The inner subproblem is solved with **L-BFGS-B** (`jac=True`, 500 inner iterations), the diagonal of $W$ and its gradient zeroed each step to forbid self-loops. The outer loop is the standard schedule: 30 augmented-Lagrangian rounds, $\rho \times 10$ whenever $h$ does not shrink to a quarter of its previous value, $\alpha \mathbin{+}= \rho h$ dual ascent, stopping when $|h| < 10^{-8}$ or $\rho > 10^{16}$. The init is a small random $W$ ($0.01\cdot\texttt{randn}$) to break the symmetry that stalls the optimizer at zeros. This stage is cheap and gives a *linear* skeleton — and, crucially, a *causal ordering* read off the column-sum of $|W|$ (more downstream nodes accumulate more incoming weight), which I use to break the refined graph into a DAG at the end.

A linear skeleton on nonlinear data is exactly what failed at rung one, so stage two repairs that linearity *without* reintroducing GraN-DAG's free-net precision problem. Threshold the linear $W$ at $0.3$ to get candidate parents, then *augment* each node's candidate set with any variable whose linear correlation with it exceeds $0.15$ — so a nonlinearly-strong-but-linearly-weak parent is not lost by the linear stage — and for each node fit a **GradientBoostingRegressor** (100 trees, depth 3) of $X_j$ on its candidate set. Keep a candidate parent only if its boosted-tree *feature importance* exceeds $0.05$. The boosted regressor sees curved mechanisms the linear $W$ cannot, and the feature-importance threshold is the sparsity discipline GraN-DAG never had: it directly answers "does $X_j$ actually use $X_k$ nonlinearly," and prunes the rest. So the precision pressure that was missing becomes the *core* of stage two.

The last step is DAG enforcement, simpler than GraN-DAG's Jacobian pass. The nonlinear refinement can re-introduce a cycle (boosted importances respect no ordering), so I impose the topological order recovered from the *linear* stage: `order_score` $= \sum |W|$ over rows gives a downstream ranking, $\texttt{rank} = \texttt{argsort}$ puts root-like nodes first, and any refined edge pointing from a higher-rank (downstream) node into a lower-rank (upstream) one is deleted. That guarantees acyclicity by construction and inherits the linear stage's order. The output obeys the harness convention $B[i,j] \ne 0$ meaning $j \to i$.

The same-named-vs-paper gap matters because it determines what to expect. This method is **not** the nonlinear MLP-NOTEARS — there is no neural network, no per-variable MLP, no $\lVert \partial_k f_j \rVert_{L^2}$ column-norm dependence summary, no grouped-$\ell_1$ on first-layer weights, no end-to-end continuous nonlinear program. It is *linear* NOTEARS plus a *gradient-boosted* nonlinear edge-selection refinement, using the smooth constraint only to get a linear skeleton and an order. The practical consequence is the mirror image of GraN-DAG: where GraN-DAG over-connected because it had no pruning, this fill will *under*-connect on the hardest scenarios, because the linear stage one supplies the candidate pool and a linear skeleton on strongly-nonlinear GP mechanisms can miss true edges entirely before the boosted refinement ever sees them — the refinement can only prune and re-weight candidates the linear stage and the $0.15$ correlation screen proposed, not discover a parent that is linearly invisible.

So the defining trade is **precision up, recall down**. Feature-importance pruning should push precision well above GraN-DAG's 0.11–0.50 and drop SHD on the dense-graph scenarios where GraN-DAG bloated, with shrinking variance because there is no free-net seed lottery. But the linear-candidate bottleneck caps recall: on SF20-GP, where GP mechanisms are smooth and strongly nonlinear, the linear skeleton will miss many true edges, so recall should crater and F1 may come in below GraN-DAG's there — a precision-heavy, recall-starved profile. On ER20-Gauss and ER12-LowSample I expect competitive, *more stable* F1 with much higher precision. Averaged across the three, F1 should edge out GraN-DAG's *and* SHD should fall dramatically — the precision/SHD win is the point — even if SF20-GP recall is lost. The residual weakness this leaves for a stronger rung is now precise: the *linear candidate-generation bottleneck*. A method that recovers the nonlinear *order* directly, without ever passing through a linear skeleton, should keep this rung's precision discipline while restoring the recall the linear stage throws away.

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    """
    import os
    import numpy as np
    from scipy.optimize import minimize

    n_samples, n_vars = X.shape
    seed = int(os.environ.get("SEED", "42"))

    # --- Hyperparameters ---
    max_iter = 30
    h_tol = 1e-8
    rho_max = 1e+16
    w_threshold = 0.3
    lambda1 = 0.01  # L1 penalty

    # Standardize data
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # --- NOTEARS (linear) on the data ---
    # Formulation: minimize 0.5/n * ||X - X W||^2_F + lambda1 * |W|_1
    # subject to h(W) = tr(e^{W o W}) - d = 0

    def _h(W):
        """Acyclicity constraint: h(W) = tr(e^{W o W}) - d."""
        M = W * W
        # Matrix exponential trace via power series (consistent with _h_grad)
        expm_M = np.eye(n_vars)
        power = np.eye(n_vars)
        for k in range(1, 12):
            power = power @ M / k
            expm_M += power
        return np.trace(expm_M) - n_vars

    def _h_grad(W):
        """Gradient of h w.r.t. W."""
        M = W * W
        # expm(M) via series (10 terms)
        expm_M = np.eye(n_vars)
        power = np.eye(n_vars)
        for k in range(1, 12):
            power = power @ M / k
            expm_M += power
        return 2 * W * expm_M

    def _loss_and_grad(W_flat, rho, alpha):
        W = W_flat.reshape(n_vars, n_vars)
        # Zero diagonal (no self-loops)
        np.fill_diagonal(W, 0)

        # MSE loss: 0.5/n * ||X - XW||^2
        R = X_std - X_std @ W  # (n, d)
        loss = 0.5 / n_samples * np.sum(R ** 2)
        # Gradient of MSE w.r.t. W
        G_mse = -1.0 / n_samples * (X_std.T @ R)  # (d, d)

        # L1 penalty
        l1_loss = lambda1 * np.sum(np.abs(W))
        G_l1 = lambda1 * np.sign(W)

        # Acyclicity
        h_val = _h(W)
        G_h = _h_grad(W)

        total_loss = loss + l1_loss + 0.5 * rho * h_val ** 2 + alpha * h_val
        G_total = G_mse + G_l1 + (rho * h_val + alpha) * G_h

        # Zero diagonal gradient
        np.fill_diagonal(G_total, 0)

        return total_loss, G_total.ravel()

    # --- Augmented Lagrangian ---
    # Small random init to break symmetry (zeros can stall the optimizer)
    rng = np.random.RandomState(seed)
    W_est = rng.randn(n_vars, n_vars) * 0.01
    np.fill_diagonal(W_est, 0)
    rho = 1.0
    alpha_dual = 0.0
    h_prev = np.inf

    for _ in range(max_iter):
        result = minimize(
            lambda w: _loss_and_grad(w, rho, alpha_dual),
            W_est.ravel(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500}
        )
        W_est = result.x.reshape(n_vars, n_vars)
        np.fill_diagonal(W_est, 0)

        h_new = _h(W_est)
        if h_new > 0.25 * h_prev:
            rho *= 10.0
        alpha_dual += rho * h_new
        h_prev = h_new

        if abs(h_new) < h_tol or rho > rho_max:
            break

    # --- Now do nonlinear refinement: for each variable, use kernel regression ---
    # Use the linear NOTEARS skeleton and refine with nonlinear regression
    from sklearn.ensemble import GradientBoostingRegressor

    # Threshold the linear result
    W_abs = np.abs(W_est)
    W_abs[W_abs < w_threshold] = 0.0

    # Refine: for each node, check if candidate parents improve nonlinear fit
    B = np.zeros((n_vars, n_vars))
    for j in range(n_vars):
        # Candidate parents from linear NOTEARS
        candidates = np.where(W_abs[:, j] > 0)[0].tolist()
        # Also add strong linear correlations as candidates
        for i in range(n_vars):
            if i == j:
                continue
            corr = np.abs(np.corrcoef(X_std[:, i], X_std[:, j])[0, 1])
            if corr > 0.15 and i not in candidates:
                candidates.append(i)
        if not candidates:
            continue

        # Nonlinear regression from candidates to j
        gbr = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed, subsample=0.8
        )
        gbr.fit(X_std[:, candidates], X_std[:, j])
        importances = gbr.feature_importances_

        for k, p in enumerate(candidates):
            if importances[k] > 0.05:
                B[j, p] = 1.0

    # Enforce DAG by removing cycles using topological ordering from linear NOTEARS
    # Use causal order from W_est
    order_score = np.sum(np.abs(W_est), axis=0)  # higher = more downstream
    topo_order = np.argsort(order_score)  # ascending = more root-like first
    rank = np.zeros(n_vars, dtype=int)
    for idx, node in enumerate(topo_order):
        rank[node] = idx

    # Remove edges that violate topological ordering
    for i in range(n_vars):
        for j in range(n_vars):
            if B[i, j] != 0 and rank[j] >= rank[i]:
                # j -> i but j is downstream of i: remove
                B[i, j] = 0.0

    return B
```
