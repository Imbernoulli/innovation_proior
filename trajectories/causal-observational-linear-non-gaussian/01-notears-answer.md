**Problem.** Recover a directed DAG from observational linear-SEM data. Fitting a linear SEM by least
squares is trivial and statistically consistent (Gaussian *and* non-Gaussian); the only obstacle is the
**acyclicity** constraint, which is combinatorial over a superexponential discrete space and forces every
classical method into discrete search with its scaling limits and bounded-in-degree assumptions. This is
the floor rung: a competent structure learner from a *different* family (continuous optimization) that
deliberately does **not** exploit the non-Gaussian signal the task is built around.

**Key idea.** Turn acyclicity into a smooth scalar equality on real matrices. Since `tr(B^k)` counts
length-`k` closed walks, acyclicity means all those traces vanish; the factorial-reweighted sum is the
matrix exponential, giving `tr(e^B) = d` for binary DAGs (converges for all matrices, terms stay
nonnegative, magnitudes tamed). Squaring the weights entrywise lifts this to signed real matrices with
the same support: `h(W) = tr(e^{W∘W}) − d`, smooth, nonnegative, zero iff DAG, with the one-line
gradient `(e^{W∘W})^T ∘ 2W`. Then solve `min (1/2n)‖X − XW‖² + λ‖W‖₁  s.t.  h(W)=0` by an augmented
Lagrangian.

**Why it works.** The whole `W` is updated at once (global, not edge-by-edge search), the score is
distribution-agnostic, and no in-degree assumption is made. The augmented Lagrangian keeps `ρ` finite via
the multiplier `α ← α + ρh`, raising `ρ` only when feasibility stalls; the `ℓ1` is linearized by
splitting `W = W⁺ − W⁻` (`≥0`), so the inner solve is plain bound-constrained L-BFGS-B; a final hard
threshold rounds the machine-tolerance-feasible solution to a clean DAG.

**Hyperparameters (this task's fill).** L2 loss only; `lambda1=0.1`, `max_iter=100`, `h_tol=1e-8`,
`rho_max=1e16`, `w_threshold=0.3`; diagonal pinned to zero (no self-loops). The reference impl reads
`W[i,j]≠0` as `i→j`, but this task's convention is `B[i,j]≠0 ⇒ j→i`, so the returned matrix is
**transposed** — the single line separating a correct graph from a fully-reversed one.

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    """
    import numpy as np
    import scipy.linalg as sla
    from scipy.optimize import minimize
    from sklearn.utils import check_array

    X = check_array(X)
    n, d = X.shape

    # Reference defaults from Zheng et al. 2018 reference impl.
    lambda1 = 0.1
    max_iter = 100
    h_tol = 1e-8
    rho_max = 1e16
    w_threshold = 0.3

    def _loss_and_grad(W):
        # Squared-error regression loss: 1/(2n) * ||X - X W||^2
        R = X - X @ W
        loss = 0.5 / n * (R ** 2).sum()
        G = -1.0 / n * X.T @ R
        return loss, G

    def _h_and_grad(W):
        # h(W) = tr(exp(W*W)) - d  (Zheng 2018 smooth acyclicity)
        M = W * W
        E = sla.expm(M)
        h = np.trace(E) - d
        G = E.T * 2 * W
        return h, G

    def _obj(w_pm, rho, alpha):
        w_pm = w_pm.reshape(2, d * d)
        W = (w_pm[0] - w_pm[1]).reshape(d, d)
        loss, G_loss = _loss_and_grad(W)
        h, G_h = _h_and_grad(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w_pm.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g = np.concatenate([
            (G_smooth + lambda1).flatten(),
            (-G_smooth + lambda1).flatten(),
        ])
        return obj, g

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    # Non-negative bounds; force diagonal to zero (no self-loops)
    bnds = [(0, 0) if (i == j) else (0, None)
            for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        while rho < rho_max:
            sol = minimize(_obj, w_est, args=(rho, alpha),
                           method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            W_new = (w_new[:d * d] - w_new[d * d:]).reshape(d, d)
            h_new, _ = _h_and_grad(W_new)
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_final = (w_est[:d * d] - w_est[d * d:]).reshape(d, d)
    W_final[np.abs(W_final) < w_threshold] = 0.0

    # NOTEARS: W[i, j] != 0 means i -> j
    # causal-learn / this task: B[i, j] != 0 means j -> i. Transpose.
    return W_final.T
```
