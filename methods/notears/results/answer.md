# NOTEARS, distilled

NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for
Structure learning) learns the structure of a directed acyclic graph by turning the combinatorial
acyclicity constraint into a single smooth equality constraint over real matrices, so DAG learning
becomes ordinary continuous constrained optimization solvable by off-the-shelf numerical solvers.

## Problem it solves

Score-based DAG / Bayesian-network structure learning from a data matrix `X ∈ R^{n×d}`. Model `X`
by a linear SEM `X_j = w_j^T X + z_j` (noise `z` not assumed Gaussian), so the weighted adjacency
matrix `W ∈ R^{d×d}` is the object to learn (column `j` = incoming weights of node `j`; `W_{ij} ≠ 0`
means `i → j`). Fitting `W` is easy and statistically consistent under the usual SEM assumptions
for the least-squares score; the obstacle is that "`G(W)` is acyclic" is a
combinatorial constraint over a superexponentially large discrete space, making the problem NP-hard
and forcing prior methods into discrete search with poor scaling and in-degree assumptions.

## Key idea: a smooth, exact acyclicity function

For a binary adjacency matrix `B`, `(B^k)_{ii}` counts length-`k` closed walks through node `i`, so
`tr(B^k)` counts all length-`k` closed walks; `B` is acyclic ⟺ `tr(B^k) = 0` for all `k ≥ 1`. Two
ways to bundle this into one scalar fail:

- `tr((I − B)^{-1}) = d` (Neumann series) — valid only when spectral radius `r(B) < 1`, a strong
  condition that is nontrivial to project onto.
- `Σ_{k=1}^d tr(B^k) = 0` (finite series) — valid for all `B`, but entries of `B^k` overflow
  machine precision even for small `d`, so values and gradients are numerically unusable.

The matrix exponential fixes both:

```
tr(e^B) = Σ_{k≥0} tr(B^k)/k! = d + Σ_{k≥1} tr(B^k)/k!   ⟹   B is a DAG ⟺ tr(e^B) = d.
```

`e^B` converges for every square matrix (no `r(B) < 1` needed), each term stays nonnegative for
binary `B` (so the equivalence holds), and the `1/k!` reweighting tames the magnitudes that broke
the finite series. To lift from binary `B` to a signed real `W` while keeping nonnegativity and the
same support, use the Hadamard square `W ∘ W` (entries `w_{ij}^2 ≥ 0`, zero exactly where `W` is):

```
h(W) = tr(e^{W∘W}) − d ,        W is a DAG  ⟺  h(W) = 0.
```

Properties: `h` is smooth; `h(W) ≥ 0` everywhere (DAGs are its global minima); `h` quantifies
"DAG-ness" as the `1/k!`-reweighted sum of weighted closed walks, each edge weighted
`w_{ij}^2`; and it has a cheap gradient from one matrix exponential:

```
∇h(W) = (e^{W∘W})^T ∘ 2W.
```

Derivation of the gradient: `∂ tr(e^M)/∂M = (e^M)^T` (keep the transpose — `W∘W` need not be
symmetric); with `M = W∘W`, `∂M_{kl}/∂W_{ij}` is `2W_{ij}` when `(k,l)=(i,j)` and zero otherwise,
giving the formula.

## The optimization program

```
min_W  F(W) = (1/2n)‖X − XW‖_F^2 + λ‖W‖_1     subject to     h(W) = tr(e^{W∘W}) − d = 0.
```

Smooth score, smooth scalar equality constraint, real-matrix variables — nonconvex (the constraint
set is nonconvex), so solutions are stationary points. Loss gradient `∇ℓ(W) = −(1/n) X^T(X − XW)`.

Solved by the **augmented Lagrangian**:

```
L^ρ(W, α) = F(W) + (ρ/2) h(W)^2 + α h(W),
```

which reaches feasibility with finite `ρ` (the multiplier `α` does the work that `ρ → ∞` would
otherwise need). Because the constraint is the single scalar `h`, the dual derivative is `∇D(α) =
h(W*_α)`, giving **dual ascent** `α ← α + ρ h(W*_α)`; `ρ` is multiplied by 10 only when an inner
solve fails to shrink infeasibility to below `0.25·h_prev`.

**Inner subproblem.** With `ρ, α` fixed, minimize `f(w) + λ‖w‖_1`, `w = vec(W)`, `f` the smooth part.

- *Proximal quasi-Newton* (general route): descent direction `d_k = argmin_d g_k^T d + ½ d^T B_k d +
  λ‖w_k + d‖_1` with `B_k` the L-BFGS Hessian; each coordinate has the closed form `z* = −c + S(c −
  b/a, λ/a)` (`a = B_{jj}`, `b = g_j + (Bd)_j`, `c = w_j + d_j`, `S` = soft-threshold), via the
  compact low-rank L-BFGS form at `O(m)` per coordinate plus active-set shrinking.
- *Doubled-variable route* (the simple implementation): split `W = W^+ − W^-` with `W^+, W^- ≥ 0`,
  so `‖W‖_1 = 1^T(W^+ + W^-)` is **linear** and the subproblem is smooth with simple bound
  constraints — solved directly by **L-BFGS-B**, no proximal machinery. Pin the diagonal box to
  `(0,0)` to forbid self-loops.

**Thresholding.** Numerically `h(W̃) ≤ 10^{-8}`, not exactly 0; hard-threshold `W = W̃ ∘ 1(|W̃| > ω)`
to round to a clean DAG. Justified by false-discovery reduction of hard thresholding and by `h`
quantifying cyclicity (small residual cyclic mass is represented by small weighted closed-walk
contributions). The implementation uses `ω = 0.3` as a fixed post-processing threshold. With `λ = 0`
and `ω` fixed, no tuning is required beyond solver tolerances.

## Defaults

The canonical function requires `lambda1` and `loss_type` as arguments; its demo call uses
`lambda1 = 0.1` with `loss_type = 'l2'`. Built-in solver settings are outer `max_iter = 100`,
`h_tol = 1e-8`, `rho_max = 1e16`, `w_threshold = 0.3`, `ρ0 = 1`, `α0 = 0`, progress factor `0.25`,
and `ρ` multiplier `×10`. Center columns of `X` for the `l2` loss.

## Working code

Linear NOTEARS via augmented Lagrangian + doubled-variable L-BFGS-B. Convention:
`W[i, j] != 0` means `i → j`.

```python
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8,
                   rho_max=1e16, w_threshold=0.3):
    """min_W L(W;X) + lambda1*||W||_1  s.t.  h(W) = tr(e^{W∘W}) - d = 0,  augmented Lagrangian.
    Returns W_est [d, d]; W_est[i, j] != 0 means edge i -> j."""
    n, d = X.shape

    def _loss(W):
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / n * (R ** 2).sum()
            G_loss = -1.0 / n * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / n * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / n * (S - X * M).sum()
            G_loss = 1.0 / n * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        # smooth acyclicity h(W) = tr(e^{W∘W}) - d; one matrix exponential gives value + gradient
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2                       # ∇h = (e^{W∘W})^T ∘ 2W
        return h, G_h

    def _adj(w):
        # doubled variables [w_pos, w_neg] (each d^2) -> W = w_pos - w_neg
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()   # l1 = 1^T(w_pos + w_neg)
        G_smooth = G_loss + (rho * h + alpha) * G_h                       # coeff (rho*h + alpha) on ∇h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)          # center columns
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:                            # feasibility stalled -> raise penalty
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h                                    # dual ascent
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0                 # round to a clean DAG
    return W_est
```

## Why it works where prior methods stall

- **Global, not local, search:** the entire matrix `W` is updated each step, so no edge-at-a-time
  acyclicity bookkeeping and no bounded-in-degree / bounded-treewidth assumption — it handles dense
  hub (scale-free) graphs that degrade greedy local search.
- **Distribution-agnostic:** the least-squares score makes no use of the noise family, so the same
  program works for Gaussian and non-Gaussian SEM.
- **Conceptually minimal:** no graphical-model-specific machinery; the method is a smooth program
  handed to standard solvers, implementable in a few dozen lines.
- **Limitation:** the program is nonconvex, so only stationary points are guaranteed; and each `h`
  evaluation is `O(d^3)` (the matrix exponential), which is why a second-order inner solver is used
  to minimize the number of such evaluations.
