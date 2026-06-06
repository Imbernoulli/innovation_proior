# Sparse recovery by ℓ1 minimization and proximal gradient (ISTA / FISTA)

## Problem

Recover a sparse (or transform-sparse) signal $x \in \mathbb{R}^n$ from far fewer linear
measurements than unknowns, $y = Ax + w$ with $A \in \mathbb{R}^{m\times n}$, $m \ll n$. The
system is underdetermined, so sparsity is the side information that makes recovery well
posed. Asking directly for the sparsest consistent signal,
$\min_x \|x\|_0$ s.t. $Ax=y$, is combinatorial and NP-hard.

## Key idea

Replace the nonconvex count $\|x\|_0$ by its tightest convex surrogate, the $\ell_1$ norm.
The $\ell_1$ ball is a cross-polytope whose faces lie on the coordinate axes, so its
minimizers are sparse; and under a restricted-isometry (near-isometry on sparse vectors)
condition on $A$, the $\ell_1$ solution coincides with the $\ell_0$ solution and recovers
$x$ exactly. With noise, solve the convex but **nonsmooth** penalized problem

$$ \min_x \ F(x) = \underbrace{\tfrac12\|Ax-y\|_2^2}_{f(x),\ \text{smooth}} \ +\ \underbrace{\lambda\|x\|_1}_{g(x),\ \text{nonsmooth}}. $$

At the scales of interest $A$ is huge and dense, so only $A$, $A^{\top}$ matrix–vector
products are affordable — a first-order method. **Proximal gradient** handles the split
exactly: linearize the smooth $f$, keep a quadratic trust term, and add $g$ as-is; each
step collapses to a gradient step followed by the proximal operator of $g$,

$$ x_{k} = \operatorname{prox}_{t g}\!\big(x_{k-1} - t\,\nabla f(x_{k-1})\big), \qquad \nabla f(x) = A^{\top}(Ax-y),\quad t = 1/L,\ \ L = \lambda_{\max}(A^{\top}A)=\|A\|^2. $$

For $g=\lambda\|\cdot\|_1$ the prox is **soft-thresholding** (derived per-coordinate from
$0 \in \lambda\,\partial|u| + (u-v)/t$):

$$ \operatorname{prox}_{t\lambda\|\cdot\|_1}(v)_i = \operatorname{sign}(v_i)\,\max(|v_i| - t\lambda,\,0). $$

Its flat dead-zone $|v_i|\le t\lambda$ sets small coordinates to **exactly zero** — the
algorithmic source of sparsity. This is **ISTA**, with rate $F(x_k)-F^\star = O(1/k)$.

ISTA is slow. Nesterov's optimal $O(1/k^2)$ first-order method for smooth problems uses one
gradient per step plus an extrapolated look-ahead point. Grafting that onto the prox step —
applying $\operatorname{prox}$ at $y_k$ instead of $x_{k-1}$ — gives **FISTA**:

$$ x_k = \operatorname{prox}_{t g}\!\big(y_k - t\,\nabla f(y_k)\big),\quad t_{k+1} = \frac{1+\sqrt{1+4t_k^2}}{2},\quad y_{k+1} = x_k + \frac{t_k-1}{t_{k+1}}(x_k - x_{k-1}), $$

with $y_1 = x_0$, $t_1 = 1$. The $t_k$ recursion and the momentum weight are exactly the
choices that make the convergence energy $t_k^2(F(x_k)-F^\star)$ telescope; together with
$t_k \ge (k+1)/2$ they yield the optimal

$$ F(x_k) - F(x^\star) \ \le\ \frac{2L\,\|x_0-x^\star\|_2^2}{(k+1)^2} = O(1/k^2), $$

at the same per-iteration cost as ISTA (one prox = one soft-threshold, one $A$, one
$A^{\top}$). If $L$ is unknown, a backtracking line search inflates a trial $L$ by $\eta>1$
until the quadratic model majorizes $f$.

## Algorithm

```
Input: A, y, λ, step t = 1/L  with  L = λmax(AᵀA)
ISTA:    x⁰ = 0;  repeat   xᵏ = soft( xᵏ⁻¹ − t·Aᵀ(Axᵏ⁻¹ − y),  t·λ )
FISTA:   x⁰ = 0, y¹ = x⁰, t₁ = 1
         for k = 1,2,…
             xᵏ      = soft( yᵏ − t·Aᵀ(Ayᵏ − y),  t·λ )
             t_{k+1} = (1 + √(1 + 4tₖ²)) / 2
             y_{k+1} = xᵏ + ((tₖ − 1)/t_{k+1})·(xᵏ − xᵏ⁻¹)
soft(v, τ)ᵢ = sign(vᵢ)·max(|vᵢ| − τ, 0)
```

## Code

```python
import numpy as np


def soft_threshold(v, thresh):
    """Prox of λ‖·‖₁ at step t (thresh = t·λ): shrink by `thresh`, zero out |v| ≤ thresh.
    The flat dead-zone is what makes the iterates exactly sparse."""
    return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)


def power_iteration_L(A, n_iter=100, seed=0):
    """L = ‖A‖² = λmax(AᵀA), Lipschitz constant of ∇(½‖Ax−y‖²). Largest safe step is 1/L."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1])
    x /= np.linalg.norm(x)
    for _ in range(n_iter):
        x = A.T @ (A @ x)
        x /= np.linalg.norm(x)
    return float(x @ (A.T @ (A @ x)))


def grad_f(A, y, x):
    """∇f(x) = Aᵀ(Ax − y): one application of A then one of Aᵀ."""
    return A.T @ (A @ x - y)


def ista(A, y, lam, n_iter=500, L=None):
    """Iterative shrinkage-thresholding for ½‖Ax−y‖² + λ‖x‖₁.  O(1/k)."""
    if L is None:
        L = power_iteration_L(A)
    t = 1.0 / L
    x = np.zeros(A.shape[1])
    for _ in range(n_iter):
        x = soft_threshold(x - t * grad_f(A, y, x), t * lam)   # grad step on f, then prox of g
    return x


def fista(A, y, lam, n_iter=500, L=None):
    """Accelerated (Nesterov) proximal gradient for ½‖Ax−y‖² + λ‖x‖₁.  O(1/k²)."""
    if L is None:
        L = power_iteration_L(A)
    step = 1.0 / L
    x = np.zeros(A.shape[1])
    z = x.copy()                 # look-ahead point y_k; y_1 = x_0
    tk = 1.0
    for _ in range(n_iter):
        x_new = soft_threshold(z - step * grad_f(A, y, z), step * lam)   # prox at extrapolated z
        tk_new = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0              # forced by the telescoping
        z = x_new + ((tk - 1.0) / tk_new) * (x_new - x)                  # momentum (t_k−1)/t_{k+1}
        x, tk = x_new, tk_new
    return x


def fista_backtracking(A, y, lam, n_iter=500, L0=1.0, eta=2.0, n_back=100):
    """FISTA with backtracking when L = λmax(AᵀA) is unknown/expensive.
    Inflate a trial L by η until the quadratic model majorizes f."""
    def f(x):
        r = A @ x - y
        return 0.5 * float(r @ r)

    x = np.zeros(A.shape[1])
    z = x.copy()
    tk, L = 1.0, L0
    for _ in range(n_iter):
        gz, fz = grad_f(A, y, z), f(z)
        for _ in range(n_back):
            cand = soft_threshold(z - (1.0 / L) * gz, lam / L)
            d = cand - z
            # majorization: f(cand) ≤ f(z) + ⟨∇f(z), d⟩ + (L/2)‖d‖²
            if f(cand) <= fz + gz @ d + 0.5 * L * (d @ d):
                break
            L *= eta
        x_new = cand
        tk_new = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
        z = x_new + ((tk - 1.0) / tk_new) * (x_new - x)
        x, tk = x_new, tk_new
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, m, k = 400, 120, 8                       # underdetermined: m < n; k-sparse signal
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    x_true = np.zeros(n)
    x_true[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    y = A @ x_true + 1e-3 * rng.standard_normal(m)

    x_hat = fista(A, y, lam=1e-2, n_iter=1000)
    print("nonzeros recovered:", int(np.sum(np.abs(x_hat) > 1e-6)),
          " support match:", set(np.flatnonzero(np.abs(x_hat) > 1e-6)) >= set(np.flatnonzero(x_true)))
```

The accelerated proximal-gradient recursion above is the standard one: a gradient step on
the smooth least-squares term, the $\ell_1$ prox (soft-threshold), and the Nesterov
extrapolation with weight $(t_k-1)/t_{k+1}$ — exactly what general proximal-gradient solvers
implement under the `acceleration="fista"` option.
