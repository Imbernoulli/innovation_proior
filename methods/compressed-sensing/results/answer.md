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
$A^{\top}$). If $L$ is unknown, a backtracking line search shrinks a trial step until the
quadratic model majorizes $f$.

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


def smooth_value(A, y, x):
    r = A @ x - y
    return 0.5 * float(np.real(np.vdot(r, r)))


def grad_smooth(A, y, x):
    """Gradient of 1/2||Ax-y||^2: one application of A then one of the adjoint."""
    return A.conj().T @ (A @ x - y)


def soft_threshold(v, thresh):
    """Prox of lam*||.||_1 at step t, with thresh = t*lam."""
    mag = np.abs(v)
    if np.iscomplexobj(v):
        return np.maximum(mag - thresh, 0.0) * np.exp(1j * np.angle(v))
    return np.sign(v) * np.maximum(mag - thresh, 0.0)


def lipschitz_constant(A, n_iter=100, seed=0):
    """L = ||A||^2 = lambda_max(A.conj().T @ A), so the largest fixed step is 1/L."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1])
    norm = np.linalg.norm(x)
    if norm == 0.0:
        raise ValueError("power iteration initialized at zero")
    x = x / norm
    for _ in range(n_iter):
        x = A.conj().T @ (A @ x)
        norm = np.linalg.norm(x)
        if norm == 0.0:
            return 0.0
        x = x / norm
    return float(np.real(np.vdot(x, A.conj().T @ (A @ x))))


def _backtracking(point, step, A, y, lam, beta=0.5, n_back=100):
    grad = grad_smooth(A, y, point)
    f_point = smooth_value(A, y, point)
    for _ in range(n_back):
        cand = soft_threshold(point - step * grad, step * lam)
        d = cand - point
        q = f_point + np.real(np.vdot(grad, d)) + (0.5 / step) * np.real(np.vdot(d, d))
        if smooth_value(A, y, cand) <= q:
            break
        step *= beta
    return cand, step


def proximal_gradient_l1(A, y, lam, x0=None, step=None, n_iter=500,
                         backtracking=False, beta=0.5, n_back=100,
                         acceleration=None):
    """Proximal gradient for 1/2||Ax-y||^2 + lam*||x||_1.

    The loop follows the standard solver layout:
    x <- prox_g(z - step*grad_smooth(z)), then z <- x + omega*(x - x_old).
    """
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    if step is None:
        backtracking = True
        step = 1.0
    if acceleration not in (None, "fista", "vandenberghe"):
        raise ValueError("acceleration must be None, 'fista', or 'vandenberghe'")

    x = x0.copy()
    z = x.copy()
    tk = 1.0
    for i in range(n_iter):
        x_old = x.copy()
        if backtracking:
            x, step = _backtracking(z, step, A, y, lam, beta=beta, n_back=n_back)
        else:
            x = soft_threshold(z - step * grad_smooth(A, y, z), step * lam)

        if acceleration == "fista":
            tk_old = tk
            tk = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
            omega = (tk_old - 1.0) / tk
        elif acceleration == "vandenberghe":
            omega = i / (i + 3.0)
        else:
            omega = 0.0
        z = x + omega * (x - x_old)
    return x


def ista(A, y, lam, n_iter=500, L=None):
    """Iterative shrinkage-thresholding for 1/2||Ax-y||^2 + lam*||x||_1."""
    if L is None:
        L = lipschitz_constant(A)
    if L <= 0.0:
        raise ValueError("L must be positive")
    step = 1.0 / L
    return proximal_gradient_l1(A, y, lam, step=step, n_iter=n_iter)


def fista(A, y, lam, n_iter=500, L=None):
    """Accelerated proximal gradient for 1/2||Ax-y||^2 + lam*||x||_1."""
    if L is None:
        L = lipschitz_constant(A)
    if L <= 0.0:
        raise ValueError("L must be positive")
    step = 1.0 / L
    return proximal_gradient_l1(A, y, lam, step=step, n_iter=n_iter, acceleration="fista")
```

The accelerated proximal-gradient recursion above is the standard one: a gradient step on
the smooth least-squares term, the $\ell_1$ prox (soft-threshold), and an extrapolation
weight selected by `acceleration`; the `acceleration="fista"` branch uses
$(t_k-1)/t_{k+1}$ after updating $t_{k+1}=(1+\sqrt{1+4t_k^2})/2$.
