We are asked to recover an unknown signal x from an underdetermined linear system y = A x + w, where A has far fewer rows than columns, so the set of exact solutions is an entire affine subspace. Without extra structure, no linear-algebraic criterion can prefer one solution over another. The textbook minimum-l2-norm answer, A^T (A A^T)^{-1} y, is actually the worst possible guess for our setting: it spreads energy uniformly across all coordinates and becomes unstable when A is ill-conditioned. It completely ignores the fact that the signals we care about are sparse, or at least sparse in a known transform such as wavelets or Fourier.

The honest formulation of the sparse-recovery problem is l0 minimization, min ||x||_0 subject to A x = y, but the l0 "norm" is a combinatorial count and minimizing it is NP-hard. Exhaustive search over supports is out of the question for the dimensions we face. Tikhonov or ridge regularization does not help either: a smooth l2 penalty shrinks coefficients but never sets any of them exactly to zero, so it produces a dense estimate. Basis pursuit, which replaces l0 by the convex l1 norm, is the right relaxation because the l1 ball is a cross-polytope whose vertices sit on the coordinate axes, so its minimizers are naturally sparse. Under a restricted-isometry condition on A, the l1 solution provably coincides with the l0 solution. With noise, the practical objective is the penalized lasso problem min 1/2 ||A x - y||_2^2 + lambda ||x||_1. This objective is convex but nonsmooth, and at imaging scale interior-point methods are impossible because they require forming or factoring matrices involving A. We need a first-order method that uses only A and A^T matrix-vector products.

The method I propose is the Fast Iterative Shrinkage-Thresholding Algorithm, or FISTA. It is an accelerated proximal-gradient method for minimizing the sum of a smooth convex function and a simple nonsmooth convex function. Here the smooth part is f(x) = 1/2 ||A x - y||_2^2, whose gradient is A^T (A x - y), and the nonsmooth part is g(x) = lambda ||x||_1. The key insight is to handle each part with the tool that suits it: f is treated by its gradient, while g is handled exactly through its proximal operator. The proximal-gradient step linearizes f around the current point, adds a quadratic trust term, and minimizes the resulting model together with g. After completing the square, this becomes a gradient step followed by the prox of g evaluated at the intermediate point.

For the l1 norm, the proximal operator is soft-thresholding. Because the l1 norm is separable across coordinates, the prox decouples into n independent scalar problems min_u lambda |u| + (1/(2t))(u - v)^2. The solution is sign(v) max(|v| - t lambda, 0). The flat dead-zone |v| <= t lambda sets small coordinates exactly to zero, which is why the algorithm produces genuinely sparse iterates. The basic iteration x_k = soft(x_{k-1} - t A^T (A x_{k-1} - y), t lambda) is ISTA. It is correct and cheap, but it converges at only O(1/k) in objective value, which is often too slow in practice.

FISTA accelerates ISTA to the optimal O(1/k^2) rate without increasing the per-iteration cost. Instead of taking the prox step from the previous iterate, it takes the prox step from an extrapolated point y_k. The extrapolation is a momentum term built from the last two iterates: y_{k+1} = x_k + ((t_k - 1) / t_{k+1}) (x_k - x_{k-1}), where the scalar sequence is defined by t_1 = 1 and t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2. These particular weights are not ad hoc; they are exactly the values that force the convergence energy t_k^2 (F(x_k) - F^*) to telescope. The result is F(x_k) - F^* <= 2 L ||x_0 - x^*||_2^2 / (k + 1)^2, matching the optimal first-order rate for smooth convex problems, while still requiring only one application of A, one application of A^T, and one coordinatewise soft-threshold per iteration. The step size is t = 1 / L, where L = lambda_max(A^T A) = ||A||^2 is the Lipschitz constant of the gradient of f. If L is unknown, a few power iterations estimate it, or one can use backtracking line search.

```python
import numpy as np


def grad_smooth(A, y, x):
    """Gradient of 1/2 ||A x - y||_2^2."""
    return A.conj().T @ (A @ x - y)


def soft_threshold(v, tau):
    """Proximal operator of tau * ||.||_1."""
    mag = np.abs(v)
    if np.iscomplexobj(v):
        return np.maximum(mag - tau, 0.0) * np.exp(1j * np.angle(v))
    return np.sign(v) * np.maximum(mag - tau, 0.0)


def power_iter_lipschitz(A, n_iter=50, seed=0):
    """Estimate L = lambda_max(A^T A) = ||A||^2 by power iteration."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    v = v / np.linalg.norm(v)
    for _ in range(n_iter):
        v = A.conj().T @ (A @ v)
        n = np.linalg.norm(v)
        if n == 0.0:
            return 0.0
        v = v / n
    return float(np.real(np.vdot(v, A.conj().T @ (A @ v))))


def fista(A, y, lam, n_iter=500, L=None, x0=None):
    """Fast Iterative Shrinkage-Thresholding Algorithm for l1-regularized least squares."""
    if L is None:
        L = power_iter_lipschitz(A)
    if L <= 0.0:
        raise ValueError("L must be positive")
    t = 1.0 / L
    if x0 is None:
        x0 = np.zeros(A.shape[1], dtype=A.dtype if np.iscomplexobj(A) else float)
    x = x0.copy()
    z = x.copy()
    tk = 1.0
    for _ in range(n_iter):
        x_old = x.copy()
        x = soft_threshold(z - t * grad_smooth(A, y, z), t * lam)
        tk_old = tk
        tk = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        omega = (tk_old - 1.0) / tk
        z = x + omega * (x - x_old)
    return x
```
