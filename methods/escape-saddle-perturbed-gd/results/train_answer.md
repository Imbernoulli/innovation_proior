Gradient descent is the workhorse of large-scale optimization because it reaches an epsilon-first-order stationary point in a dimension-free number of steps: ell(f(x_0) - f*)/epsilon^2. But in a nonconvex landscape, a first-order stationary point is not necessarily useful. It can be a saddle point or a local maximum, and in many machine-learning problems every local minimum is already global, so the real obstacle is saddles. Plain gradient descent cannot distinguish a saddle from a minimum because its gradient sensor reads zero at both, which means it can stall or crawl past saddles for an arbitrarily long time. The challenge is to escape all strict saddles and converge to a local minimum using only gradients, at the natural step size, with at most a polylogarithmic dependence on dimension.

Existing ideas each leave a gap. Cubic-regularized Newton and trust-region methods use the full Hessian and reach an epsilon-second-order stationary point quickly, but their per-step cost scales poorly with dimension. Hessian-vector-product methods soften this but still require second-order access. The gradient-only approach of adding isotropic noise at every step provably escapes saddles, but to keep the quadratic model accurate over the entire noise ball the analysis forces the step size and noise radius to scale like 1/d, yielding iteration complexity polynomial in d. Random initialization with no added noise avoids saddles almost surely, but the guarantee is purely asymptotic and provides no finite step bound. What is missing is a gradient-only algorithm that keeps the large Theta(1/ell) step and obtains a concrete, almost dimension-free rate.

The method is Perturbed Gradient Descent, or PGD. The idea is to run ordinary gradient descent with step size eta = c/ell, and to inject randomness only when the iterate appears stuck. Whenever the gradient is small and no perturbation has occurred recently, the algorithm snapshots the current point, adds a single perturbation drawn uniformly from a small ball, and then continues running plain gradient descent for a fixed number of steps. If the function value has dropped by a certified amount, the algorithm has escaped a saddle and continues. If not, the snapshot is returned as an approximate second-order stationary point. The perturbation is doing one specific job: it plants a nonzero component along the most-negative-curvature direction at the saddle. Once that component exists, deterministic gradient descent does the rest of the escaping on its own, because near a strict saddle the gradient step is power iteration on I - eta nabla^2 f, which geometrically amplifies the negative-curvature direction.

The reason this stays almost dimension-free is a geometric fact about the stuck region. For a natural step size, the set of perturbation starts that fail to escape is a curved thin pancake rather than a flat slab. One does not need to know its shape; one only needs to know that it is thin along the escape direction. A coupling argument shows that two starting points on the same line parallel to the most-negative eigenvector and separated by at least delta r / sqrt(d) cannot both remain stuck. Integrating this width over the ball, the sqrt(d) in the width exactly cancels the sqrt(d) from the ball's cross-sectional area, so the fraction of the ball that is stuck is at most delta with no surviving power of d. This is what allows the step size to stay at Theta(1/ell) and the overall iteration complexity to match the first-order rate up to log factors.

With the thresholds set as eta = c/ell, r = sqrt(c) epsilon / (chi^2 ell), g_thres = sqrt(c) epsilon / chi^2, f_thres = c sqrt(epsilon^3/rho) / chi^3, and t_thres = chi ell / (c^2 sqrt(rho epsilon)), where chi is a logarithmic factor depending on d, ell, Delta_f, epsilon, and delta, the algorithm outputs an epsilon-second-order stationary point with probability at least 1 - delta in O(ell(f(x_0) - f*)/epsilon^2 * log^4(d ell Delta_f / (epsilon^2 delta))) iterations. The large-gradient regime makes progress by the standard descent lemma, while the saddle regime makes progress through the certified function drop after each perturbation. If the objective is robustly strict-saddle, the returned point is close to a local minimum, and if a local regularity condition holds, plain gradient descent can then be used for fast linear convergence inside the basin.

```python
import numpy as np

def perturbed_gradient_descent(grad, f, x0, ell, rho, eps, c, delta, Delta_f):
    """Perturbed Gradient Descent.

    Returns an eps-second-order stationary point with high probability:
    ||grad f(x)|| <= eps and lambda_min(Hess f(x)) >= -sqrt(rho * eps).
    Iteration complexity: O(ell * (f0 - f*) / eps^2 * log^4(d ell Delta_f / (eps^2 delta))).
    """
    x = np.array(x0, dtype=float)
    d = x.size
    chi = 3.0 * max(np.log(d * ell * Delta_f / (c * eps**2 * delta)), 4.0)
    eta = c / ell
    r = (np.sqrt(c) / chi**2) * eps / ell
    g_thr = (np.sqrt(c) / chi**2) * eps
    f_thr = (c / chi**3) * np.sqrt(eps**3 / rho)
    t_thr = int(np.ceil((chi / c**2) * ell / np.sqrt(rho * eps)))

    t_noise = -t_thr - 1
    x_tilde, f_tilde = None, None
    t = 0
    while True:
        g = grad(x)
        # Small gradient and no recent perturbation -> add one isotropic kick.
        if np.linalg.norm(g) <= g_thr and t - t_noise > t_thr:
            x_tilde, f_tilde, t_noise = x.copy(), f(x), t
            y = np.random.randn(d)
            u = np.random.rand() ** (1.0 / d)
            x = x + r * u * y / np.linalg.norm(y)
        # After t_thr steps, check whether a saddle was escaped.
        if x_tilde is not None and t - t_noise == t_thr:
            if f(x) - f_tilde > -f_thr:
                return x_tilde
        x = x - eta * grad(x)
        t += 1
```
