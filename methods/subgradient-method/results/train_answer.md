We need to minimize a convex function $f:\mathbb{R}^n\to\mathbb{R}$ that is not necessarily differentiable. This is the natural shape of a great many large-scale problems: maxima of affine or smooth pieces, $\ell_1$ and absolute-value penalties, Lagrangian dual functions arising from decomposition, maximum-eigenvalue objectives, exact penalty functions, and feasibility objectives built from distances to convex sets. In all of these the minimizer can sit exactly at a kink, where an ordinary gradient simply does not exist. What I want is an algorithmic primitive that uses only first-order information at the current point, that does not require solving a growing auxiliary linear or quadratic program, and that comes with a worst-case guarantee for a whole Lipschitz convex class rather than for a special smooth model.

The obvious moves all fall short. Smooth gradient descent rests on one fact: $-\nabla f(x)$ is a descent direction, so a small enough step or a line search reduces the objective. That fact is gone the moment the first-order object is a set. The directional derivative of a convex function in direction $v$ is $\sup_{h\in\partial f(x)} h^\top v$, a supremum over all supporting slopes, not a single inner product; so if I pick one slope $g$ and march along $-g$, another legitimate slope in the same subdifferential can make the directional derivative positive and the objective can rise even infinitesimally. This is not a too-large-step nuisance, it is a structural failure of the descent-direction idea, and it also poisons any line search that assumes the chosen ray contains a lower value nearby. The conservative alternatives have their own costs: cutting-plane and localization methods reuse the same supporting inequalities but accumulate an expanding model and solve an auxiliary master problem at every step; second-order and interior-point methods are fast when a smooth barrier or Hessian is available but do not handle a kinked black-box Lipschitz objective directly; and smoothing only trades the original problem for one whose accuracy depends on the smoothing parameter. The design pressure is therefore precise: find the simplest current-point update whose analysis does not require objective descent, and choose step sizes so the genuine progress eventually dominates the accumulated overshoot.

The method I propose is the subgradient method, and what makes it work is a change in what the analysis watches. I start from the one object that survives when the gradient disappears: a vector $g$ is a subgradient at $x$ if $f(z)\ge f(x)+g^\top(z-x)$ for every $z$, i.e. $(g,-1)$ is normal to a hyperplane supporting the epigraph at $(x,f(x))$. At a differentiable point this set is the single ordinary gradient; at a kink it is a whole closed convex set, and convex calculus makes a member cheap to produce — for a pointwise maximum take a gradient of any active piece, for $|x|$ at zero take any slope in $[-1,1]$, and so on. The update is then exactly the gradient-descent move with any subgradient in place of the gradient,
$$x_{k+1}=x_k-\alpha_k\,g_k,\qquad g_k\in\partial f(x_k),$$
but because $-g_k$ need not be a descent direction I do not trust the last iterate; I track and return the best value seen,
$$f_{\text{best}}^{(k)}=\min_{1\le i\le k} f(x_i).$$

The justification comes from evaluating the subgradient inequality not at an arbitrary $z$ but at a minimizer $x^\star$. Since $f(x^\star)=f^\star$, the inequality $f^\star\ge f(x_k)+g_k^\top(x^\star-x_k)$ rearranges to
$$g_k^\top(x_k-x^\star)\ge f(x_k)-f^\star.$$
This is the switch the whole method turns on: the chosen slope may not be a descent direction for the value, but it has a nonnegative component pointing away from the optimum, and that component is at least the current suboptimality. So $-g_k$ is justified spatially — it reduces distance to $x^\star$ — even when it does not reduce $f$ at the next point. To make that precise I expand the squared distance to $x^\star$,
$$\|x_{k+1}-x^\star\|^2=\|x_k-x^\star\|^2-2\alpha_k\,g_k^\top(x_k-x^\star)+\alpha_k^2\|g_k\|^2,$$
and substitute the inequality above into the cross term to get
$$\|x_{k+1}-x^\star\|^2\le\|x_k-x^\star\|^2-2\alpha_k\big(f(x_k)-f^\star\big)+\alpha_k^2\|g_k\|^2.$$
Here the structure of the trade-off is visible: the middle term is negative and linear in the step, the progress earned from being suboptimal; the last term is positive and quadratic, the price of a finite step. Small steps are safe without any descent in value, but steps cannot stay large forever if I want exact convergence. Telescoping from $i=1$ to $k$, dropping the nonnegative final distance, assuming $\|x_1-x^\star\|\le R$ and $\|g_i\|\le G$ (the Lipschitz case), and using that the weighted average of the gaps is at least the smallest gap times the total weight, gives the master bound
$$f_{\text{best}}^{(k)}-f^\star\le\frac{R^2+G^2\sum_{i=1}^k\alpha_i^2}{2\sum_{i=1}^k\alpha_i}.$$

This single inequality dictates every step-size rule. If $\sum\alpha_i$ diverges the fixed $R^2$ term washes out, and if $\sum\alpha_i^2$ is finite the accumulated overshoot stays bounded, so a square-summable but not summable schedule such as $\alpha_k=a/k$ drives $f_{\text{best}}^{(k)}\to f^\star$. A constant step $\alpha$ instead yields $R^2/(2\alpha k)+G^2\alpha/2$, which converges only to a neighborhood of size proportional to $\alpha$ — the method keeps overshooting the kink, exactly as the spatial picture predicts. If I know in advance I will take exactly $K$ steps I can minimize the bound itself: convexity makes equal steps optimal, $\alpha_i=(R/G)/\sqrt{K}$ minimizes $R^2/(2K\alpha)+G^2\alpha/2$, and substituting gives
$$f_{\text{best}}^{(K)}-f^\star\le\frac{RG}{\sqrt{K}},$$
so an $\epsilon$-guarantee costs on the order of $(RG/\epsilon)^2$ iterations. One more rule hides in the same recursion: for a fixed current point the right-hand side is a quadratic in $\alpha_k$ whose minimizer, when $f^\star$ is known, is the Polyak step $\alpha_k=(f(x_k)-f^\star)/\|g_k\|^2$; it needs no $R$, $G$, or horizon, and substituting it into the telescoped sum gives $\sum_i (f(x_i)-f^\star)^2/\|g_i\|^2\le R^2$, again yielding the $RG/\sqrt{k}$ scale. When $f^\star$ is unknown an estimated-Polyak variant replaces it by the running best minus a vanishing margin $\gamma_k=\gamma_0/k$. The slow $1/\sqrt{k}$ rate is not loose analysis: Nesterov's resisting-oracle construction $\gamma\max_i x^{(i)}+(\mu/2)\|x\|^2$ reveals only one useful coordinate per query, forcing a gap of order $MR/\sqrt{k}$, so the fixed-horizon bound is worst-case optimal up to constants. The insight is never that a subgradient behaves like a gradient — it does not — but that a supporting hyperplane certifies enough alignment with the direction to the optimum to control squared distance, and that the step-size conditions are precisely what make the linear progress dominate the quadratic overshoot over time.

```python
"""Minimal subgradient method implementation.

This is an executable form of the Shor/Polyak subgradient scheme for
unconstrained convex minimization.  It deliberately keeps the best value seen,
because a negative subgradient step is not generally a descent step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


StepRule = Literal["square_summable", "fixed_horizon", "polyak", "polyak_est"]


@dataclass
class SubgradientResult:
    x_best: np.ndarray
    f_best: float
    x_last: np.ndarray
    values: list[float]
    best_values: list[float]


def subgradient_method(
    f: Callable[[np.ndarray], float],
    subgrad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    iterations: int,
    step_rule: StepRule = "square_summable",
    *,
    f_star: float | None = None,
    G: float | None = None,
    R: float | None = None,
    alpha0: float = 1.0,
    gamma0: float = 1.0,
) -> SubgradientResult:
    """Run a basic subgradient method.

    Args:
        f: Convex objective.
        subgrad: Oracle returning any element of the subdifferential at x.
        x0: Initial point.
        iterations: Number of oracle calls.
        step_rule: One of:
            square_summable: alpha_k = alpha0 / k.
            fixed_horizon: alpha_k = (R / G) / sqrt(iterations).
            polyak: alpha_k = (f(x_k) - f_star) / ||g_k||^2.
            polyak_est: alpha_k = (f(x_k) - f_best + gamma0/k) / ||g_k||^2.
        f_star: Required for the Polyak rule.
        G, R: Required for the fixed-horizon rule.
        alpha0: Scale for the square-summable rule.
        gamma0: Scale for the estimated-Polyak rule.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    x = np.array(x0, dtype=float)
    x_best = x.copy()
    f_best = float("inf")
    values: list[float] = []
    best_values: list[float] = []

    for k in range(1, iterations + 1):
        fx = float(f(x))
        values.append(fx)
        if fx < f_best:
            f_best = fx
            x_best = x.copy()
        best_values.append(f_best)

        g = np.array(subgrad(x), dtype=float)
        g_norm_sq = float(g @ g)
        if g_norm_sq == 0.0:
            break

        if step_rule == "square_summable":
            alpha = alpha0 / k
        elif step_rule == "fixed_horizon":
            if G is None or R is None:
                raise ValueError("G and R are required for fixed_horizon")
            alpha = (R / G) / np.sqrt(iterations)
        elif step_rule == "polyak":
            if f_star is None:
                raise ValueError("f_star is required for polyak")
            alpha = max(0.0, (fx - f_star) / g_norm_sq)
        elif step_rule == "polyak_est":
            gamma_k = gamma0 / k
            alpha = max(0.0, (fx - f_best + gamma_k) / g_norm_sq)
        else:
            raise ValueError(f"unknown step rule: {step_rule}")

        x = x - alpha * g

    return SubgradientResult(x_best, f_best, x, values, best_values)


def make_piecewise_linear(A: np.ndarray, b: np.ndarray):
    """Return f(x)=max_i a_i^T x + b_i and an active-row subgradient oracle."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    def f(x: np.ndarray) -> float:
        return float(np.max(A @ x + b))

    def subgrad(x: np.ndarray) -> np.ndarray:
        active = int(np.argmax(A @ x + b))
        return A[active].copy()

    return f, subgrad
```
