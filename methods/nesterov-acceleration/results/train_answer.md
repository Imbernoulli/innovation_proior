We want to minimize a convex differentiable function $f$ with an $L$-Lipschitz gradient using only a first-order oracle: at a queried point we may read the value and the gradient, nothing more. The question is one of oracle complexity — how many gradient calls suffice, in the worst case over this whole function class, to certify $f(x_k)-f^\* \le \epsilon$, and in the $\mu$-strongly convex subcase how the logarithmic rate should scale with the condition number $\kappa = L/\mu$. Plain gradient descent, $x_{k+1}=x_k-(1/L)\nabla f(x_k)$, is the natural baseline: smoothness gives a one-step decrease $f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$, convexity turns the gap into a gradient-distance product, and the distance to a minimizer does not grow, so $f(x_t)-f^\* \le 2L\|x_1-x^\*\|^2/(t-1)$. The proof is clean but small: it spends one local upper model, harvests one local decrease, and forgets everything else. Backtracking line search improves constants but not the order. The trouble is that the hard examples say this is not the floor. A tridiagonal quadratic can be arranged so each gradient call exposes only one new coordinate, leaving a long tail of the minimizer invisible after $t$ queries; computing that residual tail gives a floor of order $LR^2/t^2$ for smooth convex $f$, and in the strongly convex case a geometric tail with ratio $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$, i.e. a $\sqrt{\kappa}$ floor. So gradient descent is not off by a constant — its proof is too weak by a whole factor of $t$, and by a square root of the condition number in the strongly convex regime. Momentum and conjugate gradients reach the $\sqrt{\kappa}$ scale on quadratics, but that explanation is spectral: once $f$ is not quadratic, the gradient at a momentum-shifted point is no longer governed by a fixed matrix, and an inertial step taken at the current iterate certifies no global quantity that must improve. We need the square-root geometry without a quadratic-only proof.

I propose Nesterov's accelerated gradient method. The decisive move is to stop insisting that every step decreases $f$ — the sequence is allowed to be nonmonotone — and to replace "the last step decreased enough" with a global certificate that tightens across all oracle calls. Every gradient call hands us a global lower model: by convexity $f(x) \ge f(y)+\langle \nabla f(y), x-y\rangle$, and with strong convexity we may add $\frac{\mu}{2}\|x-y\|^2$. Rather than discard these, we average them into a tracked estimating sequence: functions $\phi_k$ and weights $\lambda_k \to 0$ with $\phi_k(x) \le (1-\lambda_k)f(x) + \lambda_k \phi_0(x)$ for all $x$. If we can also maintain $f(x_k) \le \phi_k^\* := \min_x \phi_k(x)$, then evaluating at the optimum gives $f(x_k)-f^\* \le \lambda_k(\phi_0(x^\*)-f^\*)$. The rate is now exactly how fast $\lambda_k$ can be shrunk while the inequality $f(x_k)\le\phi_k^\*$ survives, not whatever a one-step descent argument can squeeze out. Taking $\phi_0$ quadratic, $\phi_0(x)=\phi_0^\*+\frac{\gamma_0}{2}\|x-v_0\|^2$, and mixing in the lower model at a point $y_k$,
$$\phi_{k+1}(x)=(1-\alpha_k)\phi_k(x)+\alpha_k\Big[f(y_k)+\langle \nabla f(y_k),x-y_k\rangle+\tfrac{\mu}{2}\|x-y_k\|^2\Big],\qquad \lambda_{k+1}=(1-\alpha_k)\lambda_k,$$
keeps the estimating-sequence inequality (the bracket is always a lower bound on $f$) and keeps $\phi_k$ quadratic, $\phi_k(x)=\phi_k^\*+\frac{\gamma_k}{2}\|x-v_k\|^2$, with $\gamma_{k+1}=(1-\alpha_k)\gamma_k+\alpha_k\mu$ and $v_{k+1}=\big[(1-\alpha_k)\gamma_k v_k+\alpha_k\mu y_k-\alpha_k\nabla f(y_k)\big]/\gamma_{k+1}$.

The method itself is forced by the maintenance condition $f(x_{k+1})\le\phi_{k+1}^\*$. Expanding $\phi_{k+1}^\*$, lower-bounding $\phi_k^\*$ at $y_k$ by convexity, and dropping the nonnegative strong-convexity piece yields
$$\phi_{k+1}^\* \ge f(y_k) - \frac{\alpha_k^2}{2\gamma_{k+1}}\|\nabla f(y_k)\|^2 + (1-\alpha_k)\Big\langle \nabla f(y_k),\ \tfrac{\alpha_k\gamma_k}{\gamma_{k+1}}(v_k-y_k)+x_k-y_k\Big\rangle.$$
Two terms are dangerous. The first has the shape of a gradient-step decrease, and it matches the smoothness inequality exactly if we set $L\alpha_k^2 = \gamma_{k+1}$; then the gradient step $x_{k+1}=y_k-(1/L)\nabla f(y_k)$ pays for the $\|\nabla f(y_k)\|^2$ term via $f(x_{k+1})\le f(y_k)-\frac{1}{2L}\|\nabla f(y_k)\|^2$. The second term has no sign and cannot be bounded, so it must be annihilated: $\frac{\alpha_k\gamma_k}{\gamma_{k+1}}(v_k-y_k)+x_k-y_k=0$, i.e.
$$y_k=\frac{\alpha_k\gamma_k v_k+\gamma_{k+1}x_k}{\gamma_k+\alpha_k\mu}.$$
This is the heart of it. The gradient is not read at the current iterate and then decorated with momentum; the certificate forces a look-ahead point — a specific blend of the current output point $x_k$ and the minimizer $v_k$ of the accumulated lower model — and the gradient is sampled there. Momentum is what survives after the hidden variables $v_k,\gamma_k$ are eliminated, and its coefficient is not an inertial guess but precisely the value that cancels the unsigned cross term. With these two choices the induction $f(x_{k+1})\le\phi_{k+1}^\*$ closes, and the rate is governed by $\lambda_k=\prod_i(1-\alpha_i)$.

In the merely convex case $\mu=0$, $L\alpha_k^2=\gamma_{k+1}$ and the quadratic recurrence make the reciprocal coefficient grow linearly; in the standard two-sequence notation this is $a_0=1$, $a_{k+1}=\big(1+\sqrt{1+4a_k^2}\big)/2$ with momentum $(a_k-1)/a_{k+1}$. Since $a_k$ grows like $k/2$, $\lambda_k$ falls like $1/k^2$, giving $f(x_k)-f^\* \le C/(k+2)^2$ ($C=2L\|y_0-x^\*\|^2$ when $L$ is known, $C=4L\|y_0-x^\*\|^2$ under the 1983 halved-step line search), which matches the tridiagonal lower bound. In the strongly convex case there is a fixed point: starting from $\gamma_0=\mu$ keeps $\gamma_k=\mu$, so $L\alpha^2=\mu$ gives $\alpha=\sqrt{\mu/L}=1/\sqrt{\kappa}$, the momentum becomes the constant $\theta=(1-\alpha)/(1+\alpha)=(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$, and $\lambda_k=(1-\alpha)^k$ delivers $f(x_k)-f^\* \le \frac{L+\mu}{2}\|x_0-x^\*\|^2 e^{-k\sqrt{\mu/L}}$ — the same square-root scale the quadratic methods and the lower bound exhibit, but now certified by accumulated lower models rather than by diagonalizing a Hessian. The 1983 potential argument, telescoping $a_k^2(f(x_k)-f^\*)+\|p_k-x_k+x^\*\|^2$ with $p_k=(a_k-1)(x_{k-1}-x_k)$ and the identity $a_{k+1}^2-a_{k+1}=a_k^2$, is the same accounting compressed, and lands directly on $C/(k+2)^2$. So this is not gradient descent plus speed: it is a coupling between a point sequence and a hidden lower-bound sequence, where the extrapolated point is exactly where the next lower model must be sampled to keep its minimum above the next function value, and the nonmonotonicity is harmless because descent of $f$ is replaced by descent of a certificate whose weight shrinks at the lower-bound rate.

```python
"""Minimal implementation of the accelerated first-order scheme.

The code follows the two-sequence form in the 1983 method and in
Nesterov's later Constant Step Schemes II/III. The objective must expose
``grad(x)`` and a positive ``L`` smoothness constant. If ``mu`` is positive,
the strongly convex fixed momentum is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Protocol

import numpy as np


class SmoothObjective(Protocol):
    L: float
    mu: float

    def grad(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass
class AcceleratedGradientState:
    x: np.ndarray
    y: np.ndarray
    a: float = 1.0


def init_state(x0) -> AcceleratedGradientState:
    x = np.asarray(x0, dtype=float).copy()
    return AcceleratedGradientState(x=x, y=x.copy())


def step(obj: SmoothObjective, state: AcceleratedGradientState) -> AcceleratedGradientState:
    """Advance one accelerated-gradient iteration."""

    if obj.L <= 0:
        raise ValueError("obj.L must be positive")
    if getattr(obj, "mu", 0.0) < 0:
        raise ValueError("obj.mu must be nonnegative")

    x_prev = state.x
    x_next = state.y - (1.0 / obj.L) * obj.grad(state.y)

    mu = getattr(obj, "mu", 0.0)
    if mu > 0.0:
        if mu > obj.L:
            raise ValueError("obj.mu must be no larger than obj.L")
        root_kappa = sqrt(obj.L / mu)
        momentum = (root_kappa - 1.0) / (root_kappa + 1.0)
        a_next = state.a
    else:
        a_next = (1.0 + sqrt(1.0 + 4.0 * state.a * state.a)) / 2.0
        momentum = (state.a - 1.0) / a_next

    y_next = x_next + momentum * (x_next - x_prev)
    return AcceleratedGradientState(x=x_next, y=y_next, a=a_next)


def minimize(obj: SmoothObjective, x0, n_iters: int) -> np.ndarray:
    """Run ``n_iters`` iterations and return the final gradient-stepped point."""

    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")

    state = init_state(x0)
    for _ in range(n_iters):
        state = step(obj, state)
    return state.x


if __name__ == "__main__":
    class Quadratic:
        L = 4.0
        mu = 1.0

        def grad(self, x: np.ndarray) -> np.ndarray:
            return np.array([self.mu * x[0], self.L * x[1]])

    result = minimize(Quadratic(), np.array([10.0, -10.0]), 25)
    print(np.array2string(result, precision=6, suppress_small=True))
```
