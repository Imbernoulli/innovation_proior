The task that forces every design choice is recovering a signal or image from indirect, noisy linear measurements $b = Ax + \text{noise}$, where $A$ is a known linear operator and $x$ is the unknown object — for a wavelet-based deblurring problem, $A$ is a blur composed with a synthesis transform and $x$ stores coefficients. Pure least squares is unstable when the inverse problem is ill-conditioned, so I add a sparsity-promoting absolute-value penalty and end up minimizing $F(x)=\|Ax-b\|_2^2 + \lambda\|x\|_1$. The objective has two visibly different pieces: a smooth data-fit term that couples all coordinates through $A$, and a kinked $\ell_1$ term that acts coordinate by coordinate. The instances are image-scale — hundreds of thousands to millions of variables, with $A$ applied only implicitly through matrix-vector products — which rules out dense factorizations, cone-program reductions, and repeated exact linear solves. So I need a first-order method whose every iteration costs only a few products with $A$ and $A^\top$ plus cheap vector work, and because many such iterations may be needed, it must come with a nonasymptotic function-value rate, not merely an asymptotic convergence proof. The two obvious routes both fail. Casting the whole sum as a cone program and applying a Newton-type method demands large dense linear solves I cannot afford. Treating the whole sum as one nonsmooth convex function and running a subgradient method converges too slowly and throws away the curvature of the quadratic data term by treating the smooth and nonsmooth parts as an undifferentiated lump. The winning idea has to keep the two pieces apart: exploit the differentiable part fully without ever pretending the kinked penalty has a gradient.

I propose proximal gradient descent — ISTA for the basic scheme and FISTA for its accelerated variant. Write the problem as $\min_x f(x)+g(x)$ with $f$ convex differentiable with $L(f)$-Lipschitz gradient and $g$ closed proper convex but possibly nonsmooth with a tractable proximal map. The single structural move is to majorize only the smooth part. Since $\nabla f$ is $L$-Lipschitz, the descent lemma gives the global quadratic upper model $f(x)\le f(y)+\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|_2^2$, and I build a surrogate that uses this for $f$ while leaving $g$ exact:
$$Q_L(x,y)=f(y)+\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|_2^2+g(x).$$
For $L\ge L(f)$ this dominates $F$ and equals $F(y)$ at $x=y$, so minimizing it is a genuine majorization step — descent without ever smoothing or linearizing the kink. Its minimizer is the next iterate $p_L(y)=\arg\min_x Q_L(x,y)$. Completing the square in the $x$-dependent terms, $\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|_2^2=\tfrac{L}{2}\|x-(y-\nabla f(y)/L)\|_2^2-\tfrac{1}{2L}\|\nabla f(y)\|_2^2$, the constant drops out and the update collapses to a forward gradient step followed by a backward proximal step:
$$p_L(y)=\operatorname{prox}_{(1/L)g}\!\big(y-\tfrac{1}{L}\nabla f(y)\big),\qquad \operatorname{prox}_{tg}(v)=\arg\min_x g(x)+\tfrac{1}{2t}\|x-v\|_2^2.$$
This is the load-bearing split, and it is correct in the sense that its fixed points are exactly the minimizers of $F$. A minimizer $x^\*$ satisfies $0\in\nabla f(x^\*)+\partial g(x^\*)$, which rearranges to $(x^\*-t\nabla f(x^\*))-x^\*\in t\,\partial g(x^\*)$; the optimality condition for $u=\operatorname{prox}_{tg}(v)$ is $v-u\in t\,\partial g(u)$, and with $v=x^\*-t\nabla f(x^\*)$, $u=x^\*$ the two coincide. So the proximal step is not a heuristic bolted on after the gradient step — it is the exact minimizer of the nonsmooth subproblem given that step.

For the lasso/deblurring case the proximal map is concrete. With $g(x)=\lambda\|x\|_1$ the prox separates into scalar problems $\min_z \lambda|z|+\tfrac{1}{2t}(z-v_i)^2$. For $z>0$ the stationarity condition gives $z=v_i-\lambda t$, valid only when $v_i>\lambda t$; for $z<0$ it gives $z=v_i+\lambda t$, valid only when $v_i<-\lambda t$; and at $z=0$ the subgradient condition $0\in\lambda[-1,1]-v_i/t$ holds exactly when $|v_i|\le\lambda t$. The three cases combine into coordinatewise soft-thresholding
$$T_\tau(v)_i=\operatorname{sign}(v_i)\,\max(|v_i|-\tau,0).$$
With $f(x)=\|Ax-b\|_2^2$ we have $\nabla f(x)=2A^\top(Ax-b)$ and $L(f)=2\lambda_{\max}(A^\top A)$, so the basic iteration becomes
$$x_{k+1}=T_{\lambda/L}\!\big(x_k-\tfrac{1}{L}\,2A^\top(Ax_k-b)\big),$$
which is ISTA: apply $A$, apply $A^\top$, shrink coordinates — exactly the per-iteration budget the large problem allows. The rate follows from one engine inequality. When the accepted $L$ satisfies $F(p_L(y))\le Q_L(p_L(y),y)$, convexity of $f$ and $g$ plus the optimality condition for $p_L(y)$ (there is $\gamma\in\partial g(p_L(y))$ with $\nabla f(y)+L(p_L(y)-y)+\gamma=0$) give, for every $x$,
$$F(x)-F(p_L(y))\ge\tfrac{L}{2}\|p_L(y)-y\|_2^2+L\langle y-x,\,p_L(y)-y\rangle.$$
Taking $y=x_n$, $p_L(y)=x_{n+1}$, $x=x^\*$ makes the right side a difference of squared distances that telescopes; taking $x=y=x_n$ gives monotone decrease. Combining yields $F(x_k)-F(x^\*)\le L_{\max}\|x_0-x^\*\|_2^2/(2k)$, an $O(1/k)$ rate matching ordinary gradient descent but now valid for the composite nonsmooth objective, with $L_{\max}=L(f)$ for fixed curvature.

Smooth convex minimization can reach $O(1/k^2)$ with auxiliary-point acceleration, and the question is whether that survives after replacing the gradient step with the composite $p_L$ step. It does, and the only change needed is where the proximal map is anchored: instead of evaluating it at the current iterate, evaluate it at an extrapolated point. This is FISTA. I keep $x_k=p_L(y_k)$ but choose the anchor so a weighted potential telescopes. Introducing weights $t_k$ and aiming to control $t_k^2(F(x_k)-F(x^\*))$, with $v_k=F(x_k)-F(x^\*)$ and $u_k=t_kx_k-(t_k-1)x_{k-1}-x^\*$, applying the engine inequality twice at the same anchor $y_{k+1}$ — once with $x=x_k$, once with $x=x^\*$ — and combining forces the weight identity $t_k^2=t_{k+1}^2-t_{k+1}$ and simultaneously pins down the anchor. Solving the recurrence gives
$$t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},\qquad y_{k+1}=x_k+\frac{t_k-1}{t_{k+1}}(x_k-x_{k-1}),$$
with $y_1=x_0$, $t_1=1$. The momentum coefficient $(t_k-1)/t_{k+1}$ is therefore not tunable — it is precisely the value that makes the combined inequality become $\tfrac{2}{L_k}t_k^2v_k-\tfrac{2}{L_{k+1}}t_{k+1}^2v_{k+1}\ge\|u_{k+1}\|_2^2-\|u_k\|_2^2$, a telescoping potential. Since the recurrence also forces $t_k\ge(k+1)/2$, this gives
$$F(x_k)-F(x^\*)\le\frac{2L_{\max}\|x_0-x^\*\|_2^2}{(k+1)^2},$$
the accelerated $O(1/k^2)$ rate. FISTA uses the same primitives — one smooth gradient and one proximal shrinkage per step — and is not necessarily monotone in $F(x_k)$, because the extrapolated point can overshoot; what decreases is the proof potential, and that is what the guarantee rests on.

When $L(f)$ is unknown I add backtracking. Starting from the previous accepted curvature $L_{k-1}$, take the smallest integer $i_k\ge0$ such that $\bar L=\eta^{i_k}L_{k-1}$ satisfies $F(p_{\bar L}(y))\le Q_{\bar L}(p_{\bar L}(y),y)$, then set $L_k=\bar L$ and use $p_{L_k}(y)$ with $y=x_{k-1}$ for ISTA and $y=y_k$ for FISTA. The descent lemma guarantees acceptance once the tested curvature reaches $L(f)$, and the same proofs go through with $L_{\max}=\eta L(f)$ when $L_0\le L(f)$, or $L_{\max}=\max(L_0,\eta L(f))$ otherwise. The construction is thus a complete composite first-order recipe: split the objective, take the explicit gradient step on the smooth part, solve the nonsmooth proximal shrinkage exactly, and optionally accelerate by moving the anchor while keeping the same proximal subproblem.

```python
"""Reference ISTA/FISTA for l1-regularized least squares.

The objective is ||A x - b||_2^2 + lam ||x||_1, matching the
Beck-Teboulle convention where grad f(x) = 2 A.T @ (A @ x - b).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolveResult:
    x: np.ndarray
    history: list[float]


def soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """prox_{threshold ||.||_1}(v)."""
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def objective(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    residual = A @ x - b
    return float(residual @ residual + lam * np.linalg.norm(x, 1))


def gradient(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 2.0 * (A.T @ (A @ x - b))


def lipschitz_constant(A: np.ndarray) -> float:
    return 2.0 * float(np.linalg.norm(A, 2) ** 2)


def _initial_x(A: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
    if x0 is None:
        return np.zeros(A.shape[1], dtype=float)
    return np.array(x0, dtype=float, copy=True)


def ista(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    *,
    x0: np.ndarray | None = None,
    L: float | None = None,
    steps: int = 100,
) -> SolveResult:
    """Run ISTA: gradient step on f, proximal shrinkage on lam ||.||_1."""
    if L is None:
        L = lipschitz_constant(A)
    x = _initial_x(A, x0)
    history: list[float] = []

    for _ in range(steps):
        x = soft_threshold(x - gradient(A, b, x) / L, lam / L)
        history.append(objective(A, b, x, lam))

    return SolveResult(x=x, history=history)


def fista(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    *,
    x0: np.ndarray | None = None,
    L: float | None = None,
    steps: int = 100,
) -> SolveResult:
    """Run FISTA with Beck-Teboulle's Nesterov sequence."""
    if L is None:
        L = lipschitz_constant(A)
    x = _initial_x(A, x0)
    y = x.copy()
    t = 1.0
    history: list[float] = []

    for _ in range(steps):
        x_old = x.copy()
        x = soft_threshold(y - gradient(A, b, y) / L, lam / L)
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x + ((t - 1.0) / t_next) * (x - x_old)
        t = t_next
        history.append(objective(A, b, x, lam))

    return SolveResult(x=x, history=history)


def _demo() -> None:
    rng = np.random.default_rng(7)
    A = rng.normal(size=(40, 25)) / math.sqrt(40.0)
    true_x = np.zeros(25)
    true_x[[2, 8, 19]] = [2.0, -1.5, 0.75]
    b = A @ true_x + 0.01 * rng.normal(size=40)
    lam = 0.15
    L = lipschitz_constant(A)

    ista_result = ista(A, b, lam, L=L, steps=120)
    fista_result = fista(A, b, lam, L=L, steps=120)

    print(f"ISTA objective:  {ista_result.history[-1]:.8f}")
    print(f"FISTA objective: {fista_result.history[-1]:.8f}")
    print(f"Nonzeros ISTA/FISTA: {np.count_nonzero(ista_result.x)}/{np.count_nonzero(fista_result.x)}")


if __name__ == "__main__":
    _demo()
```
