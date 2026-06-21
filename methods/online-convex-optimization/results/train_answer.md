The same repeated-decision problem keeps reappearing under different names, and I want to treat it once. In prediction with expert advice I commit a distribution over experts before the cost vector arrives, then compete with the best fixed expert. In a repeated game I commit a mixed strategy before the opponent's realized action fixes my payoff vector, and I want no external regret. In online prediction and regression I update a weight vector after the current loss is revealed. In every case I am choosing a point in a fixed convex feasible set, an adversarially revealed loss evaluates that point, and the honest comparison is against the single best fixed point in hindsight. The feasible set $F$ is known in advance and stays the same every round — a resource polytope, a routing polytope, a probability simplex — but on round $t$ I must commit $x^t \in F$ and only then is the round's convex cost $c^t$ revealed. There is no promise that $c^{t+1}$ resembles $c^t$; the sequence may be adversarial, strategic, and carry no distribution at all. So this is neither stochastic prediction nor offline minimization of one fixed objective.

The first thing I have to avoid is asking for the impossible. I cannot aim to minimize $c^t$ on each round, since that demands knowing $c^t$ before committing $x^t$. The meaningful target is regret: after $T$ rounds I compare my cumulative cost $\sum_{t=1}^T c^t(x^t)$ against $\min_{x \in F} \sum_{t=1}^T c^t(x)$, where the comparator may see every loss but must use one fixed feasible point. Average regret going to zero — cumulative cost exceeding the best fixed hindsight decision by only a sublinear amount — is the right notion of success, and it is exactly the no-regret yardstick from Hannan consistency and the best-expert comparison. The earlier pieces all gesture at this but none composes into a general convex update. Blackwell approachability and Hannan consistency set the game-theoretic ambition without giving a concrete optimization step. Multiplicative weights gives a concrete adversarial algorithm but only for a finite action set on a simplex with linear losses, proved through an entropic potential specialized to experts. Infinitesimal gradient ascent gives gradient dynamics in games but its convergence proof depends on the special $2 \times 2$ geometry rather than any general convex-set argument. I want one mechanism that subsumes all of these and works for any closed bounded convex $F$ admitting projection and any convex differentiable cost with bounded gradients.

I propose Online Convex Optimization solved by Online Gradient Descent — Zinkevich's Greedy Projection. The update is the simplest reaction to the only local signal I have: after $c^t$ is revealed I form the gradient $g^t = \nabla c^t(x^t)$ at the point I actually played, step against it, and project back to feasibility,
$$x^{t+1} = P\big(x^t - \eta_t\, g^t\big), \qquad P(y) = \arg\min_{x \in F}\|x - y\|,$$
where $P$ is Euclidean projection onto $F$ and a subgradient may replace the gradient. What makes this provable, and what makes it the right generalization, rests on three load-bearing choices. The first is to linearize. Convexity means each $c^t$ lies above its tangent plane, so for any comparator $x^*$, $c^t(x^*) \ge c^t(x^t) + g^t \cdot (x^* - x^t)$, which rearranges to
$$c^t(x^t) - c^t(x^*) \le g^t \cdot (x^t - x^*).$$
This is decisive: to control regret for arbitrary convex losses it suffices to control the cumulative linearized losses $\sum_t g^t \cdot (x^t - x^*)$ defined by the gradients at the played points. The curved shape of $c^t$ away from $x^t$ never has to enter the analysis, which is why a single update handles every convex cost rather than one fixed objective.

The second choice is the potential. The squared distance $\|x^t - x^*\|^2$ to the comparator is the natural candidate, and the update is built to make it telescope. Writing $y^{t+1} = x^t - \eta_t g^t$ and expanding,
$$\|y^{t+1} - x^*\|^2 = \|x^t - x^*\|^2 - 2\eta_t\, g^t \cdot (x^t - x^*) + \eta_t^2\,\|g^t\|^2,$$
the very regret term I need to sum sits in the middle. Here the projection earns its place: since $x^*$ is feasible and Euclidean projection onto a convex set is non-expansive toward feasible points, $\|x^{t+1} - x^*\| \le \|y^{t+1} - x^*\|$, so projecting both restores feasibility and only shrinks the distance, never spoiling the bound. This is why Euclidean projection is the right way to handle the constraint rather than, say, an entropic mirror map — the non-expansiveness is exactly the property the squared-distance potential needs, and it holds for any closed convex $F$. Rearranging the one-step inequality and using $\|g^t\| \le G$ gives the whole proof in one line,
$$g^t \cdot (x^t - x^*) \le \frac{\|x^t - x^*\|^2 - \|x^{t+1} - x^*\|^2}{2\eta_t} + \frac{\eta_t}{2}\,G^2.$$
The first term is a potential drop; the second is the unavoidable price of reacting only after the loss is seen.

The third choice is the step schedule, and it falls out of summing this line. With nonincreasing $\eta_t$, Abel summation bounds the telescoping potential part by $D^2/(2\eta_T)$, where $D = \max_{x,y \in F}\|x-y\|$ is the diameter, leaving
$$R(T) \le \frac{D^2}{2\eta_T} + \frac{G^2}{2}\sum_{t=1}^T \eta_t.$$
The two terms argue against each other: shrink $\eta_t$ too slowly and the accumulated response error $\frac{G^2}{2}\sum_t \eta_t$ blows up; shrink it too fast and the inverse final step $D^2/(2\eta_T)$ blows up. They balance when $\eta_t \propto 1/\sqrt{t}$, because then both $1/\eta_T$ and $\sum_t \eta_t$ grow like $\sqrt{T}$. The unscaled schedule $\eta_t = t^{-1/2}$ yields Zinkevich's exact form $R(T) \le \frac{D^2}{2}\sqrt{T} + (\sqrt{T} - \tfrac{1}{2})G^2$; the scaled anytime schedule $\eta_t = D/(G\sqrt{t})$ gives $R(T) \le \frac{3}{2}DG\sqrt{T}$; and a known horizon with constant $\eta = D/(G\sqrt{T})$ gives the cleanest $R(T) \le DG\sqrt{T}$. If $D = 0$ or $G = 0$ the problem is degenerate and regret is already zero. In every nondegenerate case average regret vanishes, depending only on diameter and gradient bound, with no stochastic model and no horizon dependence worse than $\sqrt{T}$.

This $\sqrt{T}$ order is not an artifact of the proof but the true worst-case scale. On a hypercube let each loss be a random-sign linear function; my point, committed before the sign vector is drawn, has expected loss zero, while the best fixed corner in hindsight gains on the order of $n\sqrt{T}$. With $D = 2\sqrt{n}$ and $G = \sqrt{n}$ this forces expected regret of order $DG\sqrt{T}$ for any online algorithm, so Greedy Projection is order-optimal. The same machinery explains the older pieces instead of merely coexisting with them. On the simplex with linear losses it is exactly expert advice, but proved through Euclidean distance rather than an entropic weight potential. On a mixed-strategy simplex with loss equal to negative utility it becomes projected gradient ascent on revealed utilities — Generalized Infinitesimal Gradient Ascent — giving no external regret for any number of actions and removing the $2 \times 2$ restriction, since the argument is now a convex potential inequality valid in any dimension. The genuine optimization is made online not by pretending a moving sequence of losses has one minimizer, but by changing the question to regret against the best fixed feasible point, linearizing each loss at the played point, and letting projection plus squared distance telescope.

A minimal implementation makes the online timing explicit: the projection oracle for the fixed feasible set is supplied by the caller, and the round loss is revealed through a callback that is invoked only after the play has been committed.

```python
"""Greedy Projection / Online Gradient Descent.

This is a minimal implementation artifact for Zinkevich's online convex
programming algorithm. The caller supplies the projection oracle for the fixed
convex feasible set and a callback that reveals the round loss after commitment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np


Array = np.ndarray


class ConvexCost(Protocol):
    def __call__(self, x: Array) -> float:
        ...

    def grad(self, x: Array) -> Array:
        ...


def default_step_size(t: int) -> float:
    """Zinkevich's anytime step schedule eta_t = 1 / sqrt(t)."""
    if t < 1:
        raise ValueError("round index t must be positive")
    return 1.0 / np.sqrt(t)


def online_gradient_descent(
    x1: Array,
    num_rounds: int,
    reveal_cost: Callable[[int, Array], ConvexCost],
    project: Callable[[Array], Array],
    step_size: Callable[[int], float] = default_step_size,
) -> list[Array]:
    """Run projected online gradient descent.

    Args:
        x1: Any feasible starting point in the fixed convex set F.
        num_rounds: Number of online rounds to run.
        reveal_cost: Called as reveal_cost(t, x_t) after x_t is committed.
        project: Euclidean projection oracle P(y) = argmin_{x in F} ||x - y||.
        step_size: Positive learning-rate schedule.

    Returns:
        The committed plays x^1, ..., x^T.
    """
    if num_rounds < 0:
        raise ValueError("num_rounds must be nonnegative")

    x = np.asarray(x1, dtype=float)
    plays: list[Array] = []

    for t in range(1, num_rounds + 1):
        play = x.copy()
        plays.append(play)

        cost = reveal_cost(t, play.copy())
        gradient = np.asarray(cost.grad(play), dtype=float)
        if gradient.shape != play.shape:
            raise ValueError("gradient shape must match the played point")

        eta_t = float(step_size(t))
        if not np.isfinite(eta_t) or eta_t <= 0.0:
            raise ValueError("step_size(t) must be a positive finite value")

        x = np.asarray(project(play - eta_t * gradient), dtype=float)
        if x.shape != play.shape:
            raise ValueError("projection output shape must match the played point")

    return plays


def project_simplex(y: Array) -> Array:
    """Euclidean projection onto the probability simplex."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("project_simplex expects a one-dimensional vector")
    if y.size == 0:
        raise ValueError("project_simplex expects a nonempty vector")
    if not np.all(np.isfinite(y)):
        raise ValueError("project_simplex expects finite coordinates")

    u = np.sort(y)[::-1]
    cumulative = np.cumsum(u) - 1.0
    rho_candidates = np.nonzero(u - cumulative / (np.arange(y.size) + 1) > 0)[0]
    if rho_candidates.size == 0:
        raise RuntimeError("simplex projection failed to find an active coordinate")

    rho = rho_candidates[-1]
    theta = cumulative[rho] / (rho + 1.0)
    return np.maximum(y - theta, 0.0)
```
