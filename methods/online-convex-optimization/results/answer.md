# Online Convex Optimization & Online Gradient Descent (Greedy Projection)

## The problem

Repeatedly commit a point in a fixed, known, closed convex set `F ⊆ ℝⁿ`; only
*after* committing `x^t` does an adversary reveal that round's convex cost `c^t`.
No statistical assumption is placed on the cost sequence, and no relation is
assumed between successive costs. Performance is **regret** versus the single best
fixed point in hindsight,

`R(T) = Σ_{t=1}^T c^t(x^t) − min_{x*∈F} Σ_{t=1}^T c^t(x*)`,

and the demand is **sublinear** regret, so that average regret `R(T)/T → 0`.

## The key idea

Three previously separate online-learning results — prediction with expert advice
(weighted majority / Hedge), online regression (gradient / exponentiated-gradient
updates), and gradient ascent in repeated games (infinitesimal gradient ascent,
proven only for `2×2`) — are all instances of *playing a point in a convex set
against an adversarially chosen convex loss*. In that single setting the obvious
algorithm works: when each loss arrives, take one projected gradient-descent step
on it. Convexity replaces each curved loss by its tangent from below (so beating
**linear** losses suffices), the squared distance to the comparator is a potential
that telescopes, and Euclidean projection is non-expansive toward any feasible
point (so the constraint set is free). With step size `η_t = Θ(1/√t)` this attains
`O(√T)` regret — tight up to constants, by a matching `Ω(√T)` lower bound — with no assumptions
on the losses.

## The algorithm — Online Gradient Descent / Greedy Projection

Choose any `x^1 ∈ F` and step sizes `η_t > 0`. On round `t`:

1. Play `x^t`, then observe the convex cost `c^t`.
2. Let `g^t = ∇c^t(x^t)` (a subgradient suffices).
3. Update and project: `x^{t+1} = P( x^t − η_t g^t )`, where
   `P(y) = argmin_{x∈F} ‖x − y‖` is Euclidean projection onto `F`.

For a repeated game one *ascends* the utility instead — `x^{t+1} = P(x^t + η_t ∇u)` —
which is Online Gradient Descent on the loss `−u`; this is Generalized
Infinitesimal Gradient Ascent (GIGA), valid for any number of actions.

## The regret bound, proved

Let `D = max_{x,y∈F} ‖x − y‖` be the diameter and `G` bound the gradient norms,
`‖∇c^t(x)‖ ≤ G` on `F`. Fix any comparator `x* ∈ F`, write `g^t = ∇c^t(x^t)`.

**Linearize by convexity.** For convex `c^t`,
`c^t(x^t) − c^t(x*) ≤ g^t·(x^t − x*)`. Hence `R(T) ≤ Σ_t g^t·(x^t − x*)`; it
suffices to bound the linear losses.

**One step of the potential** `‖x^t − x*‖²`. With `y^{t+1} = x^t − η_t g^t`,
`‖y^{t+1} − x*‖² = ‖x^t − x*‖² − 2η_t g^t·(x^t − x*) + η_t² ‖g^t‖²`.
Projection is non-expansive toward `x* ∈ F` (Pythagoras),
`‖x^{t+1} − x*‖² = ‖P(y^{t+1}) − x*‖² ≤ ‖y^{t+1} − x*‖²`, and `‖g^t‖ ≤ G`, so

`g^t·(x^t − x*) ≤ (‖x^t − x*‖² − ‖x^{t+1} − x*‖²)/(2η_t) + (η_t/2) G²`.

**Sum and telescope.** With `η_t` non-increasing, Abel summation of the potential
differences (dropping `−‖x^{T+1} − x*‖²/(2η_T) ≤ 0`, bounding each
`‖x^t − x*‖² ≤ D²`) collapses to `D²/(2η_T)`:

`R(T) ≤ Σ_t g^t·(x^t − x*) ≤ D²/(2η_T) + (G²/2) Σ_{t=1}^T η_t`.

**Choose the step size.** The first term wants large steps, the second small ones;
`η_t = 1/√t` balances them, since `1/η_T = √T` and
`Σ_{t=1}^T 1/√t ≤ 1 + ∫_1^T t^{-1/2}dt = 2√T − 1`. Scaling the schedule by
`D/G` gives the clean statement:

> **Theorem.** With `η_t = D/(G√t)`, Online Gradient Descent guarantees
> `R(T) ≤ (3/2) GD√T` for all `T ≥ 1`. Equivalently, with a horizon-tuned constant
> step `η = D/(G√T)`, `R(T) ≤ GD√T`.

(Equivalently, in diameter/gradient notation `‖F‖, ‖∇c‖` and `η_t = t^{-1/2}`,
`R(T) ≤ (‖F‖²/2)√T + (√T − 1/2)‖∇c‖²`.) Both forms give `R(T)/T = O(1/√T) → 0`.

**Lower bound.** On the cube `‖x‖_∞ ≤ 1` with random `±1` linear costs
`f_t(x) = v_t·x`, every online point has `E[f_t(x_t)] = 0`, while the best fixed
corner satisfies `E[min_x Σ_t v_t·x] = −Θ(n√T)` (`±1` random walks). Since
`D = 2√n`, `G = √n`, this is `Ω(DG√T)` regret for *any* algorithm — so `√T` is
tight and Online Gradient Descent has the correct worst-case order.

**Special cases.** On the simplex with linear costs this gives `O(√T)` regret in
the experts problem (Euclidean-potential analogue of weighted majority, paying the
diameter instead of `log n`); applied to mixed strategies it makes gradient
ascent in repeated games **universally consistent** for any number of actions.

**Dynamic regret.** Against a moving comparator sequence `z^1, …, z^T` with path
length `Σ_{t=2}^T ‖z^{t-1} − z^t‖ ≤ L`, fixed `η` gives
`R(T, L) ≤ 7D²/(4η) + LD/η + TηG²/2`. The `7D²/(4η)` term is the fixed boundary
piece from telescoping the moving potential; `LD/η` is the price of comparator
movement.

## Code

```python
import numpy as np

def online_gradient_descent(project, reveal_cost, x1, T, step_size=None):
    """Greedy Projection / Online Gradient Descent.

    project(y) -> closest point in the fixed convex set F to y (Euclidean,
                  non-expansive toward any point already in F).
    reveal_cost(t, x) -> round t's convex cost after x has been committed.
    x1                -> arbitrary feasible start in F.
    step_size(t)      -> e.g. 1/sqrt(t), D/(G*sqrt(t)), or D/(G*sqrt(T)).

    Regret vs the best fixed x* in F is O(D G sqrt(T)), with no statistical or
    inter-round assumption on the cost sequence.
    """
    x = np.array(x1, dtype=float)
    plays = []
    for t in range(1, T + 1):
        play = x.copy()                  # commit x^t before seeing c^t
        c_t = reveal_cost(t, play)       # now the adversary reveals c^t
        g = np.asarray(c_t.grad(play), dtype=float)

        # eta_t balances D^2/(2 eta_T) against (G^2/2) sum eta_t.
        eta = step_size(t) if step_size is not None else 1.0 / np.sqrt(t)

        y = play - eta * g               # gradient step:  y^{t+1} = x^t - eta g^t
        x = project(y)                   # project back:   x^{t+1} = P(y^{t+1})
        plays.append(play)
    return plays


# Example: experts / simplex, projection onto the probability simplex.
def project_simplex(y):
    """Euclidean projection of y onto {x : x >= 0, sum x = 1}."""
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(y)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(y - theta, 0.0)
```
