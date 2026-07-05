# Rolling-Dough Robust Planning: Saddle-Point Update Schedule Under a Round Budget

## Story

A regional bakery cooperative plans daily production of `n` goods. The planner
chooses a production vector `x` (loaves / pastries per line). An adversarial market
(ingredient-price and demand shocks) chooses a perturbation vector `y` that pushes
the cooperative's cost up. The **robust-planning equilibrium** is the saddle point
of a convex-in-`x`, concave-in-`y` cost

```
L(x, y) = 1/2 x^T A x + x^T B y - 1/2 y^T C y + b^T x - c^T y
```

with `A, C` symmetric positive definite (strongly convex in `x`, strongly concave
in `y`) and `B` the cross-coupling ("which shock hits which line"). Stacking
`z = (x, y)`, the first-order optimality (KKT) conditions become a linear
**operator equation**

```
F(z) := [ grad_x L ; -grad_y L ] = M z + d = 0,
M = [[A, B], [-B^T, C]],   d = [b; c].
```

`M` has a positive-definite symmetric part, so `F` is **strongly monotone** — but
`M` is **non-symmetric** (the coupling `B` gives it a complex spectrum), so the
planning dynamics *rotate* and a naive gradient descent-ascent crawls toward `z*`.

## The budget (this is the scored resource)

The cooperative can run only a **fixed number `K` of planning rounds**. Each round
costs exactly **one operator evaluation** `F(z) = M z + d` (one supply/demand
response). There is **no wall clock**; the cost is the round count, hard-capped at
`K`. Your job is to **design the update rule** that makes the optimality residual
`||F(z_K)||_2` as small as possible within `K` rounds.

The evaluator runs a **fixed** one-new-`F`-per-round template; you supply the
per-round scalar coefficients:

```
z_{k+1} = z_k - alpha_k * F(z_k) - beta_k * F(z_{k-1}) + gamma_k * (z_k - z_{k-1})
for k = 0 .. K-1,   with   z_{-1} = z_0   and   F(z_{-1}) = F(z_0).
```

This template spans gradient descent-ascent (`beta=gamma=0`), optimistic GDA
(`beta = -alpha`), Polyak heavy-ball momentum (`gamma>0`), and any schedule of
them. It **cannot** jump to `z*`: after `K` rounds the error `z_K - z*` is a
degree-`≤K` polynomial in `M` applied to `z_0 - z*`, so for `K < dim` the residual
is bounded away from zero (a Krylov / Chebyshev limit). Choosing good coefficients
is the whole game.

## Candidate program (stdin → stdout, isolated subprocess)

Read ONE JSON **public instance** from stdin, write ONE JSON answer to stdout.

### Public instance (stdin)
```json
{
  "name": "bakery101",
  "block_size": 6,           // n
  "dim": 12,                 // 2n
  "budget": 6,               // K rounds = K operator evaluations
  "M": [[...], ...],         // dim x dim operator matrix (row-major)
  "d": [...],                // length-dim offset
  "z0": [...],               // length-dim start point (all zeros)
  "ref_step": 0.0123         // the conservative reference GDA step
}
```

### Answer (stdout)
```json
{"alpha": [a0, ..., a_{K-1}],
 "beta":  [b0, ..., b_{K-1}],
 "gamma": [g0, ..., g_{K-1}]}
```
Each list must contain **exactly `K` finite floats** with `|value| <= 1e6`.

A wrong shape, a non-finite value, an out-of-range coefficient, a crash, a timeout,
or a schedule that drives the iterate non-finite → that instance scores **0.0**.

## Objective and scoring (deterministic, minimize)

The evaluator recomputes, in its own process, the final residual
`q_cand = ||F(z_K)||_2` from your schedule, and the residual `q_base` of the naive
reference method (constant GDA at `ref_step`). It normalizes

```
r = min(1.0, 0.1 * q_base / max(q_cand, 1e-12))
```

- Matching the naive baseline scores `~0.1`.
- A worse (diverging) schedule scores `< 0.1` → down to `0`.
- Even the theoretically optimal degree-`K` method leaves headroom: `r` stays
  strictly below `1.0` on every instance. **There is no easy optimum.**

The final score is the mean of `r` over 10 seeded instances (including harder,
larger held-out ones). Everything is deterministic — the harness re-runs and
requires an identical `Ratio` and `Vector`.

## Strategy ladder (increasing quality)

1. **Naive** — constant GDA at `ref_step` (`beta=gamma=0`): reproduces the
   baseline, `~0.1`.
2. **Greedy** — tune only the constant step by a 1-D grid search: modestly better.
3. **Strong** — constant optimistic-GDA + momentum, three scalars tuned by a
   deterministic grid search over the exact budget: optimism kills the rotation,
   momentum accelerates the contraction — far below baseline, still short of
   optimal.
4. **Frontier** — per-round (non-constant) schedules toward the Krylov / Chebyshev
   optimal degree-`K` polynomial method (GMRES-style least squares over
   `M^j (z_0 - z*)`): the analytic ceiling that still leaves headroom below 1.0.
