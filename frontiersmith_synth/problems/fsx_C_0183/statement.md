# Deep-Sea Cable Tensioning — Step-Size Schedule for Saddle-Point GDA

## Setting
A deep-sea fiber cable network is held at equilibrium by trimming **slack adjustments**
`x ∈ R^d` (operator control) against worst-case **cross-current loads** `y ∈ R^d`
(the ocean, playing adversary). The tensioning cost is the convex–concave saddle function

```
f(x, y) = xᵀ A y + bᵀ x − cᵀ y + (μ/2)‖x‖² − (μ/2)‖y‖²      (μ > 0)
```

The operator seeks `min_x`, the current seeks `max_y`. The equilibrium is the saddle point.
The associated strongly-monotone operator (the *network disequilibrium field*) is

```
G(z) = [ ∇_x f ; −∇_y f ] = [ μx + b + A y ; μy + c − Aᵀ x ] ,   z = [x ; y] ∈ R^{2d}
```

At equilibrium `G(z*) = 0`. The residual `‖G(z)‖` measures how far the network is from balance.

## What you control
You are given a **fixed iteration budget** `T` and a **fixed starting point** `z0`. The tensioning
controller runs a fixed **gradient descent–ascent (GDA)** update

```
z_{t+1} = z_t − η_t · G(z_t) ,   t = 0 … T−1
```

Your job is to design the **step-size schedule** `η_0, …, η_{T−1}` (one non-negative real per step)
that leaves the network as balanced as possible after exactly `T` steps.

You do **not** get to place `z_T` directly — you only choose the steps; the evaluator runs the
dynamics itself. Because `T` is finite and the operator has a rotational (skew) part, no schedule
can drive the residual to zero, so this is genuinely open-ended: constant-step, decaying, and
Chebyshev-/cyclic-accelerated schedules all trade off differently.

## Program contract (isolated)
Your program is a standalone process. Read ONE JSON object (the public instance) from **stdin**
and write ONE JSON object (your answer) to **stdout**. Nothing else is read.

### Public instance JSON
```
{
  "d":  int,                 # dimension of x and of y (z has length 2d)
  "T":  int,                 # iteration budget (schedule length)
  "mu": float,               # strong-monotonicity constant μ > 0
  "A":  [[float]*d]*d,       # coupling matrix (d x d)
  "b":  [float]*d,           # linear term on x
  "c":  [float]*d,           # linear term on y
  "z0": [float]*(2d)         # start point [x0 (d) , y0 (d)]
}
```

### Answer JSON
```
{ "steps": [η_0, η_1, ..., η_{T-1}] }      # EXACTLY T finite real numbers
```

Any malformed answer (wrong length, non-numeric, non-finite, or a schedule that makes the iterate
overflow to non-finite) scores 0.

## Objective (minimize)
The evaluator recomputes the dynamics deterministically and reports the final residual
`obj = ‖G(z_T)‖`. Lower is better.

## Scoring
Let `baseline = ‖G(z0)‖` be the initial disequilibrium (the residual of the do-nothing schedule).
Per instance:

```
ratio = min(1, 0.1 · baseline / obj)
```

so the do-nothing schedule scores 0.1 and a schedule that cuts the residual by 10× scores 1.0.
The final score is the mean of `ratio` over 10 seeded instances of varying difficulty (larger
`σ_max(A)/μ` = more rotational = harder). Everything is deterministic; the harness re-runs and
requires identical results.
