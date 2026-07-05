# Alpine Relay: Convergence-Budget Saddle Descent

## Story

A mountain-rescue command post runs a chain of **relay stations** up a storm-lashed ridge.
Each station passes an updated rescue plan `x` (route, load-out, triage estimates) one leg
further up, while the mountain pushes back through adverse variables `y` (storm track, terrain
risk). Balancing the plan against the worst-case environment is a **convex-concave saddle
problem**: the rescue team minimizes a cost that the environment maximizes.

Every relay **leg is one optimizer iteration**. Fuel, radio range, and daylight cap the number
of legs at a fixed budget `T`. There is **no clock** in the score — the mission is graded only
by how close to balance the relay gets: the smaller the residual imbalance (the saddle operator
norm) after exactly `T` legs, the better.

## The optimization problem

Each instance fixes a monotone affine saddle operator

```
F(z) = M z + q ,   z = (x, y) in R^d ,   d = nx + ny
M = [[ P ,  A ],
     [-A^T, Q ]] ,   P = P^T >= 0 ,  Q = Q^T >= 0
```

arising from `L(x,y) = 1/2 x^T P x + x^T A y - 1/2 y^T Q y + b^T x - c^T y` with
`F = (grad_x L, -grad_y L)`. The unique saddle point `z*` solves `F(z*) = 0`. Starting from a
fixed `z0`, you **minimize the final operator norm** `||F(z_T)||_2` after exactly `T` legs.

Every leg follows one **frozen relay template** with per-leg gains `a[t]`, `b[t]` that YOU
supply:

```
g_t     = F(z_t)
z_{t+1} = z_t - a[t] * g_t - b[t] * (g_t - g_{t-1})      (define g_{-1} := g_0)
```

- `b[t] = 0` is plain gradient descent-ascent.
- `a[t] = b[t] = const` is optimistic GDA.
- A well-designed **schedule** (Chebyshev-style acceleration, per-leg line search, momentum,
  ...) drives the residual far lower under the same budget `T`.

You are given the full `(M, q, z0, T)`, so you may simulate the trajectory yourself and output
any numeric schedule you like. The budget `T` is set well below the dimension `d`, so the
residual cannot in general be zeroed — the challenge is to squeeze it as low as possible.

## Candidate program contract (stdin -> stdout)

Read ONE JSON object (the public instance) from stdin:

```json
{"name": "relay101", "nx": 12, "ny": 12, "d": 24, "T": 8,
 "M": [[...d floats...], ... d rows ...],
 "q": [...d floats...], "z0": [...d floats...],
 "baseline_step": 0.0123}
```

Write ONE JSON object to stdout:

```json
{"a": [a_0, ..., a_{T-1}], "b": [b_0, ..., b_{T-1}]}
```

- `a` and `b` MUST each be lists of exactly `T` finite real numbers with `|value| <= 1e4`.
- `baseline_step` is the step used by the weak reference relay (plain GDA, `b = 0`); output it
  as `a` with `b = 0` to reproduce the reference exactly.

## Objective and scoring (deterministic; no wall-time)

Objective: **minimize** `q_cand = ||F(z_T)||_2` produced by your schedule. Per instance:

```
q_base = ||F(z_T)||  from the weak reference: plain GDA (a = baseline_step, b = 0) for T legs
q_cand = ||F(z_T)||  from YOUR schedule
r      = clamp( 0.1 + 0.9 * log10(q_base / q_cand) / 2.5 , 0, 1 )
```

Reproducing the weak reference scores `~0.1`; reducing the residual by `2.5` orders of magnitude
under budget reaches `1.0`. That target is deliberately below what a myopic per-leg line search
reaches on the hardest (large, ill-conditioned, strongly-coupled) instances, so excellent
schedules stay strictly below `1.0` — there is real headroom. Doing **worse** than the reference
scores below `0.1`.

Any invalid output — wrong length, a non-finite or out-of-range gain, non-JSON, a crash, a
timeout, or a schedule that makes the residual overflow to inf/nan — scores `0.0` on that
instance. The final score is the mean `r` over all 10 instances (6 base + 4 harder held-out).

## Isolation

Your program runs OS-sandboxed and sees ONLY the public instance. The reference norm `q_base`,
the target `K`, and the recomputation of `q_cand` all happen inside the evaluator: the score is a
pure function of the residual the evaluator recomputes from your numbers, so there is nothing to
game beyond genuinely designing a better relay.
