# Spectral Trainer: Learning-Rate Schedule Design

A convex trainer will run **exactly `N` gradient-descent steps** on a **fixed convex
quadratic loss** and asks you to supply the **learning-rate schedule** — one step size
per iteration, chosen in advance. Design the schedule so the trainer's **final gradient
norm is as small as possible**.

## The loss

The loss is a diagonal (separable) convex quadratic

```
L(w) = sum_i  0.5 * h_i * (w_i - w*_i)^2 ,     h_i > 0
```

so coordinate `i` has curvature `h_i` and gradient component `g_i = h_i * (w_i - w*_i)`.
(Working in the Hessian's eigenbasis is without loss of generality for gradient descent,
so this diagonal view is the fully general spectral picture of quadratic optimization.)

Gradient descent with schedule `(eta_0, ..., eta_{N-1})` updates every coordinate
independently, so after all `N` steps the gradient component of coordinate `i` is

```
g_i(N) = g_i(0) * PROD_{t=0}^{N-1} (1 - eta_t * h_i)
```

and the trainer's final gradient norm — the quantity you **minimize** — is
`||g(N)||_2 = sqrt( sum_i g_i(N)^2 )`.

Because the product over `t` is order-independent, choosing the schedule is exactly
choosing a degree-`N` polynomial `p(x) = PROD_t (1 - eta_t*x)` with `p(0)=1`; the score
measures how small `p` can make `sqrt( sum_i (g_i(0) * p(h_i))^2 )`.

## Candidate contract (isolated program: stdin JSON -> stdout JSON)

Read ONE JSON object from stdin — the **public instance**:

```json
{"name": "trainer101",
 "n_steps": 24,
 "curv":  [h_0, ..., h_{d-1}],
 "grad0": [g_0, ..., g_{d-1}]}
```

- `n_steps` (`N`): number of gradient-descent steps (the schedule length you must return).
- `curv`: the positive curvatures `h_i` (the Hessian diagonal / spectrum), length `d >= 2*N`.
- `grad0`: the initial gradient components `g_i(0)` (real numbers).

Write ONE JSON object to stdout:

```json
{"lr": [eta_0, ..., eta_{N-1}]}
```

`lr` must be **exactly `N` finite real numbers** (the per-step learning rates). Negative
or divergent-but-finite step sizes are allowed (they just tend to blow the gradient up).

## Objective and scoring

Objective: **minimize** the final gradient norm `||g(N)||_2`.

Each instance is scored in `log10(gradient-norm)` space against two anchors the evaluator
computes itself:

- `G_base` — the weak **constant** schedule `eta_t = 1/L` (`L = max(curv)`); anchors ~0.1.
- `G_cheb` — the `N`-step **Chebyshev** schedule (reciprocals of the minimax-polynomial
  roots on `[mu, L]`, `mu = min(curv)`); near-optimal.
- `G_ideal = G_cheb - 1.0` — an unreachable ideal one order of magnitude beyond Chebyshev.

Let `G_cand` be `log10` of your schedule's final gradient norm. Then

```
r = clamp( 0.1 + 0.9 * (G_base - G_cand) / max(G_base - G_ideal, 1e-6),  0, 1 )
```

The overall score is the mean of `r` over all instances (`Ratio`), with the per-instance
`r` values also printed as `Vector`.

- Reproducing the weak constant baseline scores ~0.1.
- The near-optimal Chebyshev schedule scores well below 1.0.
- Only a schedule an order of magnitude better than Chebyshev could reach 1.0, and the
  discrete, gradient-weighted spectrum makes that unreachable — there is genuine headroom
  above Chebyshev for a schedule that exploits the actual gradient weights `g_i(0)`.
- A malformed answer (wrong length, non-number, `nan`/`inf`), a crash, a timeout, or
  non-JSON scores 0.0 for that instance; doing worse than the weak baseline scores `< 0.1`.

## Notes

- Scoring is fully deterministic; the instance family is seeded and fixed.
- Your program is run **isolated** in a fresh sandboxed subprocess and only ever sees the
  public instance above — the anchors are computed in the parent evaluator.
- There is no single best schedule: constant steps, Chebyshev acceleration, and
  gradient-weighted polynomial fits are all viable strategies at different points on the
  quality ladder.
