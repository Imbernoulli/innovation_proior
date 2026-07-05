# Coral Reef Survey: Robust Transect Balancing

## Story

A marine lab runs an autonomous reef-survey vehicle. Two teams argue over every
dive plan. The **survey planner** chooses a control vector `x` (how long to linger
on each transect) to *minimise* a cost; an adversarial **current/estimator** chooses
a disturbance vector `y` (worst-case drift and sensor bias) to *maximise* the same
cost. The dive plan everyone can agree on is the **saddle point** of a convex-concave
objective

```
f(x, y) = 1/2 x^T A x  +  x^T B y  -  1/2 y^T C y  +  a^T x  -  c^T y
```

with `A, C` symmetric positive-definite (each team's curvature) and `B` the
survey/current coupling. Writing `z = (x, y)`, the joint stationarity condition is
`V(z) = 0`, where `V` is the monotone saddle vector field

```
V(z) = ( A x + B y + a , -B^T x + C y + c ) = H z + q ,
H = [[A, B], [-B^T, C]]        # z^T H z = x^T A x + y^T C y > 0
```

You are given a **fixed budget of `K` extragradient-style iterations** (each using
two field evaluations; there is **no wall-time component** in scoring) starting from
a fixed point `z0`. Iteration `k` applies

```
z_half  = z_k - alpha_k * V(z_k)
z_{k+1} = z_k - beta_k  * V(z_half)
```

Setting `alpha_k = 0` recovers a plain descent-ascent step of size `beta_k`; a
positive `alpha_k` adds the extragradient extrapolation that tames the rotational
part of the saddle field. **Your task is to design the schedule `{alpha_k, beta_k}`
that makes the final survey residual `||V(z_K)||` as small as possible.**

Because the state dimension `N = 2d` exceeds `2K`, no schedule can zero the residual:
this is a genuine min-residual (Chebyshev-type) design problem with **no closed-form
optimum**.

## You write a program (isolated)

Your program reads ONE JSON object (the public instance) from **stdin** and writes
ONE JSON object (your schedule) to **stdout**. It runs in an isolated sandbox and
sees only the public instance.

### Public instance (stdin)

```json
{
  "name": "reef101",
  "N": 40,
  "K": 12,
  "H":  [[...], ...],      // N x N saddle operator (row-major)
  "q":  [...],             // length N; V(z) = H z + q
  "z0": [...],             // length N; fixed start point
  "Lspec": 10.6,           // spectral-norm bound ||H||_2
  "ref_alpha": 0.0377,     // the reference (weak) extragradient step
  "ref_beta":  0.0377
}
```

### Answer (stdout)

```json
{ "alpha": [a_0, ..., a_{K-1}], "beta": [b_0, ..., b_{K-1}] }
```

`alpha` and `beta` must each be a list of exactly `K` finite real numbers with
`|value| <= 1e8`.

## Objective

**Minimise** the final residual norm `||V(z_K)||` after exactly `K` iterations of the
extragradient template driven by your schedule, starting from `z0`.

## Scoring (deterministic, no wall-time)

For each reef the evaluator RE-RUNS the extragradient template with your schedule and
computes:

- `g0     = ||V(z0)||`                          (initial residual)
- `g_base = ||V(z_K)||` under the reference schedule (constant `ref` step)
- `g_cand = ||V(z_K)||` under YOUR schedule
- `g_ideal = g0 * 1e-4`                          (unreachable ideal floor)

and normalises on a log anchor (reference -> 0.1, ideal floor -> 1.0):

```
r = clamp( 0.1 + 0.9 * (log10 g_base - log10 g_cand)
                      / (log10 g_base - log10 g_ideal),  0, 1 )
```

Reproducing the reference schedule scores ~0.1. Beating it drives `r` up, but the
ideal floor is unreachable in `K` steps, so even a Chebyshev-optimal design stays
below 1.0. A schedule that does worse than the reference, blows the iterates up
(non-finite or norm `> 1e12`), or violates the answer shape scores 0.0 on that reef.

The reported **Ratio** is the mean of `r` over all 12 reefs (8 base + 4 harder
held-out); **Vector** lists the per-reef `r`.

## Notes / strategy hints

- The reference constant step is deliberately conservative; a larger constant step
  helps a bit but cannot adapt to the spectrum.
- Reading `H` and spreading the `K` steps across its (real-part) spectrum -- e.g. a
  Chebyshev semi-iteration -- gives the classic min-residual acceleration.
- The extragradient extrapolation (`alpha_k > 0`) and momentum/optimistic variants
  can further damp the rotational (imaginary) part of the saddle field.
- Everything is deterministic; there is no reward for guessing the held-out reefs
  other than designing a genuinely better update.
