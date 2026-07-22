# Regime-Aware Relaxation for Ill-Conditioned Fixed-Point Maps

You are tuning an iterative refinement rule for a hidden linear fixed-point map
`G(x) = M x + c` (fixed point `x* = G(x*)`). Across the instance set the map's
conditioning ranges from **benign** (a real, well-separated dominant mode) to
**near-critical** (a *complex* dominant mode whose spectral radius is close to 1,
so plain iteration oscillates and crawls).

You never see `M` or `c`. You see only a short prefix of the **plain iteration**
`x_{k+1} = G(x_k)`. From that prefix you must diagnose the spectral regime and
**prescribe an acceleration scheme** that a referee — who holds `M`, `c` — will
run for a fixed remaining budget of `G`-evaluations, continuing from the last
observed iterate. Your goal: make the final residual as small as possible.

## Program contract (isolated)

Read ONE JSON object from **stdin**, write ONE JSON object to **stdout**. Your
program runs in a sandbox and only ever sees the public fields below.

### Input (public instance)
```
{
  "n": 8,                       # dimension
  "iterates": [x0, x1, ..., xp],# p+1 observed PLAIN-iteration vectors (each length n)
  "remaining_steps": 35,        # R: G-evaluations the referee will run your scheme
  "omega_min": 0.0,             # legal relaxation-factor bounds
  "omega_max": 2.0
}
```

### Output (your prescribed scheme)
```
{
  "omega": 1.35,                # scalar OR a list of per-step factors in [omega_min, omega_max]
  "aitken_period": 3            # apply Aitken every this many steps; 0 = never
}
```

## How the referee runs your scheme

Starting from `x = iterates[-1]`, for `k = 0 .. R-1`:

1. **Relaxed step** (one `G`-evaluation): `x <- x + omega_k * (G(x) - x)`,
   where `omega_k` is your scalar, or `omega[k]` (the last entry is reused if
   your list is shorter than `R`).
2. **Aitken's Delta-squared** (no extra `G`-evaluation): if `aitken_period > 0`
   and `(k+1)` is a multiple of it, the referee applies one componentwise
   `Delta^2` extrapolation using the three most recent iterates, then restarts
   the extrapolation window.

The **residual** of a vector is `||G(x) - x||_2`. Your objective for the
instance is the residual of the final `x`.

## Scoring

Let `b` be the residual reached by the **plain** scheme (`omega = 1`,
`aitken_period = 0`) over the same `R` steps — the do-nothing baseline. Let
`obj` be your final residual. The per-instance score is

```
score = clip( 0.10 + 0.08 * log10( b / obj ),  0.0,  0.90 )
```

so matching the baseline scores `0.10`, every 10x extra residual reduction adds
`0.08`, diverging (larger residual than baseline) scores `0`, and the score is
capped at `0.90` (there is deliberate headroom above any reference solution).
Your reported **Ratio** is the mean over 10 instances; each instance's score is
also emitted in the **Vector** line. An answer of the wrong shape, out of the
`omega` bounds, or non-finite scores `0` for that instance.

## What actually helps

The relaxed map has mode amplification `(1-omega) + omega*lambda` for each
eigenvalue `lambda` of `M`, and the observed step vectors satisfy
`dx_{k+1} = M dx_k`. Two regimes pull the choice of `omega` in **opposite**
directions:

- A **real** dominant mode is annihilated by over-relaxation and mopped up
  almost exactly by Aitken's `Delta^2`.
- A **complex** (oscillatory) dominant mode near the unit circle is *amplified*
  by over-relaxation — it diverges — and Aitken on an oscillation is unstable;
  it must instead be **under-relaxed** to damp it.

Diagnosing which regime you are in — e.g. by fitting the dominant step
recurrence — and setting `omega` (and whether to invoke Aitken) per instance is
the difference between a scheme that shines on the easy maps and one that also
survives the near-critical ones.

Constraints: deterministic; time limit 2–5s; memory ≤ 512 MB.
