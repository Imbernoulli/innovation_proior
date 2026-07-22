# Tuning a Drifting Pirate Radio

**Family:** drift-tracking-probes · **Format:** B (isolated heuristic evaluation) ·
**Objective:** maximize prediction quality

## Story

A pirate radio station broadcasts on a dial. If you tune to offset `x` at tick `t`, the
received signal strength is

```
y(x, t) = A(t) * K(x - f0(t); w)          (+ small noise, when you actually probe)
K(u; w) = max(0, 1 - (u/w)^2)
```

`K` is a parabolic reception bump of half-width `w` — this **shape never changes** during an
episode. What does change, on two different clocks, is where the station sits and how loud it
is:
- `f0(t) = f0_0 + drift * t` — the transmitter's frequency creeps **linearly** (thermal drift).
- `A(t) = max(0, A_mid + A_amp * sin(2*pi*t/period + phase))` — the loudness fades in a **slow
  sinusoid** (an ionospheric cycle). Its period can be long enough that the trend you observe
  can **reverse** before the tick you must predict.

`w`, `f0_0`, `drift`, `A_mid`, `A_amp`, `period`, `phase` are fixed per instance and hidden;
only coarse (padded, off-center) ranges for the first six are given as hints — `phase` is not
hinted at all.

## Protocol (your program is invoked once per round)

You get a budget of `max_probes = 45` probes, spent across `R = 3` adaptive rounds; each round
you see every reading so far and choose the next batch (a round may spend the whole remaining
budget). Probes may only target the **observed window** `t in [0, 999]` — the frozen queries
you must ultimately predict live strictly later, in `t in [1000, 1500]`, so any static picture
you build must be extrapolated through both drifts, not just read off. Read ONE JSON object
from stdin and write ONE JSON object to stdout.

**Query round** (`phase == "query"`): stdin has
`{"phase":"query","dial":{"x_lo","x_hi"},"t_obs_max","hints":{"w_lo","w_hi","f0_0_lo","f0_0_hi","drift_lo","drift_hi","A_mid_lo","A_mid_hi","A_amp_lo","A_amp_hi","period_lo","period_hi"},"noise_std","budget":{"max_probes"},"budget_left","round","R","max_this_round","history":[{"x","t","y"},...]}`.
Reply `{"probes":[{"x":x,"t":t}, ...]}` with finite `x`, `t`. An entry with `x` outside
`[x_lo,x_hi]` or `t` outside `[0,t_obs_max]` is **silently skipped** — still charged against
budget, no reading returned. A malformed entry (wrong type, non-finite) scores the **whole
instance 0.0**. At most `min(budget_left, max_this_round)` entries per round are honored.

**Predict round** (`phase == "predict"`): stdin additionally has
`"test_queries":[{"x","t"}, ...]` (a frozen, unseen suite with `t` in `[1000,1500]`). Reply
`{"predictions":[p_0, ..., p_{K-1}]}`, one finite real number per query, same order. Wrong
length / type / non-finite scores **0.0**.

Any crash, timeout, non-JSON, or wrong shape on any round scores that instance **0.0**.

## Scoring (deterministic)

Per instance, with true signal `y_true(x,t)`:
```
err_ref = mean_j |naive(test_queries[j]) - y_true(test_queries[j])|
naive(x,t) = A_mid_guess * K(x - f0_guess; w_guess), CONSTANT in t, using only the
             midpoints of the public hint windows -- exactly what predicting with
             zero probes reproduces
err     = mean_j |pred_j - y_true(test_queries[j])|
quality = clip(1 - err/err_ref, 0, 1)
r       = 0.10 + 0.82 * quality                          # cap 0.92, floor 0.10
```
Predicting the naive constant formula everywhere (no probes) scores exactly `0.10`. Your total
score is the mean of `r` over 10 fixed instances. Several concentrate most of their held-out
queries near where the station's true trajectory actually is at that future tick — a static
snapshot fit built early and never revisited is blind exactly there.

## Isolation

Your program runs in a fresh sandboxed subprocess via `isorun.run_candidate` and sees only the
public fields above. `w`, `f0_0`, `drift`, `A_mid`, `A_amp`, `period`, `phase`, and the true
`y_true` live only in the evaluator's parent process.
