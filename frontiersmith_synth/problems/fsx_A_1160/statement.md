# Winter Freight: The Season's Spacetime Budget

You dispatch convoys across a freezing lake for one **42-day season** over
**4 fixed corridors** (numbered 0..3, roughly shortest to longest). Every
corridor's ice thickness evolves on its own clock, driven by the day's
temperature; every crossing you make also **fatigues** the corridor you
used, and that fatigue decays on a *separate*, slower clock — whether or
not you cross it again. Capacity is not a fixed property of a corridor: it
is a moving target set by two independent clocks (the season's freeze/thaw
curve, and your own accumulated wear), and you must plan against both at
once, for the whole season, before day 1.

## The physics (deterministic, all constants given in the input)

For each corridor `r` with published constants `length_factor[r]`,
`growth_rate[r]`, `thaw_rate[r]`, `h0[r]`, and the season's published daily
temperatures `T[0..n_days-1]` (freeze point is `0.0`), ice thickness evolves
independently of anyone's usage:
```
h[r][0] = h0[r]
h[r][d+1] = max(0, h[r][d] + growth_rate[r]*max(0, -T[d])
                            - thaw_rate[r]*max(0, T[d]))
```
Each corridor also carries a **fatigue** level `fatigue[r]`, starting at 0,
that decays every single day regardless of use, by the input's
`fatigue_decay` factor. The corridor's *effective* thickness on day `d` is
`eff[r][d] = max(0, h[r][d] - fatigue[r][d])`, and the **maximum safe
cargo mass** you may cross that day is
```
max_safe[r][d] = stress_limit * eff[r][d]^2 / length_factor[r]
```
(a heavier convoy stresses ice as roughly `mass * length_factor / eff^2`;
the input's `stress_limit` is the cap on that stress). If you cross corridor
`r` on day `d` with mass `m`:
- **If `m <= max_safe[r][d]`** (safe): you deliver `m` cargo, and fatigue
  gains `fatigue_gain_k * (stressed fraction) * h[r][d]`, where the
  stressed fraction is `m*length_factor[r] / (stress_limit*eff[r][d]^2)`.
- **If `m > max_safe[r][d]`** (overloaded): the ice cracks — you deliver
  **0 cargo that day**, and fatigue instead jumps by
  `crack_penalty_k * h[r][d]` (a much larger structural hit).

Then, for every corridor, `fatigue[r][d+1] = fatigue_decay * fatigue[r][d]
+ (the gain above, only for the used corridor, else 0)`. A rest day adds no
fatigue, but existing fatigue still decays. At most one corridor per day
(or rest).

## Candidate program contract

Standalone program, stdin -> stdout, isolated subprocess:
```python
import sys, json
inst = json.load(sys.stdin)
# ...compute a full season plan...
print(json.dumps({"routes": routes, "masses": masses}))
```
**Public instance (stdin)** — the WHOLE season is known in advance (a
published forecast), so this is a full-information planning problem:
```json
{"name": "lake07", "n_days": 42, "freeze_point": 0.0,
 "mechanics": {"stress_limit": 0.42, "fatigue_gain_k": 0.22,
               "crack_penalty_k": 0.30, "fatigue_decay": 0.70},
 "routes": [{"length_factor": 1.02, "growth_rate": 0.53, "thaw_rate": 0.94, "h0": 4.4}, ...],
 "temps": [-1.8, 0.6, -4.1, ...]}
```
**Answer (stdout)**: `{"routes": [r_0, ..., r_{n_days-1}], "masses": [m_0, ..., m_{n_days-1}]}`,
both length `n_days`. Each `r_d` is an integer in `{-1, 0, 1, 2, 3}` (`-1` =
rest, no crossing; that day's mass is ignored). Each `m_d` must be finite
and non-negative. Any malformed answer (wrong length, non-numeric,
negative, out-of-range route index, NaN/Inf), a crash, or a timeout scores
**0**. A valid answer is never rejected for overloading — an overload just
zeroes and fatigues that one day; the season continues.

## Objective and scoring (deterministic)

Total cargo = sum of cargo delivered over all `n_days`. We normalize
against a fixed anchor the evaluator computes itself from the public
instance alone: for every day, the best safe mass any single corridor
*could* carry **if it had never accumulated any fatigue at all**, summed
over the season. This anchor is unreachable (a real plan uses at most one
corridor per day and pays real fatigue for using it), leaving headroom.
```
r = clamp(0.1 + 0.9 * total_cargo / unreachable_anchor, 0, 1)
```
Delivering nothing scores exactly `0.1`. The reported **Ratio** is the mean
`r` over 10 seeded seasons; **Vector** lists the per-season scores.

## What to think about

A corridor you hammer every day never gets a chance for its fatigue clock
to fall behind its thickness clock — it stays pinned near a low ceiling for
the whole season, no matter how thick it eventually gets. The four
corridors are a shared, regenerating budget across *time*, not a menu of
paths to rank once.
