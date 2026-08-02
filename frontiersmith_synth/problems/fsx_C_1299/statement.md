# GEO Longitude Station-Keeping: Dead-Band on a Budget

## Story

Fuel is the only thing you cannot make more of. Your geostationary satellite sits in
an assigned longitude slot, tracked as a scalar offset `x` from slot center. Earth's
equatorial triaxiality pulls every GEO satellite toward one of two stable longitudes
with an almost-constant along-track acceleration whose *sign* depends only on which
side of the nearest stable point your slot sits on -- a **systematic** drift, not
noise. Riding on top of it are small, effectively random perturbations (solar-pressure
fluctuations, mismodeled forces):

```
x[t+1] = x[t] + bias + noise[t]     (noise[t] drawn uniformly from [-noise_amp, noise_amp])
```

The satellite must stay inside a station-keeping box `|x| <= box_half_width`; the
first step it steps outside, the mission ends. You control it with correction burns:
firing one instantly resets `x` to a `target_pos` of your choosing, at a fuel cost of
`|x - target_pos| / efficiency(t)`. Efficiency is periodic and low most of the time --
`efficiency(t) = eff_high` only while `(t mod period) < window_len`, and `eff_low`
otherwise -- reflecting that burns timed to the right point in the orbit (ground-station
geometry, thermal constraints) are far cheaper than ad-hoc ones. As a last resort,
*once you are already outside your own dead-band* (`x > band_hi` or `x < -band_lo`),
if `|x|` would also cross `0.95 * box_half_width` mission control forces an immediate
burn to `target_pos` regardless of efficiency or `patience`, so a slow policy cannot be
blindsided into an unrecoverable exit -- but that forced burn is usually the expensive
kind, and it only ever helps if your dead-band is narrower than `0.95 * box_half_width`:
a wider `band_hi`/`band_lo` never registers a violation before the hard box-exit, so it
gets no protection at all.

Correcting every deviation immediately keeps `x` tightest to slot center, but it burns
on nearly every step, almost always missing the efficient window, and runs the tank dry
long before the mission horizon. **Maximize the number of steps survived** (capped at
the horizon) within your `fuel_budget`.

## Input (public instance, one JSON object on stdin)

```json
{"name": "slot-EastDrift-A", "horizon": 391, "box_half_width": 100.0,
 "bias": 0.55, "noise_amp": 0.35, "fuel_budget": 88.21,
 "period": 24, "window_len": 4, "eff_high": 1.0, "eff_low": 0.2}
```

- `horizon` (int): steps simulated; surviving all of them is full credit.
- `box_half_width` (float): the box is `[-box_half_width, box_half_width]`.
- `bias` (float): systematic per-step drift; sign is the direction it pushes `x`.
- `noise_amp` (float): each step's random perturbation is uniform in
  `[-noise_amp, noise_amp]` (drawn by the evaluator; you do not see the realization).
- `fuel_budget` (float): total burn cost you may spend.
- `period`, `window_len` (int): efficiency is `eff_high` for the first `window_len`
  steps of every `period`-step cycle (absolute step index, not relative to your
  burns), `eff_low` the rest.

## Output (one JSON object on stdout)

You submit a **policy**, not a per-step trajectory -- the evaluator runs it against
the real (hidden) noise sequence:

```json
{"band_lo": 95.0, "band_hi": 79.99, "target_pos": -65.0, "patience": 20}
```

- `band_lo`, `band_hi` (float `>= 0`): a correction is due once `x < -band_lo` or
  `x > band_hi`.
- `target_pos` (float): the position a correction burn resets `x` to.
- `patience` (int `>= 0`): once a correction is due, steps you're willing to wait for
  an `eff_high` window before burning anyway (the 0.95-margin safety burn can still
  fire sooner).

Any of the following scores that instance `0.0`: a missing field, wrong type, NaN/Inf,
negative `band_lo`/`band_hi`/`patience`, `|target_pos| > 1e6`, a crash, a timeout, or
output that is not the JSON object above.

## Scoring (deterministic)

For each instance the evaluator computes `base_life` = steps survived by "never
correct" (its own weak reference), `ub_life = horizon`, and `cand_life` = steps your
policy survives, then normalizes:

```
r = clamp(0.1 + 0.9 * (cand_life - base_life) / max(1e-9, ub_life - base_life), 0, 1)
```

Matching the do-nothing reference scores ~0.1; surviving longer scores higher, capped
at 1.0 -- fuel-tight instances keep even a strong policy below 1.0, leaving headroom.
Your final score is the mean of `r` over all 10 instances (mixed drift direction,
noise scale, and window width, including harder held-out profiles).

## Notes

- Scoring never measures wall-clock time; the time limit only bounds your program.
- Your program is run in an isolated subprocess and sees only the public instance
  above -- it computes a policy once, it does not interact with the simulation.
