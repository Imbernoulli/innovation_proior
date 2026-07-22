# One Controller Versus Twenty-Five Hostile Disturbances

A rigid mass sits on a spring and damper and is regulated to position `x = 0` by a force
actuator `u(t)`. Its motion obeys

```
m * x'' + c * x' + k * x = u(t) + d(t)
```

where `d(t)` is an external disturbance force. You must design **one fixed feedback
controller** and it will be replayed, unchanged, against a **published suite of 25
disturbance profiles**: 15 sinusoids at listed frequencies/amplitudes/phases, 6 transient
pulse trains, and 4 slow bias-plus-ramp drifts. Your score is driven by the **worst** of
those 25 profiles (plus a small average term), so the controller must survive every one of
them — not just do well on average.

## What you submit

Your program reads ONE JSON object from stdin — the full public instance — and must print
ONE JSON object describing your controller:

```
{"kp": <float>, "kd": <float>, "ki": <float>,
 "resonators": [{"freq": <float>, "zeta": <float>, "gain": <float>}, ...]}
```

`resonators` may contain up to `max_resonators` entries (0 is fine, and the key may be omitted
entirely to mean "no resonators"). `kp` in `[0,320]`, `kd` in `[0,13]`, `ki` in `[0,200]`,
resonator `freq` in `(0,60]`, `zeta` in `[0.005,1.5]`, `gain` with `|gain| <= 300`. Missing
`kp`/`kd`/`ki`, wrong type, a non-finite value, an out-of-range value, or more than
`max_resonators` entries makes that instance score 0.

The `kp`/`kd` budget is deliberately tight: it is not enough to buy your way out of resonance
by simply pushing the base loop's bandwidth and damping as high as the plant would otherwise
allow. Getting the worst case down further requires the resonators.

## How the evaluator turns your answer into a controller

Let `e(t) = -x(t)` be the tracking error. Each resonator `k` is a small internal filter
driven by the error signal:

```
r_k'' + 2*zeta_k*w_k*r_k' + w_k^2*r_k = w_k^2 * e(t)          (w_k = resonator's "freq")
```

The commanded force is

```
u_raw(t) = kp*e(t) + kd*(-x'(t)) + ki*integral(e) - sum_k gain_k * r_k(t)
u(t)     = u_max * tanh(u_raw(t) / u_max)                      (smooth actuator ceiling)
```

The integral term uses simple anti-windup: it stops accumulating further in whichever
direction it is already saturated past `+-50` (it never affects a well-behaved `ki`, but it
keeps a runaway `ki` choice from blowing up the simulation instead of just scoring badly).

`u_max` is given in the instance. A resonator whose `freq` sits exactly on one of the
suite's published sinusoid frequencies, with a small `zeta`, strongly damps the closed
loop's steady response to precisely that sinusoid — this is the whole point of reading the
published list.

## Simulation and scoring

The evaluator integrates the closed loop with fixed-step RK4 (`dt`, horizon `T` given in the
instance) separately for each of the 25 published profiles, starting at rest (`x=v=0`),
and computes `E_j = integral over [0,T] of |x(t)| dt` and a mild actuation-effort term
`energy_j = integral over [0,T] of (u(t)/u_max)^2 dt` for profile `j`. Your objective
(**lower is better**) is

```
obj = max_j(E_j) + 0.25 * mean_j(E_j) + 0.05 * mean_j(energy_j)
```

This is compared against a fixed, plant-naive reference controller the evaluator computes
itself (never a submission) to produce your final per-instance score in `[0,1]`; scores are
averaged over 10 instances (different plants, different published spectra). The score
saturates well below 1.0, so there is always room to do better.

## Constraints and guarantees

- `0.35*w0 <= freq <= 6*w0` roughly bounds where published sinusoid frequencies live,
  `w0 = sqrt(k/m)` being the plant's own open-loop frequency — but read the instance, don't
  assume.
- The plant is always passively stable (`m,c,k > 0`) and the actuator output is smoothed by
  `tanh`, so a well-formed answer can never make the simulation diverge.
- Time limit per instance run is generous; your program does a small, fast computation (no
  simulation needed on your end) and prints its answer.

## Notes

The suite is fully published in the input — nothing about the disturbances is hidden. A
controller tuned only to "be fast and stable on average" without looking at exactly which
frequencies, amplitudes, and timings appear in the suite will do fine on most profiles but
can resonate badly on whichever published sinusoid happens to sit near its own closed-loop
frequency. Reading the spectrum and shaping the loop against it is worth doing.
