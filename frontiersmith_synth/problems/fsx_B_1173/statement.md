# Listening Line: TDOA Localization with a Degenerate Array

## Problem

A line of `R` moored hydrophone buoys listens for a tagged transmitter's acoustic
pings. Sound travels at a known, fixed speed `c`. Buoy `0` is the timing reference;
every other buoy `r` reports the **time difference of arrival** relative to buoy 0,
`tau_r`. Each buoy also has its own small, unknown clock offset `beta_r` (a fixed
nuisance bias for the whole array, the same for every emitter it ever hears).

Let `X_ref` be the arithmetic mean of the `R` buoy positions. Because every emitter
in this task pings from near the array (its "operating corridor"), the true
hyperbolic TDOA relation is well approximated throughout by its **linearization**
around `X_ref`, and that linearization *is* the forward model used to generate every
measurement here:

```
tau_r(X) = J_r . (X - X_ref) + beta_r + noise_r
```

`J_r` is the exact gradient of the hyperbolic TDOA equation `(|X-P_r|-|X-P_0|)/c` at
`X_ref`: `J_r = (u_r - u_0) / c`, where `u_k` is the unit vector from `X_ref` toward
buoy `k`. You can compute every `J_r` directly from the given buoy positions and
`c` — it needs no unknown quantity. `noise_r` is independent per-measurement
timing jitter.

You are given `K_cal` **calibration** emitters at *known* positions together with
their TDOA readings (so you can characterize the biases), and `K_test` **held-out**
emitters for which only the TDOA readings are given — you must report your best
estimate of each held-out emitter's `(x, y)` position.

## Why this is not always easy

`J_r` for `r=1..R-1` stacks into an `(R-1) x 2` matrix `J`. How accurately `(x, y)`
can be recovered depends entirely on `J`'s conditioning, fixed by the buoy layout
alone (not by any particular emitter). When the buoys sit nearly on one straight
line, `J`'s two column directions become nearly parallel: one direction in the plane
is strongly determined by the data, the orthogonal one is barely constrained at all —
tiny measurement noise there is amplified into an enormous, meaningless position
error if fit head-on. A solver that treats the two directions differently — trusting
data where the array supports it, falling back on other available information (such
as where the calibration emitters cluster) where it does not — beats one that always
fits everything to the raw data the same way.

## Input (stdin)

```
test_id R c
R lines: x y                                  (buoy positions; buoy 0 = reference)
K_cal
K_cal lines: x y tau_1 ... tau_{R-1}           (calibration emitter: known position + TDOA)
K_test
K_test lines: tau_1 ... tau_{R-1}              (held-out emitter: TDOA only)
```
`4 <= R <= 7`, `K_cal = 14`, `K_test = 10`. All values are ASCII decimals.

## Output (stdout)

Exactly `K_test` lines, each `x y`: your estimated position for the corresponding
held-out emitter, in the same order as the input.

## Feasibility

Every one of the `2*K_test` output tokens must parse as a finite real number.
Non-numeric, missing, extra, `nan`/`inf`, or absurdly large (`>1e7` in magnitude)
tokens make the whole submission score `0.0`.

## Scoring

For each held-out emitter, `closeness = D / (D + err)` where `err` is the Euclidean
distance between your estimate and the true position and `D = 100` (meters). Let `F`
be the mean closeness over the `K_test` emitters. The checker also computes `B`, the
same mean-closeness score for its own trivial predictor (the buoy centroid `X_ref`,
reported for every emitter, ignoring all TDOA data). Your final ratio is
`min(1000, 100*F/B) / 1000`, so reproducing the trivial predictor scores `0.1`, and
the array's own noise floor keeps even the best possible fit below `1.0`.

## Worked Example (illustrative FORM only)

With 2 toy buoys at `(0,0)` and `(10,0)` (`c=1`, not to real scale), `X_ref=(5,0)`
and `J_1 = (-2, 0)`. If `tau_1 = -0.2` and `beta_1 = 0`, the x-offset solving
`J_1 . (dx,dy) = tau_1` is `dx = 0.1`; you would report `X_ref + (0.1, dy)`, with
`dy` depending on how well that second direction is constrained — exactly the
situation the real (many-buoy) instances scale up.

## Constraints
- `4 <= R <= 7`, `c` fixed and given, `K_cal = 14`, `K_test = 10` per test.
- Time limit 5s, memory 512MB.
