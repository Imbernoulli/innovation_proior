# Smog Ledger: Attributing Emissions Under a Shifting Wind Rose

## Problem
A single receptor sensor sits at the origin `(0,0)`. `K` candidate factories at
fixed coordinates each emit an unknown, constant emission rate `E_i >= 0`. A
logbook records `D` days of monitoring: each day's wind (direction it blows
*toward*, degrees; speed) and the sensor's total measured concentration. Your
job: output an estimate of every factory's emission rate.

**Transport model.** For a source at `(sx,sy)`, let `r` be its distance from the
receptor and `brg` the compass bearing *from the source toward the receptor*
(the direction wind must blow for that source's plume to reach the sensor). On
a day with wind direction `wd` and speed `ws`, let `delta` be the smallest
angle between `wd` and `brg`, and `sigma = SIGMA_MAX/(1+ALPHA*ws)` (higher wind
speed -> narrower, sharper plume; near-zero speed -> broad, near-isotropic
spread). The source's transport factor that day is
```
kernel = exp(-delta^2 / (2*sigma^2)) * exp(-r / L0)
```
Each day's *total* is diluted by wind speed, `dilution = 1/(1+BETA*ws)`, and a
regional background trend keyed to that day's true index `d` is added:
`background(d) = A0 + A1*sin(2*pi*d/P)`. So the measured concentration on day
`d` (before sensor noise) is `dilution * sum_i(kernel_i * E_i) + background(d)`.
Some factories sit in tight bearing **clusters**: two or three of them lie only
a few degrees apart as seen from the receptor, so on any *single* day their
kernel values are nearly identical -- they are confounded. Only days whose wind
direction differs enough between them can ever separate a cluster's members,
and that separation gets sharper (smaller `sigma`) on windier days.

## Input (stdin)
```
test_id K D
A0 A1 P SIGMA_MAX ALPHA L0 BETA
<K lines>: x_i y_i            (factory i's coordinates, i = 1..K)
<D lines>: day_id wind_dir wind_speed concentration
```
The `D` day-rows are **not** printed in day_id order -- use the explicit
`day_id` field (not row position) whenever you need a day's true index (e.g.
for the background term).

## Output (stdout)
`K` whitespace-separated non-negative numbers: your estimated `E_1 .. E_K`, in
the same order as the input's factory list.

## Feasibility
The output must contain exactly `K` tokens, each a finite number satisfying
`0 <= E_i <= 10^6`. Any violation (wrong count, non-numeric, `nan`/`inf`,
negative, or out of range) scores `Ratio: 0.0`.

## Scoring
The grader knows the true emission rates (hidden) and a further set of
held-out days (never shown to you, spanning many wind directions/speeds) drawn
from the same physical model. Some factories in this instance belong to a
tight bearing cluster (see above); let `err_cluster` / `err_standalone` be the
relative-L1 recovery error restricted to the clustered / non-clustered
factories respectively (recovering the clusters correctly is the point of the
instance, so it counts for more). Let
```
err_recov = 0.75*err_cluster + 0.25*err_standalone
err_hold  = rel_L1_error(your predicted held-out concentrations, true held-out concentrations)
F         = 0.85*err_recov + 0.15*err_hold
```
(errors are relative-L1, capped, and `F` is floored at a small epsilon). The
grader also computes `B = F` for its own trivial construction (guess the same
uniform rate for every factory, calibrated to the average day). Then
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching the trivial guess's error scores `Ratio ~= 0.1`; a SMALLER combined
error than the trivial guess pushes the ratio up without bound (capped at
1.0). Correctly untangling the clustered factories -- which requires trusting
the *sharp, direction-varying* days over the *loud, stagnant* ones -- is what
drives the error down.

## Constraints
`6 <= K <= 10`, `D <= 150`. Time limit 5s, memory 512m.

## Example (illustrative FORM only, not to scale)
Two sources: A at `(30,0)` (bearing-to-receptor 180 deg), B at `(0,30)`
(bearing-to-receptor 270 deg) -- well separated, not a cluster. A day with
`wd=180, ws=3` (`sigma` small) gives A a kernel near `exp(0)=1` and B near
`exp(-90^2/(2*sigma^2)) ~= 0`: that reading is almost entirely A's. A day with
`wd=225` (halfway between) gives both similar, smaller kernels: alone it
cannot tell A from B apart -- only combining it with the first, differently
aimed day resolves both.
