# Spearfishing Angles in a Layered Lagoon

A spear-fisher's laser rangefinder shoots a ray straight down through a
lagoon whose water is stratified into **exactly three flat, horizontal
layers** (thermocline/salinity bands) of unknown thickness and unknown
refractive index, down to a fixed sensor plane at depth `D` below the
surface. Your job: from a handful of **calibration shots**, build a model
that predicts where and when the ray reaches the sensor plane at *any* aim
angle.

The catch: the diver only ever calibrates the rangefinder near-vertical
(steep aim causes glare on the surface), so every calibration shot has a
small entry angle. You will be scored on the *full* range of aim angles —
including angles steep enough that the ray never reaches the sensor at all,
because it undergoes **total internal reflection (TIR)** at one of the
hidden interfaces.

## Physics (given, applies to every instance)

Layers are flat and parallel, entry medium index `n0` is given. Snell's law
composes across the whole stack: if a ray enters at angle `theta0` from the
vertical, then in **every** layer `i` (index `n_i`), `n_i * sin(theta_i) =
n0 * sin(theta0)` — as long as this is solvable (`sin(theta_i) <= 1`) in
every layer the ray reaches. If some layer `j` would require `sin(theta_i) >
1`, the ray totally internally reflects at the interface above layer `j` and
**never reaches the sensor** (no offset, no time — "no exit"). Within a
layer of thickness `d` at angle `theta_i`, the ray contributes horizontal
offset `d*tan(theta_i)` and travel time `d*n_i/cos(theta_i)` (speed `1/n_i`
in that layer); offset and time are summed over all layers reached.

**Illustrative example only — NOT any real instance's hidden stack:** a
single layer with `d=4, n=1.2` at `n0=1.0`, `theta0=30 deg` gives
`sin(theta_i)=sin(30)/1.2=0.4167`, offset `= 4*tan(asin(0.4167)) = 1.833`,
time `= 4*1.2/cos(asin(0.4167)) = 5.28`.

## Input (stdin)

```
n0  D
n_train  test_id
theta_deg_1  offset_1  time_1
...
theta_deg_k  offset_k  time_k
```

`n_train` noisy calibration shots follow, all at small `theta_deg` (near
vertical). `offset`/`time` have small measurement noise added.

## Output (stdout): your hypothesized layer stack

```
L
d_1  n_1
...
d_L  n_L
```

`1 <= L <= 8` layers, listed top to bottom. `n_i > 0` for all `i`; `d_i > 0`
for `i < L` (the LAST layer's printed thickness is ignored and replaced by
whatever depth remains to reach `D` — only its index matters). You need not
use `L=3`; use however many layers best explain the data.

## Feasibility

Output must parse as above with finite numbers, `L` in range, positive
thicknesses for all but the last layer, and the non-final thicknesses must
sum to strictly less than `D`. Any violation, or any non-finite value,
scores `0`.

## Objective (minimise)

The checker ray-traces **your** stack (same physics above) at a fixed set of
held-out angles spanning moderate to near-grazing incidence, and compares
against the true stack's behaviour at those same angles:

- If both your stack and the truth predict the ray exits, the point's cost
  is a normalised combination of `|offset error|` and `|time error|`
  (capped, so one bad point cannot dominate).
- If your stack and the truth **disagree about whether the ray exits at
  all**, the point pays a fixed penalty (the same cap) — this is what
  punishes a model that never predicts TIR, or predicts it in the wrong
  place, even though it agrees perfectly with the near-normal training
  shots.
- Both agreeing "no exit" costs nothing at that point.

The mean per-point cost is scaled up slightly by your layer count `L`
(fewer layers preferred, all else equal), then compared against the
checker's own internal baseline — a single straight, unrefracted ray
(`n=n0`) — to form the ratio you are graded on. Reproducing that baseline
scores about `0.1`; lower held-out cost raises your score.

## Constraints

Time limit `5s`, memory `512MB`. `n_train` is under 100 rows. All scoring is
exactly deterministic given your submitted output.
