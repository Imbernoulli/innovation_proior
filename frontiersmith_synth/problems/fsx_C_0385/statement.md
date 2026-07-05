# Rooftop Garden Sensor Grid: Missing-Reading Reconstruction

## Background
A city operates a network of instrumented **rooftop gardens**. Each garden logs, per
observation window, a vector of environmental sensor readings — soil moisture, canopy
temperature, PAR light, air humidity, substrate pH, wind, CO2, leaf-wetness, and more.
The sensors are cheap and drop out constantly, so the raw reading matrix is riddled
with holes. The irrigation controller needs **complete** vectors, so operations wants
the missing readings **reconstructed** from the surviving ones.

Your job: design a **missing-value imputation algorithm** that fills every hole in a
garden's reading matrix as accurately as possible — across a *diverse family* of
gardens whose readings range from strongly structured (a few latent drivers such as
sun and watering cycle make each channel largely predictable from its neighbours) to
nearly structureless (little cross-channel correlation, heavy noise).

## Isolation / how your program is run
Your program is run as an **isolated subprocess**. It reads ONE JSON object (the *public
instance*) from **stdin** and writes ONE JSON value (your answer) to **stdout**. It never
sees the withheld true readings, any held-out data, or the evaluator's memory.

```python
import sys, json
inst = json.load(sys.stdin)   # public instance (below)
# ... compute fills for the holes ...
print(json.dumps(fill))       # the ONLY thing the evaluator reads
```

## Public instance (stdin)
```json
{
  "X":       [[float|null, ...], ...],   // N x d readings matrix; null = a dropped reading
  "n":       int,                        // number of observation windows N
  "d":       int,                        // number of sensor channels
  "missing": [[i, j], ...],              // the (row, col) holes to fill, in a FIXED order
  "n_miss":  int,                        // == len(missing)
  "seed":    int                         // a per-instance seed you MAY use for your own RNG
}
```
- Every observed entry of `X` is a finite float; every hole is `null` and appears exactly
  once in `missing`.
- The reading matrices are standardized per channel (roughly zero mean, unit variance).

## Answer (stdout)
A JSON list of `n_miss` floats — one imputed value per hole, aligned to `missing`:
```json
[float, float, ..., float]            // fill[t] imputes the hole missing[t]
```
- Length **must** equal `n_miss`. A dict `{"fill": [...]}` is also accepted.
- Any non-finite value (`NaN`/`Infinity`), non-numeric entry, or wrong length scores **0**
  on that instance.

## Objective (maximize)
For each instance the evaluator recomputes the **coefficient of determination R^2**
between your fills and the withheld true readings:

```
R^2 = 1 - sum_holes (y_true - y_fill)^2 / sum_holes (y_true - mean(y_true))^2
```

It normalizes against an internal weak baseline (**per-channel mean imputation** — fill
each hole with its column's observed mean):

```
r = clamp( 0.1 + 0.9 * (r2_cand - r2_base) / max(1 - r2_base, 0.15), 0, 1 )
```

so reproducing the mean-imputation baseline scores ~0.1 and a perfect reconstruction
(R^2 = 1) scores 1.0. The final score is the **geometric mean of `r` over all instances**,
so a method that reconstructs one regime well but collapses on another is heavily
penalized. The garden family mixes:

- **strongly low-rank, clean** gardens (a few latent drivers; regression / matrix
  completion recover most holes),
- **medium-structure** gardens with moderate noise,
- **near-full-rank, noisy** gardens where little cross-channel signal survives (even an
  ideal linear model recovers only a sliver — leave headroom),
- varied **missing fractions** and **grid sizes**, including held-out heavy-missingness
  and wide-grid instances.

No single fixed trick wins them all — an imputer that **generalizes** across regimes wins.

## Scoring output
The evaluator prints:
```
Ratio: <geometric mean of per-instance r, in [0,1]>
Vector: [r_1, r_2, ...]
```
