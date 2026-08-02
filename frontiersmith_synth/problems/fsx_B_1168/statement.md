# Which Gauge Is Lying, and By How Much

## Problem
A monitoring network has `n` sensors linked by a **redundancy graph**: an edge `(u,v)`
means the two sensors observe physically-coupled quantities, so neighbouring sensors read
similar values. Over `T` timesteps each sensor reports one noisy number per step.

Some sensors are **faulty instruments**: sensor `i`'s reported value is corrupted by an
unknown *offset* `a_i`, an unknown *drift* `b_i` (added as `a_i + b_i*t`), or both — for
every step, forever, regardless of what is actually happening physically. Separately, the
network may also experience a **genuine local event**: a real, temporary physical change
that shows up at one sensor AND, attenuated, at its nearby graph neighbours, then fades.
An event is real signal — it belongs in the reconstruction. A fault is instrument error —
it must be removed. At any single faulty-or-eventful sensor, in isolation, the two can look
identical: both are "this sensor's number is unusually different right now." The only way
to tell them apart is to check whether the anomaly is *corroborated by the graph*: a fault
is inconsistent with its neighbours; a real event is consistent with them.

You may declare **at most `F` sensors** as faulty and give each a correction `(a_i, b_i)`.

*Illustrative FORM only — not the hidden law:* a toy checker might score a submission by
`min(1, correct_guesses / total)`; the shape of the real formula is given exactly below and
does **not** resemble this.

## Input (stdin)
```
test_id
n T F
m
u_1 v_1
...
u_m v_m
r_1[1] r_1[2] ... r_1[T]
...
r_n[1] r_n[2] ... r_n[T]
```
`test_id` seeds the hidden ground truth (you do not need it). `n` sensors (0-indexed),
`T` timesteps, budget `F`. `m` undirected redundancy edges. Then `n` rows of `T`
space-separated floats: sensor `i`'s reported readings `r_i[0..T-1]`.

## Output (stdout)
```
D
s_1 a_1 b_1
...
s_D a_D b_D
```
`D <= F` declared-faulty sensor ids (0-indexed, pairwise distinct, each in `[0,n)`), each
with a correction `(a, b)`. A sensor not declared is left uncorrected.

## Feasibility
Invalid (scores `Ratio: 0.0`) if: `D` is missing/non-integer/outside `[0,F]`; any `s_j`
is out of range or repeated; any `a_j`/`b_j` is missing, non-numeric, non-finite
(`nan`/`inf`), or absurdly large (`|a|>1e4` or `|b|*max(1,T)>1e4`); or there is trailing
garbage after the last declared line.

## Objective and Scoring
Let `X[i][t] = r_i[t] - a_i - b_i*t` for declared sensors (else `X[i][t] = r_i[t]`). The
checker knows the hidden clean field `Y[i][t]` (never shown to you) and computes:
- `RMSE` = root-mean-square of `X[i][t] - Y[i][t]` over all `n*T` entries.
- `F1` = harmonic mean of precision/recall of your declared set against the true fault
  set (0 if you declare 0 sensors while faults exist).
- `Quality = (1 / (RMSE + 0.40)) * (0.45 + 0.55 * F1)`.

Let `B` be the same `Quality` for the "declare nothing" reconstruction (`X = r`, `F1 = 0`).
With maximization normalization:
```
sc = min(1000.0, 100.0 * Quality / max(1e-9, B))
Ratio = sc / 1000.0
```
Declaring nothing always scores exactly `0.1`.

## Constraints
- `8 <= n <= 27`, `30 <= T <= 80`, `1 <= F <= 4`.
- Redundancy graph is connected; each `.in` file is well under 5 MB.
- Time limit 5s, memory 512m.

## Example
Suppose `n=3`, all connected in a path `0-1-2`, `T=4`, `F=1`, sensor `1` truly has offset
`a=+3` (readings elevated by 3 at every step) and there is no event. Declaring nothing
gives some `RMSE = RMSE_base > 0`, `F1 = 0`, hence `B = Quality`. Declaring `1 3.0 0.0`
removes almost all the error at sensor 1, driving `RMSE` down sharply and `F1` to `1.0`,
so `Quality` rises well above `B` and `Ratio` climbs correspondingly above `0.1` (capped at
`1.0`). Declaring a healthy neighbour instead leaves the real offset uncorrected AND
introduces a fresh, wrong correction elsewhere — `RMSE` gets worse, not better, and
`Ratio` stays near `0.1`.
