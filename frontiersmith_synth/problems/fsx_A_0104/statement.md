# Greenhouse Pod Dispersion: No Three Sensors in a Row

## Problem
A high-tech greenhouse is monitored by `n` climate pods (temperature / humidity /
CO2 sensors) placed on the floor, normalised to the unit square `[0,1]^2` (the
greenhouse growing area). The control system interpolates the climate field by
**triangulating** the pods and reading each triangle as a local patch. A patch is
only trustworthy if its pods are well spread — three pods that are nearly in a
straight line form a razor-thin triangle whose interpolation is numerically
unstable. The robustness of the *whole* network is therefore limited by its
**worst** (smallest-area) triangle.

You must place the `n` pods so that the **minimum triangle area over all triples of
pods is as large as possible**.

This is a **Heilbronn-triangle** placement, a classic extremal point-configuration
problem: it is believed hard, has **no known optimal construction** for these sizes,
and admits many distinct strategies (lattice / algebraic constructions, farthest-
spread heuristics, local search, annealing).

## Input (stdin)
```
n
```
- `n` — the number of pods to place (always prime; `17 <= n <= 53`).

## Output (stdout)
```
<n lines, each: x y>
```
Print the `n` pod coordinates, one `x y` pair per line (real numbers). Extra
trailing whitespace is ignored; only the first `n` pairs are read.

## Feasibility
An output is valid iff **all** hold (tolerance `1e-6`):
- at least `n` coordinate pairs are printed (the first `n` are used);
- every coordinate is finite (no `nan` / `inf`);
- every pod lies in the greenhouse: `0 <= x <= 1`, `0 <= y <= 1`;
- no two pods coincide (pairwise distance `>= 1e-7`).

Any violation scores `Ratio: 0.0`.

## Objective
Maximize
```
F = min over all triples {i,j,k} of  area(pod_i, pod_j, pod_k)
```
where `area` is the ordinary Euclidean triangle area
`0.5 * |(x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i)|`.
Three collinear pods give area `0`, so any near-collinear triple drives `F` toward `0`.

## Scoring
The checker builds its own baseline `B`: the **quadratic-residue point set**
```
P_i = ( i/n , (i*i mod n)/n ),   i = 0 .. n-1
```
Because `n` is prime this set always has every triangle non-degenerate, so `B` (its
exact minimum triangle area) is strictly positive and feasible. With maximization
normalization:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the baseline scores `Ratio = 0.1`; a layout whose minimum triangle area
is `10x` the baseline caps at `1.0`. The optimum is unknown, so headroom is genuine.

## Constraints
- `n` prime, `17 <= n <= 53` (10-case difficulty ladder).
- Time limit 5s, memory 512m.

## Example
For `n = 17` the quadratic-residue baseline has minimum triangle area
`B ≈ 1.73e-3` and scores `Ratio = 0.1`. A layout found by local search whose
smallest triangle has area `F ≈ 6.9e-3` gives
`sc = 100 * 6.9e-3 / 1.73e-3 ≈ 400`, i.e. `Ratio ≈ 0.40`.
