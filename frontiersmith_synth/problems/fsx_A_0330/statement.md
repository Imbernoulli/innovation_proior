# Alpine Lift Circuit: Overnight Inspection Route

## Story

A large ski resort must inspect every lift station on the mountain before opening.
A single snowcat leaves the **base depot**, drives to each **lift station** exactly
once, and returns to the depot. Each station has a 3-D map position
`(x, y, elevation)`. Because climbing costs fuel, the snowcat's travel cost between
two points is the **rounded 3-D Euclidean distance** (horizontal distance combined
with vertical climb).

Your job: choose the **order** in which the snowcat visits the stations so the total
length of the closed inspection circuit is as **short** as possible.

This is an offline AtCoder-heuristic-style routing contest (a metric Travelling
Salesman instance over lift stations). Scoring is a **fixed deterministic formula**
under a fixed local-search **operation budget** — never wall-clock time.

## Candidate program contract (isolated stdin -> stdout)

Your program is a standalone process. It reads **one** JSON object (the public
instance) from stdin and writes **one** JSON object (your answer) to stdout. It never
sees anything but the public instance.

### Input (stdin): one JSON object

```json
{
  "name": "resort3101",
  "n": 60,
  "coords": [[x0, y0, z0], [x1, y1, z1], ...]
}
```

- `n` — the number of lift stations (does **not** include the depot).
- `coords` — a list of length `n + 1`. Index `0` is the **base depot**; indices
  `1..n` are the lift stations. Every entry is `[x, y, z]` with non-negative integer
  coordinates.

### Output (stdout): one JSON object

```json
{ "order": [i_1, i_2, ..., i_n] }
```

- `order` must be a **permutation of the integers `1..n`** — every station visited
  exactly once, no depot index, no repeats, nothing out of range.
- The circuit driven is `depot(0) -> i_1 -> i_2 -> ... -> i_n -> depot(0)`.

### Distance

For points `a` and `b`:

```
d(a, b) = round( sqrt( (xa-xb)^2 + (ya-yb)^2 + (za-zb)^2 ) )
```

### Objective (minimize)

```
cost(order) = d(depot, i_1) + sum_{k} d(i_k, i_{k+1}) + d(i_n, depot)
```

## Scoring (deterministic)

For each instance the evaluator computes:

- `b` — the cost of the **roster order** `[1, 2, ..., n]` (the do-nothing baseline).
- `obj` — the cost of **your** circuit.

Your normalized score for that instance is

```
r = clamp( 0.1 * b / max(obj, 1e-9),  0,  1 )
```

so reproducing the roster order scores exactly **0.1**, a longer circuit scores
**below 0.1**, and shorter circuits score higher. No metric TSP circuit can beat its
own lower bound, so even excellent routes stay **below 1.0** (there is real
headroom). The final `Ratio` is the mean of `r` over all `n_instances` (a spread of
station counts, including several larger held-out mountains).

Invalid output — wrong length, a repeat, a missing/extra index, a non-integer, a
crash, a timeout, or non-JSON — scores **0.0** on that instance.

## Suggested strategies (open-ended)

- **Roster order** (baseline): output `[1..n]`. Scores ~0.1.
- **Greedy nearest-station chain**: from the depot, repeatedly hop to the closest
  un-inspected station. Big win, but leaves long crossing "return" hops.
- **Nearest-neighbour + 2-opt**: uncross circuit edges by reversing segments whenever
  it shortens the closed tour, iterating under a fixed pass budget.
- **Multi-restart / or-opt**: several seeded starts plus single-station relocations
  and 2-opt to squeeze out the remaining slack, all within the op budget.
