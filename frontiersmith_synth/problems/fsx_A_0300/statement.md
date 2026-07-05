# Lechuguilla Deep: The Survey Loop

A caving expedition is mapping a large horizontal level of a cave. A previous trip
bolted a set of numbered **survey stations** to the rock at fixed 2-D coordinates
(metres on the survey grid). Ground-penetrating sonar has located a set of
**formations** — crystal galleries, gypsum chandeliers, aragonite bushes — each at a
known point and each worth an integer amount of **documentation value**.

The expedition will string **one closed survey loop**: a rope traverse that visits an
ordered subset of the stations and returns to its start, forming a **simple**
(non self-intersecting) polygon whose vertices are stations. Every formation that ends
up **strictly inside** the loop lies in the mapped chamber and its value is documented.
Stringing rope costs effort in proportion to the loop's length.

This is an AtCoder-heuristic-contest-style **offline optimisation**: a polygon-selection
problem over a single fixed instance, scored by a deterministic contest formula, with no
easy optimum. Enclosing everything wastes rope reaching far, low-value formations; a
single tight triangle misses most galleries; the good plans hug a chosen subset of
formation clusters. Region growing, cluster selection, convex-hull hugging and local
search all give genuinely different loop values.

## Objective (maximise)

For a loop with vertex sequence `V` (stations, cyclic order):

```
loop value = sum( value(f) for every formation f strictly inside V )
             - lam * perimeter(V)
```

`perimeter(V)` is the Euclidean length of the closed polygon; `lam` is the rope-cost
coefficient. A formation exactly **on** an edge is **not** counted (boundary excluded).

## You write an isolated program (stdin -> stdout)

Read ONE JSON object (the public instance) from stdin, write ONE JSON object to stdout.

### Input (public instance)

```json
{
  "name": "cave301",
  "S": 1000,
  "lam": 0.15,
  "stations": [[0,0], [1000,0], [1000,1000], [0,1000], [172,181], ...],
  "features": [[281, 274, 63], [720, 291, 55], ...]
}
```

* `S` — coordinate extent; stations `0..3` are the corners `(0,0),(S,0),(S,S),(0,S)`.
* `stations[i]` — integer coordinates of station `i`.
* `features[j]` — `[x, y, v]`, an integer formation point of positive value `v`.
* No three stations are collinear (general position).

### Output (your loop)

```json
{"tour": [4, 9, 17, 12]}
```

`tour` is the cyclic list of station **indices** forming the loop.

### Validity

A loop is valid iff `tour` is a list of **>= 3 distinct integer** indices, each in
`0 <= i < len(stations)`, whose polygon (in the given cyclic order) is **simple**: no two
non-adjacent edges intersect and no three consecutive vertices are collinear. A
self-intersecting loop, an out-of-range / duplicate / non-integer index, fewer than 3
vertices, a crash, a timeout, or non-JSON output scores **0.0** on that instance.

## Scoring (deterministic)

Per instance the evaluator computes, itself:

* `base` = value of the **best single triangle** over 3 stations (a weak reference; > 0).
* `ub`   = sum of **all** formation values with **no** rope cost (an optimistic,
  unreachable upper bound).
* `cand` = your loop's value.

and normalises

```
r = clamp( 0.1 + 0.9 * (cand - base) / (ub - base), 0, 1 )
```

Stringing only the best triangle scores about `0.1`; the unreachable
all-formations-no-rope bound scores `1.0`; doing worse than the best triangle scores
below `0.1`. Because `ub` ignores rope and enclosure geometry, even excellent loops stay
well below `1.0`. The reported score is the mean `r` over 12 seeded instances (including
larger, denser held-out caves). Everything is deterministic; there is no wall-time term.

## Isolation

Your program runs in a fresh OS-sandboxed subprocess and sees only the public instance.
The references and the simplicity / strict-containment checks are computed by the
evaluator process, which your program cannot reach.
