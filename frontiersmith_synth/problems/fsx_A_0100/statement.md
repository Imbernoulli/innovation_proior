# Tell Qadesh: The Connected Dig Plan

An archaeological survey has mapped a square dig site, **Tell Qadesh**, as an `N x N`
grid of cells. Ground-penetrating radar assigns every cell `(r, c)` an integer
**yield** `v[r][c]`: positive where buried artifacts are expected, negative where the
cell is barren rubble that costs effort to clear.

Your team will excavate exactly **one connected trench** — a set of dug cells that is
**4-connected** (via orthogonal up/down/left/right neighbours) so the pit can be
walked and shored as a single site. Digging cell `(r, c)` collects `v[r][c]`. Every
exposed face of the trench must be shored: the **shoring cost** is `lambda` per unit
of **perimeter**, where the perimeter is the number of grid edges between a dug cell
and either a non-dug cell or the outside of the grid.

Maximise the **plan value**:

```
plan value  =  (sum of v over dug cells)  -  lambda * (perimeter of the dug region)
```

The empty plan (dig nothing) is allowed and has value `0`. This is an offline
heuristic-optimisation task in the spirit of AtCoder Heuristic Contest: a rich
region-selection landscape with a fixed deterministic scorer and no easy optimum.
Unconstrained, the objective is a graph-cut; the **connectivity requirement** makes it
NP-hard, so region growing, hill-climbing, boundary smoothing and seeded local search
all give genuinely different results. Aim to make good use of your compute budget
(the harness runs your program under a per-instance time limit purely as a safety cap;
your *score* depends only on the plan value you output, never on wall-clock time).

## Program contract

Your solution is a standalone program. It reads ONE JSON object (the public instance)
from **stdin** and writes ONE JSON object (your plan) to **stdout**.

### Input (stdin)

```json
{
  "name": "tell101",
  "N": 16,
  "lam": 2,
  "grid": [[ -1, -2, 40, ... ], ...]   // N rows, each with N integer yields
}
```

- `N` — grid side length.
- `lam` — the shoring cost per unit perimeter (`lambda`, a positive integer).
- `grid` — the `N x N` yield matrix; `grid[r][c]` is the yield of cell `(r, c)`,
  with rows `0..N-1` and columns `0..N-1`.

### Output (stdout)

```json
{"cells": [[r0, c0], [r1, c1], ...]}
```

- `cells` — the list of dug cells (0-indexed `[row, col]` pairs).

### Validity

A plan is **valid** iff `cells` is a list of `[r, c]` integer pairs, each with
`0 <= r, c < N`, with **no duplicate cells**, and — if the list is non-empty — the
cells form a **single 4-connected region**. The empty plan `[]` is valid (value `0`).

A disconnected set, an out-of-range or duplicate cell, a non-integer coordinate, a
crash, a timeout, or non-JSON output makes that instance score **0.0**.

## Scoring

You are graded on **12** fixed, deterministically-generated instances (a barren
negative background dotted with positive artifact hotspots; some instances place the
hotspots far apart so bridging them rarely pays, others place them close so a bridge
can pay off). For each instance the evaluator computes:

- `base` = value of the single best cell dug as a size-1 trench
  (`max_c (v[c] - 4*lambda)`) — a weak reference,
- `ub` = sum of all strictly-positive yields with no shoring cost
  (`sum_{v[c] > 0} v[c]`) — an optimistic, generally unreachable bound,
- `cand` = the plan value your trench achieves,

and normalises with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (cand - base) / (ub - base),  0,  1 )
```

Digging only the best single cell scores about `0.1`; reaching the (unreachable)
all-artifacts bound scores `1.0`; doing worse than the best single cell scores below
`0.1`. Your final score is the mean of `r` over all instances. Because `ub` ignores
both connectivity and shoring, even excellent trenches stay well below `1.0` — there
is always headroom to search for a better plan.

## Notes

- The evaluator is deterministic; there is no randomness in the instances or the
  scorer. Any randomness in your own solver should be seeded so your output is
  reproducible.
- Your program is run in an isolated subprocess and only ever sees the public
  instance above; the references and the validity check live in the evaluator.
