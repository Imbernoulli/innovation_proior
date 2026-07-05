# Harbor Container Port: The Secured-Zone Cordon

**Family:** heuristic-contest-offline · **Format:** B (isolated heuristic evaluation) · **Objective:** maximize

## Story

A container port must fence off **one contiguous secured zone** inside its
rectangular storage yard. The yard is an `H x W` grid of cells. Each cell holds a
container stack with an integer **value** `v[i][j]`:

- **high positive** — a valuable export stack you *want* inside the cordon,
- **near zero / mildly negative** — low-grade cargo,
- **strongly negative** — a hazmat / reefer stack that is costly to keep secured.

You choose a set `S` of cells to enclose. It earns the sum of the values inside `S`
but you must pay for the **fence**: `lam` per unit of **perimeter**, where the
perimeter is the number of unit grid edges that separate a chosen cell from a
non-chosen cell **or** from the yard's outer boundary.

The port authority wants a **single 4-connected secured zone** (you cannot fence
two separate compounds) that maximizes

```
profit(S) = sum_{c in S} v[c]  -  lam * perimeter(S).
```

This is NP-hard. Region growing, best-of-N seeded restarts, and add/remove local
search are all viable; no simple rule is optimal. You are graded on a fixed,
deterministic family of yard instances — an offline heuristic-contest scorer, not
wall-clock time.

## The candidate program (stdin → stdout)

Your program reads ONE JSON object (the **public instance**) from stdin and writes
ONE JSON object to stdout.

### Input (public instance)

```json
{
  "name": "yard101",
  "H": 12,
  "W": 12,
  "lam": 1,
  "grid": [[ v_00, ..., v_0{W-1} ], ..., [ v_{H-1}0, ..., v_{H-1}{W-1} ]]
}
```

- `H`, `W` — yard dimensions (rows × columns).
- `lam` — fence cost per unit of perimeter (a positive integer).
- `grid` — `H` rows of `W` integer container-stack values.

### Output (your cordon)

```json
{"cells": [[i0, j0], [i1, j1], ...]}
```

Each `[i, j]` is a cell inside the secured zone, with `0 <= i < H` and
`0 <= j < W`.

## Validity

A cordon is **valid** iff `cells` is:

- a **non-empty** list of `[i, j]` integer pairs,
- all **in bounds** (`0 <= i < H`, `0 <= j < W`),
- **distinct** (no repeated cell), and
- a **single 4-connected component** (up/down/left/right adjacency).

Any violation — wrong shape, a repeated or out-of-bounds cell, a disconnected or
empty set, a crash, a timeout, or non-JSON output — scores **0.0** on that
instance.

## Scoring (deterministic)

For each instance the evaluator computes two references (in the parent process;
your program never sees them):

- `base = max(0, max_cell (v[c] - 4*lam))` — the best **single-cell** cordon (a
  one-cell zone has perimeter 4). This is the weak anchor.
- `ub   = sum_cell max(0, v[c] - 2*lam)` — an optimistic per-cell upper bound
  (pretends every kept cell can be interior, exposing only 2 fence edges). It is
  generally unreachable.

Your cordon's `profit` is normalized with an affine anchor (weak single-cell cordon
→ 0.1, optimistic ideal → 1.0):

```
r = clamp( 0.1 + 0.9 * (profit - base) / max(1e-9, ub - base), 0, 1 )
```

- Fencing just the single best stack scores ~**0.1**.
- Reaching the (unreachable) optimistic bound scores **1.0**.
- Doing worse than the single best stack scores **< 0.1**.

Because `ub` ignores connectivity and boundary reality, even strong region-growing
plus local search stays well below 1.0 — there is real headroom. The final score is
the mean of `r` over all 10 instances (a per-instance `Vector:` is also printed).

## Isolation

Your program is run **OS-sandboxed** in a fresh subprocess and only ever sees the
public instance. The references (`base`, `ub`), the full instance state, and the
scorer live in the parent process, which the sandbox cannot reach — introspection
or filesystem snooping buys you nothing. Only your stdout JSON is read.
