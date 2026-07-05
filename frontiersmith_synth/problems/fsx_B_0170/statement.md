# Honeycomb Waggle Router

## Problem

An apiary is a **honeycomb comb**: a set of `nq` hexagonal cells (the *physical
qubits*). Two cells are **adjacent** iff they share a wall; a wall is the only
place a direct two-bee **waggle** (a diagonal `RZZ` phase interaction) may be
performed. The comb's walls form the hardware coupling map.

Every **bee** (a *logical qubit*) starts in its home cell: bee `i` begins in
cell `i` (identity placement).

You are given a **nectar-sharing schedule**: a multiset of required pairwise
waggles between bees. Because every waggle is a diagonal `RZZ` rotation, they all
**commute**, so the target unitary is fixed by *which* bee-pairs interact and
*how many times* each pair interacts — the order is irrelevant.

The comb only lets you waggle bees in **adjacent** cells. To realise a waggle
between two bees that are not adjacent, you must first walk bees around the comb
using **SWAP** moves. A `SWAP` exchanges the two bees occupying an adjacent pair
of cells. Your job: emit a program of `SWAP` and waggle (`G`) instructions that
realises the whole schedule exactly, using **as few SWAP moves as possible**.

## Input (stdin)

```
nq ne
a b            (ne lines: an undirected hardware wall between cells a and b)
...
m
u v            (m lines: one required waggle between bee u and bee v)
...
```

- Cells and bees are `0`-indexed, `0 <= a,b,u,v < nq`.
- The comb graph is connected. Repeated `u v` lines encode multiplicity.
- Initial placement is the identity: bee `i` occupies cell `i`.

## Output (stdout)

A sequence of instructions, one per line:

- `S a b` — SWAP the bees currently in cells `a` and `b`. Requires that `a` and
  `b` share a wall (are a hardware edge).
- `G a b` — perform a waggle between the two bees currently in cells `a` and `b`.
  Requires that `a` and `b` share a wall.

Blank lines are ignored.

## Feasibility

Simulate the program from the identity placement. It is **feasible** iff:

1. every `S`/`G` acts on a real hardware edge (`a != b`, wall exists), and
2. the multiset of bee-pairs actually waggled (mapped back to bee identities via
   the current placement at each `G`) equals the required schedule **exactly** —
   no missing, no extra, correct multiplicity.

Any violation scores `0`.

## Objective (minimize)

`F` = the number of `S` (SWAP) instructions emitted. Fewer is better.

## Scoring

Deterministic. The checker builds an internal baseline `B` = a naive per-waggle
router that processes the schedule in the given order and, for each waggle, walks
one bee along a BFS shortest path until adjacent, then waggles. With your op
count `F`:

```
Ratio = min(1000, 100 * B / max(1e-9, F)) / 1000
```

Reproducing the naive baseline scores about `0.1`; a `10x` reduction in SWAP
moves caps the ratio at `1.0`.

## Constraints

- `nq` up to a few hundred cells; schedules up to a few hundred waggles.
- Exact integer arithmetic only; scoring is bit-for-bit deterministic.

## Example (worked score)

Suppose a schedule needs `B = 900` SWAPs under the naive baseline. If your router
emits `F = 300` SWAPs, `Ratio = 100 * 900 / 300 / 1000 = 0.30`. Emitting `F = 90`
SWAPs gives `100 * 900 / 90 / 1000 = 1.0` (capped).
