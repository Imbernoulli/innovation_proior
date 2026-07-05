# Cold-Chain Qubit Routing: Minimum-SWAP QAOA Transpilation

## Problem
A vaccine cold-chain facility runs a QAOA optimizer on a small quantum backend to
schedule co-inspections. The backend is a **heavy-hex-inspired storage lattice**: a
fixed graph of refrigerated **slots** (physical qubits). Two lots can be jointly
cross-checked (a two-qubit **ZZ interaction**) only while they sit in **adjacent
slots** of this coupling map.

Each **lot** (logical qubit) `i` starts in slot `i` (identity placement). The QAOA
cost circuit prescribes a multiset of required ZZ interactions between pairs of lots
(one occurrence per problem edge per layer). Because interacting lots are usually not
in adjacent slots, you must physically **SWAP** lots between adjacent slots to bring
each interacting pair together before applying its interaction. Every SWAP is a
cold-chain transfer, so you must **minimize the number of SWAPs** while realizing the
target circuit exactly.

Since ZZ terms commute, only the *multiset* of applied interactions must match the
target — the order is free.

## Input (stdin)
```
Q E K
u_1 v_1
...            (E coupling edges, undirected, physical slots 0..Q-1)
u_E v_E
a_1 b_1
...            (K required interactions, unordered lot pairs; multiplicity matters)
a_K b_K
```
Lot `i` initially occupies slot `i`. The coupling graph is connected.

## Output (stdout)
A straight-line SWAP schedule, one instruction per line:
```
S p q     apply a SWAP between slots p and q (must be a coupling edge); swaps their lots
G a b     apply the ZZ interaction between lots a and b (their current slots must be adjacent)
```

## Feasibility
- Every `S p q`: `{p,q}` must be a coupling edge.
- Every `G a b`: the pair `{a,b}` must be a required interaction with remaining
  multiplicity > 0, and lots `a,b` must currently occupy adjacent slots.
- At the end, every required interaction must be applied **exactly** its required
  number of times (exact circuit equivalence). Any violation scores 0.

## Objective (minimize)
`F` = the number of `S` (SWAP) instructions.

## Scoring
Let `B` be an internal route-and-undo baseline built by the checker (for each required
interaction, bring the pair together along a shortest slot-path, apply, then undo).
```
ratio = min(1.0, 0.1 * B / max(1e-9, F))
```
Reproducing the baseline scores ~0.1; a 10x reduction in SWAPs caps at 1.0.

## Constraints
- `Q` up to ~30 slots; `K` up to ~80 interactions.
- Integer arithmetic only; scoring is fully deterministic.

## Example
For a 2-slot line `0-1` with the single required interaction `0 1`, the lots already
occupy adjacent slots, so the schedule `G 0 1` uses `F=0` SWAPs (baseline `B=0`; this
degenerate case does not occur in the graded instances, which always require routing).
A worked non-trivial case: if lot 0 is in slot 0 and lot 3 in slot 3 on a path
`0-1-2-3`, then `S 0 1`, `S 1 2` brings lot 0 to slot 2 (adjacent to slot 3), and
`G 0 3` applies the interaction — 2 SWAPs versus the route-and-undo baseline's 4.
