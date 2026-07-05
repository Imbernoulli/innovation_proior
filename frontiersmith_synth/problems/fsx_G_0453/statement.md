# Coupling-Map Routing: Minimal-SWAP Compilation of a Fixed Interaction Schedule

## Problem
An IBM-style superconducting quantum processor exposes a fixed **coupling map**: a set
of `P` physical qubits and a set of undirected edges. A two-qubit gate may be executed
**only between two physically adjacent qubits** (qubits joined by a coupling edge).

You are given a quantum program as an ordered **schedule** of `M` two-qubit interactions
on `L` logical qubits. Interaction `t` names two logical qubits `(a_t, b_t)` that must be
brought together and interact — the interactions must be realized **in the given order**.

You first choose an **initial placement** assigning each logical qubit to a distinct
physical qubit. Whenever the next scheduled interaction's two logical qubits are not on
adjacent physical qubits, you insert **SWAP gates**. A SWAP acts on a coupling edge
`(p, q)` and exchanges the logical contents of physical qubits `p` and `q`. By swapping,
you slide logical qubits across the lattice until the pair for the current interaction is
adjacent, then the interaction executes.

Your job is to compile the whole schedule using **as few inserted SWAP gates as
possible**. (Qubit routing / SWAP minimization is NP-hard; the optimum is unknown.)

## Input (stdin)
```
P E M L
```
then `E` lines, each `p q`, an undirected coupling edge (`0 <= p,q < P`, `p != q`);
then `M` lines, each `a b`, the logical qubits of interaction `t` in schedule order
(`0 <= a,b < L`, `a != b`). The coupling graph is connected. Here `L == P`.

## Output (stdout)
First a **placement line** of `P` integers:
```
place[0] place[1] ... place[P-1]
```
where `place[phys]` is the logical qubit initially sitting on physical qubit `phys`, or
`-1` if that physical qubit starts empty. Every logical qubit `0..L-1` must appear exactly
once.

Then, for each interaction `t = 0..M-1` **in order**, emit its routing block:
```
k   p1 q1   p2 q2   ...   pk qk
```
`k` is the number of SWAPs applied **before** interaction `t`, followed by the `k` swap
edges (each `(pi, qi)` must be a coupling edge), applied left to right. Whitespace and
line breaks are free; the checker reads a flat token stream.

Tokens must be integers. `nan`, `inf`, floats and any non-integer token are **rejected**.

## Feasibility
Starting from your placement, the checker applies each block's swaps in order (each must be
a legal coupling edge) and then requires the two logical qubits of interaction `t` to be on
**physically adjacent** qubits. Any parse error, illegal swap, invalid placement, out-of-range
count, trailing tokens, or a non-executable interaction scores **0**.

## Objective
Minimize `F`, the **total number of inserted SWAP gates** (summed over all blocks).

## Scoring
Let `B` be the SWAP count of a reference baseline (identity placement `logical i -> physical i`,
plus, for each interaction, a greedy shortest-path route that slides the first qubit toward the
second). With your total `F`,
```
Ratio = min(1, 0.1 * B / F)
```
Reproducing the baseline scores `0.1`. Halving the swap count doubles the ratio; reaching a
tenth of `B` caps at `1.0`. Because the true minimum is unknown and lies below what simple
greedy routing achieves, substantial headroom remains.

## Constraints
- `1 <= P <= 400`, connected coupling graph, `L == P`.
- Per-block swap count `0 <= k <= P*P`.
- Deterministic exact-integer scoring; nothing is timed.

## Example
A `3x3` grid (`P = 9`) with a schedule whose baseline needs `B = 20` swaps. A submission
that routes the whole schedule with `F = 10` swaps scores `min(1, 0.1*20/10) = 0.2`. The
identity-placement baseline (`F = 20`) would score `0.1`.
