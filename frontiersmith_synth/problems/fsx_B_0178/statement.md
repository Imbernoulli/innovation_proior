# SwarmRoute: SWAP-Minimal Hand-off Routing on a Drone Mesh

## Problem
A delivery **drone swarm** operates over a fixed mesh of **relay hubs**. Two hubs are
*linked* if drones parked on them are close enough to perform a **synchronized package
hand-off** directly. A hand-off between two drones can only happen while both drones sit
on a pair of linked hubs.

Drone `i` starts parked on hub `i` (the identity layout). You are given a set of
**required hand-offs** — unordered pairs of drones that must each be executed exactly
once during this mission phase. (This is exactly one mixing layer of a QAOA-style
two-qubit interaction schedule transpiled onto a hardware coupling map: drones = logical
qubits, hubs = physical qubits, links = the coupling graph, hand-offs = the required
two-qubit gates.)

Because many required drone pairs are *not* parked on linked hubs, you must **relocate**
drones by issuing `SWAP` moves along links. A `SWAP p q` exchanges the drones currently
on the two linked hubs `p` and `q`. Your job is to emit a program of `SWAP` and `APPLY`
moves that executes **exactly** the required set of hand-offs, using **as few `SWAP`
moves as possible**.

## Input (stdin)
```
n m k
u_1 v_1            # m lines: the mesh links (undirected), 0-indexed hubs
...
a_1 b_1            # k lines: the required hand-offs (unordered drone pairs)
...
```
- `n` hubs / drones (drone `i` starts on hub `i`).
- `m` links; the mesh is connected.
- `k` required hand-offs; every required pair is at graph distance ≥ 2.

## Output (stdout)
A sequence of moves, one per line, optionally ending with `END`:
```
SWAP p q     # swap the drones on linked hubs p,q         (counts toward cost)
APPLY p q    # execute the hand-off of the two drones on linked hubs p,q  (free)
END          # optional; blank lines are ignored
```

## Feasibility
A submission scores **0** unless ALL hold:
- every token is well-formed and every hub id is in `[0,n)`;
- every `SWAP`/`APPLY` uses two **distinct, linked** hubs;
- every `APPLY` targets a **required** hand-off (as the unordered pair of the two drones
  currently on `p,q`), with **no duplicate** and **no non-required** hand-off;
- the multiset of executed hand-offs equals the required set **exactly** (functional
  equivalence to the target circuit).

## Objective
Minimize the number of `SWAP` moves, `F`.

## Scoring
Let `B` be the checker's baseline: route each required pair independently on a fresh
identity layout via a deterministic shortest path, apply it, then undo the swaps
(`B = sum 2·(dist−1)`). With minimization normalization:
```
Ratio = min(1000, 100 · B / max(1e-9, F)) / 1000
```
Reproducing the baseline scores ≈ 0.1; a solution using ~10× fewer swaps caps at 1.0.

## Constraints
- `7 ≤ n ≤ 16`, sparse connected mesh; `k ≈ n` required hand-offs.
- Deterministic scoring only (exact integer counts).

## Example
For a path mesh `0-1-2-3` with required hand-off `{0,3}` (distance 3), the baseline
routes `0` up to hub `2` (2 swaps), applies at `(2,3)`, then undoes (2 swaps): `B=4`.
A solution emitting `SWAP 0 1`, `SWAP 1 2`, `APPLY 2 3` uses `F=2` swaps →
`Ratio = min(1000, 100·4/2)/1000 = 0.2`.
