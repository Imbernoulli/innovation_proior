# Inverter-Fleet QAOA: SWAP-Minimal Routing on a Fixed Coupling Map

## Problem

A solar farm's fleet of `N` grid-tie inverters is scheduled by a QAOA
(Quantum Approximate Optimization) routine running on a real quantum backend.
Each inverter is assigned a logical qubit. The QAOA **cost layer** applies a
two-qubit interaction (a `ZZ`-rotation) between every pair of inverters that
shares a power-coupling constraint. The set of these required pairs is the
**cost layer** of the circuit.

The backend, however, can only apply a two-qubit gate to a pair of *physical*
qubits that are directly connected on its fixed **coupling map** (an
undirected graph). Initially logical qubit `i` sits on physical qubit `i`
(the identity placement). To execute an interaction between two logical qubits
that are not physically adjacent, you must first insert `SWAP` gates along the
coupling map to bring them together. A `SWAP` on a coupling-map edge exchanges
the two logical qubits currently on those physical qubits.

All cost-layer interactions are diagonal and therefore **commute**: they may be
executed in any order. Your job is to emit a straight-line gate program
(`SWAP`s + interaction firings) that executes every required interaction on
physically adjacent qubits, using **as few SWAP gates as possible**.

## Input (stdin)

```
N M K
u_1 v_1
...
u_M v_M        # M undirected coupling-map edges over physical qubits 0..N-1
a_1 b_1
...
a_K b_K        # K required logical interactions (unordered, distinct pairs)
```

The coupling map is connected. `0 <= u,v,a,b < N`.

## Output (stdout)

A straight-line program, one instruction per line, from the identity placement:

```
S u v      # SWAP the logical qubits on physical qubits u and v; (u,v) must be a coupling-map edge
G a b      # execute the required interaction between logical qubits a and b
```

An interaction `G a b` is legal only if `{a,b}` is a required interaction that
has not been fired yet, and the physical qubits currently holding logical `a`
and `b` are adjacent on the coupling map.

## Feasibility

Your program is feasible iff every instruction is legal at the moment it is
applied AND every one of the `K` required interactions is fired exactly once.
Any violation (bad token, out-of-range qubit, SWAP on a non-edge, firing a
non-required / duplicate / non-adjacent interaction, or leaving an interaction
unfired) scores `0`.

## Objective

Minimize `F` = the number of `S` (SWAP) instructions emitted.

## Scoring

Let `B` be the checker's internal route-then-restore baseline:
`B = 2 * sum over required pairs (a,b) of ( dist(a,b) - 1 )`, where `dist` is
the coupling-map graph distance under the identity placement. With `F` your
SWAP count, the score is

```
Ratio = min(1.0, 0.1 * B / max(1e-9, F))
```

Reproducing the baseline scores `~0.1`; halving the SWAP count doubles the
score; using `<= B/10` SWAPs caps the ratio at `1.0`.

## Constraints

`8 <= N <= 26`, sparse coupling maps, `K` up to `~N + 20`. The reference
checker runs in `O((N+M)*K)` time.

## Example

Coupling map is a path `0-1-2-3`; the only required interaction is `{0,3}`
(distance 3). The baseline restores after routing, spending `2*(3-1) = 4`
SWAPs. A better program `S 0 1 / S 1 2 / G 0 3` moves logical qubit `0` next
to logical qubit `3` with only `2` SWAPs (`F=2`), scoring
`0.1 * 4 / 2 = 0.2`.
