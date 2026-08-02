# Place the Interaction Graph First: Qubit Routing on a Branchy Coupling Graph

## Problem
A quantum device has `n` physical qubits `0..n-1` connected by a fixed
**coupling graph**: two-qubit gates can only be applied directly between
physical qubits that are graph-adjacent. Your logical program has `n`
logical qubits and a fixed ordered list of `m` two-qubit gates
`gate_1, ..., gate_m` (each gate has a type `t in {0,1}` and an unordered
pair of logical qubits). Every gate must be executed, in this exact order
(a gate's two logical qubits must be graph-adjacent physical qubits *at the
moment it runs*), but whenever the two logical qubits currently in play
aren't adjacent, you may insert `SWAP` operations on adjacent physical
qubits to move them closer. A `SWAP` costs 3 elementary operations (it takes
3 real two-qubit gates to implement); running one program gate costs 1.

There is a second, free lever: if two consecutive-in-program occurrences of
the *same* type on the *same* logical pair have nothing else touching either
qubit strictly between them (in program order — swaps elsewhere don't
count), they compose to the identity and **both may be omitted** (you must
omit both or neither).

You choose a starting placement (which physical qubit each logical qubit
begins on) and then the sequence of `SWAP`/execute operations. Minimize the
total elementary-operation count. The coupling graph in every test case is
connected but sparse (about `n-1` edges) — it may be a simple chain or a
branchy, tree-like shape; read the edges, don't assume a shape.

## Input (stdin)
```
n m e
e lines: u v            (coupling-graph edge, physical qubit ids 0..n-1)
m lines: t a b           (gate i: type t in {0,1}, logical qubits a,b, a!=b)
```
Gates are given in program order (gate 1 first).

## Output (stdout)
```
n
p_0 p_1 ... p_{n-1}       (initial mapping: logical i starts on physical p_i;
                            must be a permutation of 0..n-1)
T
T lines, each one of:
S p q                      SWAP physical qubits p,q (must be a coupling edge)
G i                        execute gate i (1-indexed, strictly increasing
                            across all G lines in the output; you may omit an
                            index only as part of a valid cancelling pair)
```

## Feasibility
The output is replayed against the evolving mapping. It is rejected
(score 0) if: `p_0..p_{n-1}` is not a permutation; any `S p q` is not a
coupling-graph edge; any `G i` fires while its two logical qubits are not
currently on adjacent physical qubits; `G` indices are not strictly
increasing; or an omitted gate index is not part of a genuine cancelling
pair (same type, same unordered logical pair, no other gate touching either
qubit strictly between them in program order) with its partner also
omitted.

## Objective
Minimize `F = 3 * (#SWAPs) + 1 * (#gates actually executed)`.

## Scoring
The checker independently builds a baseline `B`: identity initial mapping,
no cancellation, and every gate routed by detouring through a fixed anchor
physical qubit 0 (move one endpoint to qubit 0, then from qubit 0 to the
other endpoint). `Ratio = min(1, 0.1 * B / F)`.

## Constraints
`6 <= n <= 32`, `1 <= m <= 500`, coupling graph connected,
`1 <= T <= 20000`. Time limit 4s, memory 512MB.

## Example
`n=4`, path graph `0-1-2-3`, gates: `(0,0,3)` then `(0,0,3)` again (an
immediate repeat — cancels for free, 0 cost) then `(1,1,2)` (already
adjacent under the identity mapping — 0 swaps, cost 1). A valid output:
mapping `0 1 2 3`, `T=1`, single line `G 3` (gates 1,2 are omitted as a
cancelling pair; gate 3 costs 1). `F=1`. This tiny example is for output
mechanics only — real instances need real routing decisions.
