# Affine Peel: Golfing a Reversible Permutation Circuit

## Problem
You are handed a bijection `P` on the `n`-bit strings `{0,1}^n`, given by its full
truth table (its image on every one of the `2^n` inputs). Build the **cheapest
reversible circuit** that computes `P`, using only the gates

- `NOT t`        — flip wire `t`;
- `CNOT c t`     — `wire[t] ^= wire[c]`;
- `TOF c1 c2 t`  — `wire[t] ^= wire[c1] & wire[c2]` (a Toffoli).

The circuit runs on `n + a` wires: wires `0..n-1` are the data register, wires
`n..n+a-1` are **ancilla**. On input `x` the ancilla start at `0`; after the
circuit wires `0..n-1` must equal `P(x)` and every ancilla wire must be back to
`0`. This must hold for **all** `2^n` inputs simultaneously (the circuit is
straight-line — no measurement, no branching).

## Input (stdin)
```
n a costN costC costT
P(0) P(1) P(2) ... P(2^n - 1)
```
`n` (6..8) is the bit width; `a` the ancilla budget; `costN,costC,costT` the per-gate
weights of `NOT`, `CNOT`, `TOF`. The `2^n` integers are the table of `P` (a
permutation of `0..2^n-1`), whitespace-separated.

## Output (stdout)
Your circuit, one gate per line, in application order (`NOT t` / `CNOT c t` /
`TOF c1 c2 t`, wires in `0..n+a-1`). An empty output means the empty circuit.

## Feasibility
The circuit is accepted only if it realizes `P` **exactly** on all `2^n` inputs
with all ancilla restored to `0`. Controls and target of a gate must be distinct.
Any schema/range violation, or any input on which the circuit disagrees with `P`
or leaves an ancilla dirty, scores `0`.

## Objective (minimize)
Weighted gate cost
```
F = costN * (#NOT) + costC * (#CNOT) + costT * (#TOF).
```
Toffolis are the expensive resource (`costT` is large) — the game is to spend as
few of them as possible.

## Scoring
Let `B` be the cost of a reference **structure-blind** synthesis (compute `P` into
ancilla term-by-term from its Reed–Muller expansion, swap into place, uncompute
with the expansion of `P^{-1}`). Your score is
```
Ratio = min(1, 0.1 * B / F).
```
Matching the reference gives `~0.1`; a `10x`-cheaper circuit caps at `1.0`.

## What makes it hard
Expanding `P` bit-by-bit spends a Toffoli on **every** nonlinear term of **every**
output bit — and the affine mixing smears the same handful of nonlinear terms
across many bits. The affine layer, though, is free-ish: `NOT`/`CNOT` realize any
map `x -> M x XOR c` with no Toffolis at all. The planted instances hide a large
GF(2)-affine shell wrapping a tiny nonlinear core; separate the shell first and
the Toffoli bill collapses onto the core. Discovering *how* to peel it from the
raw table — and how small the residue really is — is the whole problem.

## Example
With `n=7, costT=4`, the structure-blind reference spends about `46` Toffolis
(cost `B≈262`). A circuit that first strips the affine shell needs only `3`
Toffolis (cost `≈45`), scoring `≈0.58`; realizing the peeled core out-of-place
instead costs `6` Toffolis (`≈92`, scoring `≈0.29`). The exact residue size and
affine matrix live in the table you are given — read them out.

## Constraints
`6 ≤ n ≤ 8`, `a = n+2`, time limit 5s, memory 512 MB. The checker is `O(2^n)`.
Same submission ⇒ same score, always.
