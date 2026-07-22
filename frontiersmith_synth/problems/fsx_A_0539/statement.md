# The Lock Only Real Keys Visit

## Problem
A hardware lock reads an `n`-bit key `x = x_0 x_1 ... x_{n-1}` and must raise a single
output bit. In the field, only a known list of **real keys** is ever presented — the
`m` care vectors below, each with the output the lock must produce. On every other
`n`-bit pattern the output is a **don't-care**: you may return anything. Behaviour is
graded *only* on the real keys.

Build the cheapest **combinational circuit** (a straight-line GF(2) program) whose output
matches the required bit on every real key, using **as few gates as possible**.

## Input (stdin)
```
n m
```
then `m` lines, each an `n`-character string of `0/1` (the key, `x_0` first) followed by
one output bit `0/1`. All `m` keys are distinct. `1 <= n <= 40`, `m <= 2^15`.

## Output (stdout)
```
G
```
then `G` gate lines, then `OUT w`. Wires are numbered: inputs are wires `0..n-1`
(`wire i = x_i`); the `g`-th gate line (0-indexed) creates wire `n+g`. Every gate may
reference only **strictly earlier** wires. A gate is one of:
```
AND a b      OR a b      XOR a b      NOT a      ONE      ZERO
```
(`ONE`/`ZERO` take no operands; they are the constants 1 and 0.) `OUT w` names the wire
carrying the result, `0 <= w < n+G`. `0 <= G <= 500000`.

## Feasibility
Evaluate the circuit on each real key `x`. If the output wire disagrees with the required
bit on **any** real key, the submission scores `0`. (Malformed output — bad tokens, a wire
that references a later/own wire, a non-integer or non-finite index, trailing tokens —
also scores `0`.)

## Objective (minimize)
`G`, the number of gate lines. Fewer gates is better.

## Scoring
Let `B` be a reference gate count the judge builds itself: it recovers the affine hull of
the real-key set, rewrites the target in free coordinates as a Reed–Muller (ANF) form, and
counts a naive **one-AND-per-monomial** circuit for it. Your ratio is
```
ratio = min(1.0, 0.1 * B / G)
```
so reproducing the reference scores about `0.1`, and a circuit `10x` smaller caps at `1.0`.
The score is the mean ratio over all cases.

## What makes it hard
The real keys do not fill the cube — they all lie on a low-dimensional **affine flat**.
Two moves compound:
* **Don't-care exploitation.** You never have to be correct off the flat, so you should
  synthesize the function *restricted to the flat*, not complete it to a total function and
  synthesize that (which is far larger).
* **Affine collapse.** On the flat, the target is a quadratic GF(2) form. Its cost in AND
  gates is governed by its **symplectic rank**, not by how many monomials it shows in raw
  coordinates. A form can look dense (many `x_i x_j` terms) yet have low rank: after the
  right linear change of coordinates it becomes `f_1·h_1 XOR ... XOR f_s·h_s` with only `s`
  products. Detecting that structure is worth far more than locally factoring the monomials
  you happen to see. The exact form differs per case and is carried entirely by the input —
  you must read it out of the real keys, not guess it from this statement.

## Example
`n = 3`, real keys `{000→0, 011→0, 101→0, 110→1}` (these four lie on the flat
`x_0 XOR x_1 XOR x_2 = 0`, so `x_2` is determined by `x_0, x_1`). One valid answer:
```
1
AND 0 1
OUT 3
```
which computes `x_0 AND x_1` — check it matches all four keys. It uses `G = 1` gate (the
`OUT` line is not a gate). Whether that is near-optimal depends on the case; this tiny
instance is illustrative only.
