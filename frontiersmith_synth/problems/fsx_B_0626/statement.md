# Conjugated Shuffle: Minimal-Weight Reversible Resynthesis

## Problem
You are given the complete behavior table of a mysterious "shuffling machine": a bijection
`pi` on `n`-bit register states (`N = 2^n` states total), specified as `pi[x]` for every
`x = 0..N-1`. The internal wiring is gone; only the observed input/output table survives.
Your job is to rebuild the machine as a straight-line **reversible circuit** — a sequence
of generalized controlled-NOT gates acting on the `n`-bit register — that reproduces `pi`
exactly, using as little total gate weight as possible.

A gate is specified by a set of **controls** (each a register bit together with a required
value 0 or 1) and a **target** bit (distinct from every control bit). Applying a gate flips
the target bit of the current register value **iff every control condition holds**; a gate
with 0 controls is an unconditional NOT, 1 control is a CNOT, 2 controls is a Toffoli, and
more controls give a wider gate. Gates are applied to the register in sequence, starting
from register value `x`, and the final register value must equal `pi[x]` — for **every**
`x` simultaneously (the same circuit must work for all `2^n` starting states).

Gate weight: 0 controls costs **1**, 1 control costs **2**, 2 or more controls costs a flat
**5** (wider gates aren't cheaper, but they aren't punished beyond the Toffoli rate either).

Some of these machines are secretly built by first re-labelling the register with a fixed,
invertible linear (XOR-based) change of basis, then wiring a small, local shuffle that only
looks at a handful of the re-labelled bits and leaves the rest completely alone, then
undoing the re-labelling. In the *original* labelling this can look like a fully generic,
unstructured permutation — but if you find the right basis, most of the register turns out
to be inert wiring and only a small piece needs any real logic at all.

## Input (stdin)
```
n
pi[0] pi[1] ... pi[N-1]
```
`pi` is a permutation of `0..N-1` (`N = 2^n`), `4 <= n <= 12`.

## Output (stdout)
```
L
```
followed by `L` gate lines, each:
```
k c1 p1 c2 p2 ... ck pk t
```
`k` = number of controls (`0 <= k <= n-1`); each `(ci, pi)` is a control bit index
(`0 <= ci < n`, all distinct) and its required value (`0` or `1`); `t` is the target bit
index (`0 <= t < n`, distinct from every control bit). `0 <= L <= 20000`.

## Feasibility
Every token must parse as an integer with legal bounds (no `nan`/`inf`/out-of-range/
duplicate/malformed gates). Simulating the declared circuit on **all** `N` starting
register values must reproduce `pi[x]` exactly for every `x`. Any violation scores 0.

## Objective
Minimize the total weighted gate count `F = sum` over gates of `weight(k)` (`1` for
`k=0`, `2` for `k=1`, `5` for `k>=2`).

## Scoring
The checker builds its own reference circuit for `pi` — a structure-blind construction
that fixes up one input at a time via a fixed-length, full-register-width gate schedule
per correction — with weighted cost `B_raw`, and sets `B = 0.6 * B_raw`. With your
circuit's cost `F`:
```
Ratio = min(1, 0.1 * B / F)
```
Smaller `F` is better; the reference construction scores well below `0.1 * B/B_raw = 0.06`
of the ceiling, so there is real room both above and below it. The true minimum-weight
circuit is not known to be efficiently computable in general, so the ceiling stays open.

## Constraints
- `4 <= n <= 12`, `N = 2^n`.
- Time limit 5s, memory 512MB. Deterministic scoring; no timing measurements.

## Example
For a small machine with `n=3` where `pi` is the transposition swapping states `3` (`011`)
and `5` (`101`) and fixing every other state, one valid circuit is `L=3` gates:
`2 0 1 2 0 1` (controls: bit0=1, bit2=0; target bit1), then
`2 0 1 1 0 2` (controls: bit0=1, bit1=0; target bit2), then repeat the first gate.
Simulating this on all 8 states reproduces `pi` exactly, at weighted cost `5+5+5=15`
(illustrative shape only — not the hidden structure of the actual test instances, which
are much larger).
