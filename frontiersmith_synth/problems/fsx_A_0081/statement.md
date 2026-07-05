# Queenless Apiary: Raid-Free Hive Activation

## Problem
An apiary is laid out as a combinatorial lattice of hive slots. Each slot is addressed
by an `n`-tuple of *marker rings* over the three colors `{0,1,2}` (yellow, amber, black),
so there are exactly `3^n` slots, one per string in `{0,1,2}^n`.

Some slots are **flooded** (unusable this season). You may **activate** any subset of the
remaining slots, but robber bees exploit a resonance called a **raiding line**: three
*distinct* activated hives `a`, `b`, `c` such that, in **every** ring position `i`,
`(a_i + b_i + c_i) mod 3 == 0`. (Equivalently, `c` is the unique third hive that completes
the line through `a` and `b`.)

The queen wants as many hives running as possible without ever forming a raiding line.
This is exactly the **cap set** condition over `F_3^n`.

## Input (stdin)
```
n
b
<b lines, each an n-character ternary string = a flooded slot>
```
`n` is the address length; `b` is the number of flooded slots. Flooded strings are distinct.

## Output (stdout)
```
k
<k lines, each an n-character ternary string = an activated hive>
```
Print the number of activated hives `k`, then the `k` addresses, one per line.

## Feasibility
An output is valid iff **all** hold:
- each address is a string of exactly `n` characters, each in `{0,1,2}`;
- the `k` addresses are pairwise distinct;
- no activated address is a flooded slot;
- no three distinct activated hives form a raiding line.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = k`, the number of activated hives (a cap set in `F_3^n` avoiding flooded slots).

## Scoring
Let `B` be the size of the checker's own trivial construction: the **ring diagonal**
`{ 0^n, e_1, ..., e_n }` (the all-zero address plus each single-1 address), restricted to
non-flooded slots. This is always a valid raid-free set, so `B = n + 1`.
With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the diagonal scores `Ratio = 0.1`; a set `10x` larger caps at `1.0`.

## Constraints
- `3 <= n <= 5`, so `27 <= 3^n <= 243`.
- The `n+1` ring-diagonal slots are never flooded (the baseline is always available).
- Time limit 5s, memory 256m.

## Example
Suppose `n = 3` and no slots are flooded. The diagonal `{000, 100, 010, 001}` has `B = 4`
and is raid-free, scoring `0.1`. A cap set of size `9` (e.g. a maximum cap set in `F_3^3`)
gives `F = 9`, `sc = 100*9/4 = 225`, `Ratio = 0.225`.
