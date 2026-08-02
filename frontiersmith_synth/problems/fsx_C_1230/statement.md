# Picking Versions That Can All Coexist

## Problem
A project manifest lists `n` packages, indices `0..n-1` **in a fixed
declaration order**. Package `i` has `m_i` available versions, numbered `1`
(oldest) through `m_i` (newest); each version carries a preference weight.
Some `(package, version)` pairs also carry **requirement edges**: "if
package `i` is installed at version `v`, package `j` must be installed at
some version in `[lo, hi]`." Every requirement edge points to a package
with a **strictly larger index** than its source (`j > i`), so the
dependency structure is acyclic in declaration order. Every package must
be installed at exactly one version — there is no "leave it out" option.

## Input (stdin)
```
n
for each package i = 0..n-1:
  m_i
  for each version v = 1..m_i:
    pref_i_v  r
    j_1 lo_1 hi_1  j_2 lo_2 hi_2  ...  (r triples, this version's requirements)
```
All tokens are whitespace-separated (a version's line holds its preference,
its requirement count `r`, then `r` triples `j lo hi`).

## Output (stdout)
`n` integers `c_0 ... c_{n-1}` with `1 <= c_i <= m_i` — the installed
version of every package, in index order (whitespace-separated, any
layout).

## Feasibility
Output scores `Ratio: 0.0` if: the token count differs from `n`; any token
is non-finite, non-integral, or outside `[1, m_i]`; or, for **any** package
`i` at its chosen version `c_i` and **any** of that version's requirement
edges `(j, lo, hi)`, `c_j` falls outside `[lo, hi]`.

## Objective
Maximize total preference of the installed set:
```
F = sum_i pref_i[c_i]
```
subject to every requirement edge holding under the final assignment.

## Scoring
The checker's own reference `B` is the universal fallback "install every
package at version 1" — always feasible, since every version-1 requirement
range in every test case is constructed to contain `1`.
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching `B` scores `0.1`; the cap leaves generous headroom above any
reference solution.

## Constraints
`1 <= n <= 40`, `1 <= m_i <= 8`, `1 <= pref_i_v <= 100`, each version's
requirement count `r <= 3`. Time limit 5s, memory 512MB.

## Example
(a different, unrelated shape — illustrates the mechanics only, not the
harder planted cases.)

`n=3`. Package 0: v1 `pref=5, r=0`; v2 `pref=9, r=1` requiring package 2 in
`[2,2]`. Package 1: v1 `pref=4, r=0`; v2 `pref=8, r=0`. Package 2: v1
`pref=3, r=0`; v2 `pref=7, r=0`.

Output `2 2 2`: package 0 is at v2, which requires package 2 in `[2,2]`,
and package 2 is at 2 — satisfied. `F = 9+8+7 = 24`. `B` (all version 1)
`= 5+4+3 = 12`. `Ratio = min(1000, 200)/1000 = 0.2`.

Output `2 2 1` instead: package 0 is at v2, requiring package 2 in `[2,2]`,
but package 2 is at 1 — infeasible. `Ratio: 0.0`, even though package 1
looks perfectly fine on its own.

## Note on structure
Two different packages can each carry their own requirement range on the
*same* later package. Whether their ranges leave any room for that shared
package depends on which versions of the two sources are chosen — some
combinations leave no room at all, others leave several rooms of
different value, and switching which of the two sources gives way is not
always the version-number-based fix it looks like. None of this coupling
is spelled out in this statement; it lives entirely in the requirement
edges of each specific input, and different packages can be coupled to
completely different, unrelated later packages within the same instance.
