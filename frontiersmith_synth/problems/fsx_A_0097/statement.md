# Reef Manifold: Maximizing Distinct Serviceable Reaches

## Problem
An aquarium's plumbing wall is a **manifold**: a rail on which you mount **standoff
fittings**. Each fitting is machined to an integer **offset length** (in mm), and you
may install at most `k` fittings, each with a **distinct** offset drawn from the range
`[0, V]`.

To route water to a valve, the pump chains **exactly two fittings** (the two may be the
**same** fitting used twice) and the valve is served at the **reach length** equal to the
**sum** of the two offsets. A valve can be serviced at reach `r` iff `r = a + b` for some
(not necessarily distinct) chosen offsets `a, b`.

You want the manifold to service as many **distinct reach lengths** as possible. In additive
terms, the set of serviceable reaches is the **sumset** `A + A = { a + b : a, b in A }`, and
your goal is to **maximize `|A + A|`** — the number of distinct pairwise sums (including
doubles `a + a`).

This is the classical **B_2 / Sidon packing** objective: distinct sums are maximized when the
pairwise sums rarely collide, but the offsets must all fit inside the bounded range `[0, V]`,
so you cannot simply space them out arbitrarily.

## Input (stdin)
```
k V
```
- `k` — the maximum number of fittings you may install (the size budget).
- `V` — the inclusive upper bound on an offset; every offset is an integer in `[0, V]`.

## Output (stdout)
```
m
<m lines, each one integer = an installed offset>
```
Print the number of installed fittings `m` (with `0 <= m <= k`), then the `m` offsets, one
per line.

## Feasibility
An output is valid iff **all** hold:
- `0 <= m <= k`;
- exactly `m` integer offsets follow;
- every offset is an integer with `0 <= offset <= V`;
- the offsets are **pairwise distinct**.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = |A + A|`, the number of **distinct** values of `a + b` over all ordered or
unordered pairs `a, b` in `A` (doubles `a + a` count). The sumset is computed exactly with
integer arithmetic.

## Scoring
Let `B` be the size of the checker's own trivial construction: the **contiguous ramp**
`{0, 1, ..., m0-1}` with `m0 = min(k, V+1)` fittings. Its sumset is the contiguous block
`{0, 1, ..., 2*m0-2}`, so `B = 2*m0 - 1`.
With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the ramp scores `Ratio = 0.1`; a manifold servicing `10x` more distinct reaches
caps at `1.0`.

## Constraints
- `12 <= k <= 40`.
- `k <= V <= 2 * k * k` (the range is always wide enough for the ramp baseline, and
  sometimes too tight for a perfect Sidon set — that is where the problem gets hard).
- Time limit 5s, memory 512m.

## Example
Suppose `k = 4`, `V = 20`. The ramp `{0,1,2,3}` has sumset `{0,1,2,3,4,5,6}`, so `B = 7`
and reproducing it scores `0.1`. The Sidon set `{0,1,4,9}` has all six pairwise sums plus
four doubles distinct: `A+A = {0,1,2,4,5,8,9,10,13,18}`, giving `F = 10`,
`sc = 100*10/7 = 142.857`, `Ratio = 0.142857`.
