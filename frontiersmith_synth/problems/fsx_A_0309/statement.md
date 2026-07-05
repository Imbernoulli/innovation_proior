# Watchtowers Over the Reserve: Minimizing Sector Blind Spots

## Problem
A fire-management agency oversees a square nature reserve modeled as the unit
region `[0,1]^d` (`d = 2` is the flat map; `d = 3` adds a normalized
elevation band). You must place exactly `M` watchtowers inside the reserve.

Fire risk is audited over **anchored sectors**: every axis-aligned box of the
form `[0, a_1) x [0, a_2) x ... x [0, a_d)` (its lower corner pinned at the
reserve's origin). A sector's *coverage imbalance* is the gap between the
fraction of towers that fall inside it and the fraction of the reserve's area
(volume) it occupies:

```
imbalance(a) = | (#towers inside [0,a)) / M  -  volume([0,a)) |
```

Your placement is judged by its **worst** anchored sector — the classical
**star discrepancy** of the tower set:

```
D*  =  sup over all anchored boxes [0,a)  of  imbalance(a)
```

A small `D*` means no anchored sector is ever badly over- or under-watched.

## Input (stdin)
One line with two integers:
```
d M
```
`d` is the reserve dimension (2 or 3) and `M` is the number of towers to place.

## Output (stdout)
`M` lines, each with `d` real numbers in `[0,1]` giving the coordinates of one
tower (whitespace-separated; extra precision is fine). Output exactly `M * d`
numbers total.

## Feasibility
- Exactly `M * d` numeric tokens must be emitted.
- Every coordinate must be finite and lie in `[0,1]` (a tolerance of `1e-12`
  is allowed at the boundary).
Any violation scores `0`.

## Objective
**Minimize** the exact star discrepancy `D*` of your tower set. The checker
computes `D*` exactly: the supremum is attained on the finite grid whose
per-axis breakpoints are the tower coordinates (plus the boundary `1.0`), and
each grid corner is tested as both a closed box (towers `<=` corner) and an
open box (towers `<` corner).

## Scoring
Let `F = D*` of your towers and let `B = D*` of the checker's internal
**ranger-diagonal** baseline — the `M` towers placed at `((i+0.5)/M, ...,
(i+0.5)/M)` for `i = 0..M-1`. The score is

```
Ratio = min(1000, 100 * B / F) / 1000
```

Reproducing the diagonal baseline yields `Ratio ~ 0.1`; halving the worst-case
imbalance roughly doubles the score; a `10x`-better placement caps at `1.0`.
There is no known optimal construction — minimizing star discrepancy for a
given `M` is an open problem, so many strategies remain competitive.

## Constraints
- `d in {2, 3}`, `8 <= M <= 30`.
- Deterministic scoring: the objective is exact-arithmetic geometry with a
  fixed `1e-12` tolerance; no randomness, timing, or hidden state.

## Example
For `d = 2, M = 4`, the diagonal baseline places towers at
`(0.125,0.125), (0.375,0.375), (0.625,0.625), (0.875,0.875)`; its worst
anchored sector (roughly `[0,0.5) x [0,0.5)`) gives `D* ~ 0.25`. A staggered
placement such as `(0.125,0.625), (0.375,0.125), (0.625,0.875),
(0.875,0.375)` breaks up that diagonal cluster and lowers `D*`, raising the
score above `0.1`. (Illustrative coordinates only — not an optimal answer.)
