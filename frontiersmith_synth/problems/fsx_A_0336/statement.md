# Drone Swarm Flight-Signature Matrix

## Problem

A delivery company runs a swarm of `N` autonomous drones over `N` service zones. For every
(drone, zone) pair the flight controller commits a **flight orientation**: `+1` if the drone
traverses that zone forward (clockwise loop) or `-1` if it traverses it in reverse. The full
schedule is therefore an `N x N` matrix `A` with every entry in `{-1, +1}` — the swarm's
**flight-signature matrix**.

For the swarm to be maximally robust (routes maximally distinguishable, no drone's coverage
pattern reconstructible as a signed combination of the others), the controller wants the
`N` row-signatures to be as *linearly independent as possible*. The scalar that measures this
is the absolute determinant `|det(A)|`: it is `0` when the signatures are linearly dependent
(a fragile schedule) and grows as the rows spread out toward mutual orthogonality.

The **lead drone** (row `0`) has already published its committed schedule as a fixed beacon
pattern `r` (a `±1` vector). Your matrix must reproduce that first row exactly; you design the
remaining `N-1` rows freely.

Maximize `|det(A)|`.

Because `N` is **odd**, no orthogonal (Hadamard) `±1` matrix exists, so the theoretical
per-entry bound is unreachable and the optimum is genuinely open — many different schedules
are viable and there is no closed-form best answer.

## Input (stdin)

```
N
r_0 r_1 ... r_{N-1}
```

- Line 1: the odd integer `N` (`9 <= N <= 27`).
- Line 2: `N` space-separated values, each `-1` or `+1` — the lead-drone beacon pattern `r`.

## Output (stdout)

`N` lines, each with `N` space-separated integers, each `-1` or `+1`: the flight-signature
matrix `A` in row-major order. (Whitespace/newlines are free-form; the checker reads exactly
`N*N` tokens.)

## Feasibility

Output is rejected (score `0`) unless it parses to exactly `N*N` integer tokens, every token
is `-1` or `+1`, and **row `0` equals the beacon pattern `r`**.

## Objective

Maximize the exact integer `D = |det(A)|`, computed by the checker via fraction-free
(Bareiss) Gaussian elimination — no floating point, no tolerance.

## Scoring

Let `F = log2(D)` (with `F = 0` if `D = 0`) and let the internal baseline be
`B = N - 1`, which is `log2` of the determinant of a canonical staircase schedule
(`|det| = 2^{N-1}`). The score is

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

A schedule matching the staircase baseline scores `Ratio = 0.1`. Doubling the number of
independence "bits" toward the (unreachable) Hadamard bound raises the score; the bound is
never attained for odd `N`, so the score stays strictly below saturation and there is always
headroom.

## Constraints

- `9 <= N <= 27`, `N` odd.
- Entries strictly in `{-1, +1}`.
- Deterministic exact-integer scoring.

## Example (worked score)

For `N = 9` a staircase schedule (lower-triangular `+1`, upper `-1`, columns sign-adjusted so
row 0 equals `r`) has `|det| = 2^8 = 256`, so `F = 8 = B` and `Ratio = 0.100`. A circulant
schedule built from `r` reaches `|det| ~ 2^{12.7}`, giving `F/B ~ 1.59` and `Ratio ~ 0.159`.
A locally optimized schedule reaches `|det| ~ 2^{13.3}`, `Ratio ~ 0.166`.
