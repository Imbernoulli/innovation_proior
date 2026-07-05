# Drone Swarm Rendezvous: Maximum Sumset Diversity

## Problem

A delivery company operates a swarm of `k` autonomous drones. Every drone is given a
single integer **launch offset** — a slot in a shared, discrete flight schedule of
horizon `M` (offsets are integers in `[0, M]`). Two drones with offsets `x` and `y`
open a **rendezvous window** at the combined slot `x + y` (a drone may also rendezvous
with itself, opening the window `2x`, used for self-diagnostics). The mission controller
wants the swarm to touch **as many distinct rendezvous slots as possible**, so that
package hand-offs are spread across the schedule instead of piling onto a few congested
moments.

Formally, choose a set `A` of distinct integer offsets. The set of rendezvous slots is
the **sumset**

```
A + A = { x + y : x in A, y in A }.
```

Maximize `|A + A|`, the number of distinct rendezvous slots.

This is a finite additive-combinatorics extremal problem: for a horizon `M` too short to
admit a perfect Sidon (B_2) set of size `k`, the maximum achievable `|A+A|` has **no
known closed-form optimum**, so many construction strategies compete.

## Input (stdin)

One line with two integers:

```
k M
```

- `k` — the number of drones (you assign `k` offsets, or fewer).
- `M` — the schedule horizon; every offset must lie in `[0, M]`. (Here `M = 9k`.)

## Output (stdout)

First a count `c`, then the `c` chosen offsets:

```
c
a_1 a_2 ... a_c
```

(Whitespace/newlines between tokens are free-form; the checker reads `c` then exactly `c`
offset tokens.)

## Feasibility

- `1 <= c <= k`.
- Each offset is an integer with `0 <= a_i <= M`.
- All `c` offsets are **distinct**.
- Exactly `c` offset tokens are present after the count (no extra, no missing).

Any violation (bad count, out-of-range, duplicate, non-integer / `nan` / `inf`, wrong
token count) scores `0`.

## Objective

Maximize the sumset cardinality `F = |A + A|` (computed exactly by the checker).

## Scoring

The checker builds an internal baseline `B = |AP + AP| = 2k - 1`, where `AP =
{0,1,...,k-1}` is the arithmetic-progression assignment. Your score is

```
sc = min(1000, 100 * F / B)
Ratio = sc / 1000
```

so the plain arithmetic progression scores `Ratio = 0.1`, and you would need to reach
`10x` the baseline sumset to cap out. Reaching the full interval is impossible (the
interval bound `|A+A| <= 2M+1` and the Sidon bound `|A+A| <= k(k+1)/2` both cut in),
so genuine headroom remains above every simple heuristic.

## Constraints

- `8 <= k <= 56`, `M = 9k`.
- Scoring is exact integer arithmetic and fully deterministic.

## Example (worked score)

Suppose `k = 4`, `M = 36`.

- Baseline `AP = {0,1,2,3}`: `A+A = {0,1,2,3,4,5,6}`, so `B = 7`, `Ratio = 0.1`.
- A spread choice `A = {0, 1, 3, 7}`: pairwise sums are
  `{0,1,2,3,4,6,7,8,10,14}` = 10 distinct slots, so `F = 10` and
  `Ratio = min(1000, 100*10/7)/1000 = 0.142857`.

A Sidon-flavoured packing that avoids coincident sums earns a substantially higher score.
