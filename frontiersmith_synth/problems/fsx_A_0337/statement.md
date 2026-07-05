# Sum-Frequency Wiring of a Coupler Rail

## Problem

A quantum lab has a linear mounting rail with integer slots `0, 1, ..., M`. You must
mount `n` tunable qubit *couplers* at **distinct** slots
`A = {a_1, a_2, ..., a_n}` (each `0 <= a_i <= M`).

When two couplers at slots `a_i` and `a_j` are driven together they emit a **sum-frequency
(beat) resonance** at frequency `a_i + a_j`. This includes a coupler mixing with itself
(`i = j`), giving `2*a_i`. The set of all emitted resonance frequencies is the **sumset**

```
A + A = { a_i + a_j : 1 <= i <= j <= n }.
```

Two couplers whose sum-frequencies collide cannot be addressed independently — this is
*spectral crowding*. Your goal is to place the couplers so that as many **distinct**
sum-frequencies as possible are produced, i.e. to maximize `|A + A|`.

The rail is deliberately short (`M` grows only linearly with `n`), so a perfectly
collision-free ("Sidon") layout does not fit: every layout is forced into some collisions,
and the densest collision-avoiding packing on a short rail is not known in closed form.

## Input (stdin)

One line with two integers:

```
n M
```

`n` is the number of couplers to place; `M` is the largest usable slot index.

## Output (stdout)

Print exactly `n` distinct integers (whitespace-separated; newlines or spaces both fine),
the coupler slots `a_1 ... a_n`. Order does not matter.

## Feasibility

An output is feasible iff:

- it contains exactly `n` integer tokens (no floats, no `nan`/`inf`, no extra tokens);
- every value is in `[0, M]`;
- all `n` values are pairwise distinct.

Any violation scores `Ratio: 0.0`.

## Objective

Maximize `F = |A + A|`, the number of **distinct** pairwise sums (including doubles `2*a_i`).

## Scoring

The checker computes `F` exactly, and an internal baseline `B = |C + C|` where
`C = {0, 1, ..., n-1}` is the trivial consecutive block (for which `|C + C| = 2n - 1`).
The reported score is

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

So the consecutive block scores `Ratio = 0.1`, and a layout with ten times as many distinct
resonances as the block would cap at `1.0`. Because the rail is short, no layout can reach
that cap — there is genuine, unresolved headroom.

## Constraints

- `12 <= n <= 30`, `M = 9*n` (so sums live in `[0, 2M]`, with `2M + 1` frequency slots).
- Deterministic exact integer scoring; no time or randomness enters the score.

## Example

Suppose `n = 4`, `M = 36`.

- Consecutive block `A = {0,1,2,3}`: `A+A = {0,1,2,3,4,5,6}`, so `F = 7 = 2n-1`, `Ratio = 0.1`.
- Spread layout `A = {0,1,3,7}`: `A+A = {0,1,2,3,4,6,7,8,10,14}`, so `F = 10`,
  `Ratio = min(1000, 100*10/7)/1000 = 0.142857`.

(These are illustrative small numbers; the graded instances use larger `n`.)
