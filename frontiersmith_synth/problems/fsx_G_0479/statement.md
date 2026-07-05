# Interference-Free Frequency Comb (Sidon Set) Planning

## Problem
You are planning a frequency comb for a wideband transmitter. The usable spectrum is
divided into `n` equally-spaced channels, indexed `1..n`. Some channels are already
occupied by legacy services and are **forbidden**.

Whenever two carriers at channel indices `f_i` and `f_j` are active, the amplifier
produces a **second-order intermodulation product** proportional to `f_i + f_j` (the
self term `2*f_i` is the second harmonic). To keep every intermodulation product on its
own frequency — so no two carrier pairs interfere at the same place — the set of chosen
channels must be a **Sidon set** (a B2 set): **all pairwise sums `f_i + f_j` with
`i <= j` are distinct.**

Select as many **allowed** channels as possible whose comb is interference-free.

## Input (stdin)
```
n k
c_1 c_2 ... c_k
```
- Line 1: `n` (number of channels) and `k` (number of forbidden channels).
- Line 2: the `k` forbidden channel indices (sorted, space separated; the line may be
  empty when `k = 0`).

## Output (stdout)
A single set of chosen channel indices, space or newline separated, e.g.
```
1 2 4 8 15
```

## Feasibility
Your output is valid only if:
- every index is an integer in `[1..n]`;
- indices are distinct;
- no index is forbidden;
- the set is a Sidon set: all pairwise sums `f_i + f_j` (`i <= j`) are distinct.

Any violation scores `0`.

## Objective (maximize)
`F = number of channels selected.`

## Scoring
Deterministic. Let `B` be the power-of-two guard-band comb `{1,2,4,8,...} <= n` (always
feasible and Sidon). The score is
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
So the baseline comb scores `Ratio ~ 0.1`, and a ten-times-larger comb caps at `1.0`.
The maximum Sidon-set size for a general band is not known in closed form, so there is
genuine headroom and multiple viable strategies (greedy Mian-Chowla, algebraic
perfect-difference constructions, local search) — none of which is provably optimal here,
especially once forbidden channels break the algebraic constructions.

## Constraints
- `100 <= n <= 12000`, `0 <= k <= n`.
- Time limit 5s, memory 512m.

## Example (worked score)
Suppose `n = 16`, no forbidden channels. The comb `{1, 2, 4, 8}` (powers of two) is
Sidon, `F = 4`; the baseline `B = 5` (`{1,2,4,8,16}`), giving
`Ratio = 100*4/5 / 1000 = 0.08`. The larger comb `{1, 2, 5, 11, 16}` is also Sidon
(all ten pairwise sums distinct), `F = 5`, giving `Ratio = 100*5/5 / 1000 = 0.10`.
Denser Sidon combs score proportionally higher.
