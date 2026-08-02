# Unexploitable Play Against a Logged Opponent

## Problem
You are the row player in a zero-sum game: you have `m` actions, your opponent has
`n` actions, and an integer payoff matrix `A` gives your payoff `A[i][j]` (positive)
whenever you play row `i` and they play column `j` (their payoff is `-A[i][j]`).

You must publish a **mixed strategy** `p` — a probability distribution over your
`m` rows — before play. Your opponent is fully adaptive: after seeing `p`, they
will play whichever single column hurts you most. So your guaranteed value is

```
worst_case(p) = min_j ( sum_i p_i * A[i][j] )
```

no matter what they do, you get at least this much. The higher `worst_case(p)`,
the less **exploitable** `p` is.

To help you, the input also reports a log of this opponent's play across many
past games: `H_j` = number of times they historically played column `j`, out of
`N = sum_j H_j` total plays. This log is real, but it is **not a promise about
this game** — the opponent you actually face will always play their exact best
response to your published `p`, never their historical tendency. A strategy that
merely punishes the historical log's favorite columns can be catastrophic
against a column the log rarely used.

## Input (stdin)
```
m n
A_0,0 A_0,1 ... A_0,n-1
...
A_m-1,0 ... A_m-1,n-1
N
H_0 H_1 ... H_n-1
```
All `A` entries are positive integers. `H_j` are non-negative integers summing to `N`.

## Output (stdout)
Print `m` non-negative real numbers `p_0 ... p_{m-1}` (space and/or newline
separated), your mixed strategy. Any reasonable formatting/precision is fine.

## Feasibility
Invalid output (score `Ratio: 0.0`) if any of:
- not exactly `m` finite numbers are given (no `nan`/`inf`);
- any number is below `-1e-6`;
- the numbers do not sum to `1` within `1e-4`.

## Objective
Maximize `F(p) = min_j sum_i p_i * A[i][j]`, your worst-case guaranteed payoff
against a fully-informed adversary who always best-responds to your published `p`.

## Scoring
Let `B` be the value of the checker's own trivial reference strategy: the single
**pure** row that would be best if the opponent's column were assumed uniformly
random (i.e. ignoring the log `H` entirely) — `i* = argmax_i mean_j A[i][j]`,
`B = min_j A[i*][j]` (always positive; ties broken by lowest index).
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the baseline scores `Ratio = 0.1`; achieving `10x` the baseline caps at `1.0`.

## Constraints
- `2 <= m = n <= 25`.
- `1 <= A[i][j] <= 999`.
- `0 <= H_j`, and `N = sum_j H_j >= 1`.
- Time limit 5s, memory 512m.

## Example
Suppose `m = n = 2`, `A = [[6, 2], [3, 7]]`, `H = [5, 5]` (so `N = 10`).

Baseline: row averages are `4.0` (row 0) and `5.0` (row 1), so `i* = 1` and
`B = min(A[1][0], A[1][1]) = min(3, 7) = 3`.

If you output `p = [0.5, 0.5]`:
```
col 0: 0.5*6 + 0.5*3 = 4.5
col 1: 0.5*2 + 0.5*7 = 4.5
F = min(4.5, 4.5) = 4.5
sc = min(1000, 100 * 4.5 / 3) = 150
Ratio = 0.15
```
(This 2x2 matrix is illustrative FORM only, not a scaled-down version of the
larger structured instances you will actually be scored on.)
