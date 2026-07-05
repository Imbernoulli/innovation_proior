# Forest-Fire Watchtowers: Maximum Interference-Free Channel Completion

## Problem
A national forest is monitored by watchtowers on an `n x n` grid of sites. Every tower must
be tuned to exactly one of `n` radio **channels** `0, 1, ..., n-1`. To keep fire-spotting
transmissions from jamming one another, the fire authority enforces three interference rules:

> No two towers on the **same grid row**, on the **same grid column**, or in the
> **same frequency-reuse sector** may use the same channel.

The `n` **sectors** partition the grid into `n` groups of `n` sites each (a fixed
frequency-reuse plan given to you as part of the instance). A tuning that obeys all three
rules is exactly a **gerechte design** (a Latin square whose sectors are also each a complete
set of channels) -- strictly harder to satisfy than a plain Latin square.

Some towers were surveyed and **pre-tuned** in an earlier season (fixed givens forming a valid
partial gerechte design); the rest are switched off. Your job is to tune as many additional
towers as possible **without ever breaking an interference rule** and **without retuning or
switching off any pre-tuned tower**. You may leave towers off.

Deciding whether a partial gerechte design can be completed is NP-complete, and *maximizing*
the number of tuned towers is harder still -- there is no closed-form optimum. Greedy tunings
dead-end (a later tower ends up with every channel blocked in its row, column, or sector);
smarter ordering and backtracking tune strictly more.

## Input (stdin)
```
n
sector row 0 : n tokens        # n rows, each with n sector ids in [0, n-1]
...
sector row n-1
channel row 0 : n tokens       # n rows: pre-tuned channel in [0,n-1], or '.' (off)
...
channel row n-1
```
The first `n` lines give the sector id of every site; each sector id appears exactly `n` times.
The next `n` lines give the pre-tuned grid: a channel index `0..n-1` (a pre-tuned tower) or
`.` (an off tower). The givens form a valid partial gerechte design (no given conflicts).

## Output (stdout)
Print the completed grid: `n` lines, each with `n` whitespace-separated tokens. Each token is
either a channel index `0..n-1` (a tuned tower) or `.` / `-1` (a tower left off).

## Feasibility
Your tuning is rejected (score `0`) if any of these hold:
- the output does not contain exactly `n*n` tokens;
- any tuned value is not an integer in `[0, n-1]`;
- any pre-tuned given site is altered or switched off;
- any channel repeats within a row, within a column, or within a sector (among tuned towers).

## Objective (maximize)
`F` = the number of tuned (non-off) towers in your feasible tuning (givens included).

## Scoring
Let `B` = the number of pre-tuned givens (the baseline "tune nothing new" plan). The score is
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
so echoing only the givens scores `0.1`, and tuning `10x` the givens caps the ratio at `1.0`.
Scoring is exact integer arithmetic and fully deterministic (no wall-clock, no randomness).

## Constraints
- `n` is **odd**, `9 <= n <= 13` (large scale).
- Reveal density (givens) roughly `0.40 - 0.48` of the sites.
- A full interference-free tuning is guaranteed to exist (the instance is a revealed
  gerechte design), but you are not required to find it -- partial tunings are scored.

## Example
Consider a `3 x 3` toy (below the real size, for illustration). Sectors:
```
0 1 2
2 0 1
1 2 0
```
Pre-tuned channels (`B = 3` givens):
```
0 . .
. . .
. . 2
```
Echoing the input unchanged gives `F = 3`, `Ratio = 0.1`. A full interference-free completion
tunes all `9` towers (`F = 9`), giving `sc = min(1000, 100*9/3) = 300`, `Ratio = 0.3`.

## Notes / strategies
- Plain row-major greedy dead-ends: an early convenient channel can leave a later tower with no
  legal channel at all, so it is left off.
- Ordering the tightest towers first (most-constrained-variable) and backtracking when a choice
  blocks the future recovers many more towers; the added sector rule makes myopic tunings fail
  more often than in a plain Latin square, widening the gap between weak and strong methods.
- There is no wall-time or GPU component in the score.
