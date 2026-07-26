# Beacon Nights: Booking a Schedule That Splits the Worst Story

## Problem
A telescope has `D` observation **nights**, numbered `1..D`. Each night offers several
bookable **pointings**; you may book **at most one pointing per night**, and **at most
`B` nights in total** -- your schedule is the set of pointings you commit to before any
data comes back. Every pointing has a positive quality **weight** and observes a fixed
list of sky **sectors**.

There are `K` rival **story families**, each describing a different kind of transient
event. Family `f` lists `M_f` candidate **narratives**; narrative `i` of family `f` is
pinned to one sector `sec(i)` and carries a signature **value** `val(i) >= 1`.

If a booked pointing observes sector `sec(i)`, that pointing's **reading** for narrative
`i` equals `val(i)`; otherwise the reading is `0` (the pointing stayed quiet about `i`).
Your booked schedule **distinguishes** two narratives `i, j` of the *same* family iff at
least one booked pointing produces a different reading for `i` than for `j` (either
because it observes one sector but not the other, or because it observes both but their
values differ). Distinguishing a pair means you could, in principle, have ruled one of
the two narratives out -- that is one **hypothesis elimination**.

Some families are easy: their narratives sit on sectors that plenty of pointings cover.
Others are hard: many of their narratives share one sector, and the *only* pointings
that ever observe it are "boring", low-weight ones lost among far more numerous,
higher-weight pointings that observe everything else. A schedule chosen purely by
weight can fill its whole budget without ever touching the sector that would split
that family's cluster.

## Input (stdin)
```
N D K B T
<T lines>: sid night weight numSec sec_1 ... sec_numSec
<K blocks>:
  M_f
  <M_f lines>: sector value
```
`N` = number of sectors, `D` = number of nights, `K` = number of story families, `B` =
maximum nights you may book, `T` = number of pointings. Each of the next `T` lines
describes one pointing: id `sid` (`1..T`), its `night` (`1..D`), its `weight`, and the
`numSec` sectors it observes. Then `K` blocks follow, one per family, each giving
`M_f` narratives as `(sector, value)`.

## Output (stdout)
```
m
s_1 s_2 ... s_m
```
Print `m` (the number of pointings you book, `0 <= m <= B`), then the `m` chosen
pointing ids on the next line (space-separated; print an empty line if `m = 0`).

## Feasibility
An output is valid iff **all** hold: `0 <= m <= B`; every `s_k` is an integer in
`[1, T]`; the `s_k` are pairwise distinct; and no two chosen pointings belong to the
same night. Any violation, or any non-numeric/non-finite token, scores `Ratio: 0.0`.

## Objective
For each family `f`, let `sep(f)` be the fraction of the `C(M_f, 2)` narrative pairs of
`f` that your booked schedule distinguishes. Maximize
```
F = min_f sep(f)
```
the **worst story family's** separation fraction -- your schedule is only as good as the
story it explains *least* well.

## Scoring
The checker builds its own trivial schedule `Fbase`: book nights `1..min(B,D)`, using
each night's first-listed pointing. This is always feasible and always scores `Fbase >
0`. Then:
```
sc = min(1000.0, 100.0 * F / max(1e-9, Fbase))
Ratio = sc / 1000.0
```
Matching the baseline scores `Ratio ~= 0.1`; doing meaningfully better raises it.

## Constraints
- `K = 4`, `6 <= M_f <= 17`, `10 <= D <= 19`, `4 <= B < D`, `T <= ~150`.
- Time limit 5s, memory 512m.

## Example
Suppose a family has narratives `(sec=5,val=1)`, `(sec=5,val=2)`, `(sec=7,val=1)`. A
booked pointing observing `{5}` gives readings `(1, 2, 0)` -- all three pairs
distinguished, `sep = 3/3 = 1.0`. A booked pointing observing `{7}` instead gives
`(0, 0, 1)` -- only the two pairs touching the third narrative are distinguished,
`sep = 2/3`. (Illustrative only -- the real instances have many more narratives and
pointings, and `F` is the minimum `sep(f)` over all `K` families at once.)
