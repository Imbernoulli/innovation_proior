# Green-Wave Scheduling at the Bottleneck Lock Chamber

## Problem

A staircase of canal locks funnels barge traffic through one bottleneck **chamber**.
Barges arrive heading either **up** (direction 0) or **down** (direction 1). Each
**cycle** of the chamber takes exactly `t` ticks, moves a **batch** of up to `C` barges
(total length at most `L`) all traveling the *same* direction, and draws water from one
**shared reservoir**: same direction as the previous cycle costs `ws` units, but a
**switch** costs `wa > ws` units (the gates must be fully drained and reflushed). The
reservoir starts at `W0`, regenerates `rho` units per tick the chamber sits **idle**
(never while cycling), and must stay non-negative. Cycles cannot overlap, and none may
finish after the planning horizon `H`.

Each barge has an arrival tick, a direction, a length, a due tick, and an importance
weight. Output a schedule of cycles minimizing total weighted lateness plus the water
actually spent; a barge you never move costs a heavy fixed penalty instead.

## Input (stdin)

```
n C L t H W0 rho ws wa
a_1 d_1 len_1 due_1 wt_1
...
a_n d_n len_n due_n wt_n
```
All values are integers. `d_i in {0,1}`.

## Output (stdout)

```
K
s_1 dir_1 m_1 idx_{1,1} ... idx_{1,m_1}
...
s_K dir_K m_K idx_{K,1} ... idx_{K,m_K}
```
`K` cycles, strictly increasing start ticks, each listing the barges (1-indexed) it
carries.

## Feasibility

- `0 <= K <= 10n+20`; cycles satisfy `s_1 < s_2 < ...`, `s_j >= s_{j-1}+t`, `s_j+t <= H`.
- `0 <= m_j <= C`; batch total length `<= L`.
- every barge index used in **at most one** cycle overall (transits once, or never).
- every barge in cycle `j` has `d_i == dir_j` and `a_i <= s_j`.
- reservoir, simulated in cycle order: `W += rho*(s_j - end_of_previous_cycle)`, then
  `W -= ws` (same direction as the previous cycle, or the first cycle) or `W -= wa`
  (a switch); `W` must never go negative.

Any violation scores `0`.

## Objective

Let `finish_i = s_j + t` for the cycle carrying barge `i` (undefined if never carried).
```
F = sum_{transited i} wt_i * max(0, finish_i - due_i)
  + sum_{untransited i} wt_i * H
  + (total ws/wa actually paid across all cycles)
```
Minimize `F`.

## Scoring

The checker builds its own baseline: repeatedly take the earliest-arrived
still-waiting barge, batch it and its same-direction, earliest-arrived companions up
to capacity, and run that cycle if affordable — a dispatcher that batches but has no
idea about the water budget or due dates. Let `B` be that baseline's `F`. For your
feasible output with cost `F`,
```
Ratio = min(1000, 100 * B / max(1e-9, F)) / 1000
```
A schedule matching the baseline scores about `0.1`; ten times cheaper caps at `1.0`.

## Constraints

`8 <= n <= 75`, `3 <= C`, `t = 10`, small memory footprint, checker runs in `O(n + K)`.

## Example (worked score)

`n=3, C=2, L=10, t=10, H=40, W0=40, rho=1, ws=5, wa=25`:
```
barge 1: a=0  d=0 len=2 due=15 wt=3
barge 2: a=0  d=1 len=2 due=15 wt=2
barge 3: a=10 d=0 len=2 due=25 wt=1
```
One feasible schedule:
```
3
0 1 1 2
10 0 1 1
20 0 1 3
```
Cycle 1 (dir 1, first cycle costs `ws=5`) carries barge 2, finishing at `10` (on time).
Cycle 2 (dir 0, a switch costs `wa=25`) carries barge 1, finishing at `20` (5 late,
`wt_1*5=15`). Cycle 3 (dir 0, same direction, `ws=5`) carries barge 3, finishing at
`30` (5 late, `wt_3*5=5`). Water `5+25+5=35`, lateness `0+15+5=20`: `F=55`.

The checker's baseline serves earliest-arrived-first: barge 1 (`ws=5`, on time), then
switches to barge 2 (`wa=25`, 5 late, `wt_2*5=10`) — with the reservoir now at `0` it
can't afford another cycle, stranding barge 3 (weight 1, penalty `H=40`). Water
`5+25=30`, lateness `0+10`, penalty `40`: `B=80`. `Ratio = min(1000,100*80/55)/1000 =
0.1454...`. Planning the one necessary switch up front instead of chasing whichever
barge looks urgent scores higher — that is the whole game at larger `n`, where dozens
of barges and a tight shared reservoir make the number of *affordable switches* the
real bottleneck, not any single due date.

*(illustrative only — every test instance is generated fresh, no fixed schedule to
pattern-match.)*
