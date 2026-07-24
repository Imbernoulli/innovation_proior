# Ember Kilns: Push the Firing Line to Its Ceiling

## Problem
A row of `N` **kilns**, numbered `1..N`, holds `M` indistinguishable **embers**
(several embers may rest in the same kiln). Kiln `i` has a fixed **yield** `w_i >= 0`.
At the end of the firing you **collect** the yield of every kiln that holds at least
one ember; stacked embers beyond the first add nothing.

You may replay the firing line with two kinds of moves, each legal at an
**interior** kiln `i` (`2 <= i <= N-1`) only:

- **scatter** `s i` — requires at least 2 embers in kiln `i`. Remove 2 embers from
  kiln `i`; place 1 ember into kiln `i-1` and 1 into kiln `i+1`.
- **gather** `g i` — requires at least 1 ember in kiln `i-1` and at least 1 in
  kiln `i+1`. Remove those two embers; place 2 embers into kiln `i`.

Both moves keep the ember count `M` fixed, and each one shifts +`i-1` and +`i+1`
against `-2i`, so the total cell-sum of the embers never changes either. Much
less obvious is *which* arrangements these conservation laws actually let you
reach: most pleasing-looking target arrangements lie outside the reachable
class of the initial one, and some heavy-yield kilns are unreachable bait.
Conversely, the best reachable arrangement can hide behind moves that
temporarily *lose* collected yield or cross zero-gain plateaus.

Your job: output a legal move sequence whose **final** arrangement collects as
much yield as possible.

## Input (stdin)
```
N M
w_1 w_2 ... w_N
p_1 p_2 ... p_M
```
`p_j` is the starting kiln of ember `j` (sorted, repetitions allowed).

## Output (stdout)
```
k
<m_1> <i_1>
...
<m_k> <i_k>
```
`k` = number of moves (`0 <= k <= 400`); each following line is `s i` or `g i`
with `2 <= i <= N-1`. Exactly `2k+1` whitespace-separated tokens after `k` are
required (no extra tokens).

## Feasibility
An output scores 0 unless all hold: `k` in range; every move well-formed and in
range; every move **legal at the moment it is played** (the checker replays the
sequence from the initial arrangement); the token count matches exactly.
Non-numeric, missing, or trailing tokens score 0.

## Objective
Maximize `F` = sum of `w_i` over kilns holding at least one ember in the final
arrangement. Intermediate arrangements do not count — only the last one.

## Scoring
The checker's internal baseline `B` is the yield collected by the **do-nothing**
strategy (zero moves), i.e. the yield of the initial arrangement; `B >= 1` always.
```
sc = min(1000.0, 100.0 * F / B)      Ratio = sc / 1000.0
```
Doing nothing scores exactly `0.1`; collecting 10x the initial yield caps at `1.0`.

## Constraints
- `8 <= N <= 13`, `4 <= M <= 6`, `0 <= w_i <= 25`.
- Time limit 5s, memory 512m. The instances are small, but the reachable class
  is not the whole board: brute force over all arrangements is wasted work, and
  local hill-climbing provably stalls below the reachable ceiling on several
  tests.

## Example (illustrative miniature, not a real test)
`N = 5, M = 4`, yields `w = [0, 4, 2, 4, 20]`, embers all in kiln 3:
state `{3,3,3,3}`, initial collection `B = 2`.

- `s 3` -> `{2,3,3,4}` collects `4+2+4 = 10` (gain).
- `s 3` -> `{2,2,4,4}` collects `4+4 = 8` (a loss — a myopic player refuses it).
- `s 2` -> `{1,3,4,4}` (collects 6), then `s 4` -> `{1,3,3,5}` (collects 22),
  then `s 3` -> `{1,2,4,5}`, collecting the `w_5 = 20` treasure: `F = 28`.

Playing only immediately-gaining moves stalls at `F = 10`; routing through the
temporary loss reaches `F = 28` (Ratio capped at `1.0` in this toy, since
`100*28/2 > 1000`). Doing nothing scores `0.1`.

## Notes
The scores are exact integers; no randomness, timing, or tolerances are
involved anywhere. Same submission, same score, forever.
