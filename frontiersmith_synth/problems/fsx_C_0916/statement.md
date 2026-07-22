# Benchmark Stakes on the Mountain Road

## Problem

You are a surveyor laying **benchmark stakes** along a mountain road running
from meter `0` (the trailhead, elevation always known) to meter `L`. A field
crew's altimeter drifts, so whenever a party needs a certified elevation at
some position, a ranger walks there from the **nearest usable stake at or
before that position**, re-calibrating every `W` meters along the way (one
"probe" per `W`-meter increment, rounded **up**). A stake only helps queries
**at or beyond it** — data propagates forward down the road, never backward,
so a stake above a query is useless for that query. The trailhead at `0` is
always available as a fallback.

You are given the season's logged query trace: `Q` distinct positions that
were actually queried, each with a count of how many times. Real traffic
clusters into "hotspots" — overlooks, junctions, a scenic curve — differing in
width, weight, and location across the trace. You have a budget of `K` new
stakes (beyond the free trailhead). Choose their positions to minimize the
total probes the whole season's traffic would need.

## Input

```
L K Q W
```
then `Q` lines, position `i` in strictly increasing order:
```
pos_i cnt_i
```

## Output

Print exactly `K` integers (any whitespace layout) — the stake positions you
choose, each satisfying `0 <= p <= L`. Positions may repeat; a repeat simply
wastes that slot of the budget.

## Feasibility

Exactly `K` integers must be printed, each in `[0, L]`. Any other token count,
any out-of-range value, or trailing tokens after the `K`-th integer scores 0.

## Objective

Let `S` be your printed stakes together with the always-available position
`0`. For query `i` at position `pos_i`, let `s_i` be the **largest** value in
`S` with `s_i <= pos_i` (this always exists because of the trailhead). Your
cost for that query is `cnt_i * ceil((pos_i - s_i) / W)` — one probe charge
per whole `W`-meter increment walked, rounded up. Minimize

```
F = sum over all Q queries of cnt_i * ceil((pos_i - s_i) / W)
```

## Scoring

Let `B` be the probe count of the **uniform-spacing** reference — the
textbook "one stake every `L/(K+1)` meters" layout that ignores the trace
entirely: stakes at `floor(L*i/(K+1))` for `i = 1..K`, plus the free
trailhead. Your score for a test is
```
ratio = min(1000, 100 * B / max(1, F)) / 1000
```
so reproducing blind uniform spacing scores `0.1`, and any real reduction
in probe count below that reference scores higher, capped at `1.0` (ten
times fewer probes than uniform spacing). Per-test ratios are averaged
over 10 tests.

## Constraints

* `1 <= K <= 20`, `1 <= Q <= 700`, `Q` distinct positions `1 <= pos_i <= L`.
* `1 <= L <= 2,000,000`, `1 <= W <= 30`, `1 <= cnt_i <= 20000`.
* `K < Q` is guaranteed (the budget can never cover every logged position).
* Time limit: 4 s. Memory limit: 512 MB.

## Example

`L=20, K=2, Q=5, W=3`, trace:
```
2 3
5 2
9 5
13 1
18 4
```
Uniform-spacing reference: `K+1=3`, stakes at `{6, 13}` (plus trailhead `0`).
`ceil(2/3)*3 + ceil(5/3)*2 + ceil((9-6)/3)*5 + ceil(0/3)*1 + ceil((18-13)/3)*4
= 3 + 4 + 5 + 0 + 8 = 20`, so `B = 20`.

Placing stakes at `{9, 18}` instead: position `2` and `5` still fall back to
the trailhead (`3 + 4 = 7`); position `9` and `18` cost `0` (a stake sits
exactly there); position `13` uses the stake at `9`:
`ceil((13-9)/3) * 1 = 2`. Total `F = 7 + 0 + 2 + 0 = 9`.
Score `= min(1000, 100*20/9)/1000 = 0.2222`.

Placing the same two stakes at the *positions' weighted middle* instead
(e.g. `{9, 13}`, splitting the five positions into two evenly-sized halves)
gives `F = 7 + 0 + 0 + 8 = 15` (position `18` now costs
`ceil((18-13)/3)*4 = 8`) — a natural-looking split that scores only
`min(1000, 100*20/15)/1000 = 0.1333`, because a stake helps only what lies
**at or beyond** it, not what surrounds it.
