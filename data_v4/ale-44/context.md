# Conveyor / Belt Layout (route items by tile rotation)

## Research question

You are given an `H x W` grid. Some cells are **sources**, each emitting one item in a fixed
direction; some cells are **sinks**. Every other cell is empty and may hold at most one **belt
tile**, which is a directional conveyor pointing in one of the four cardinal directions. An item sits
on its source at tick 0 and advances **one cell per tick**, always leaving its current cell in that
cell's direction (the source's emission direction on a source cell, the tile's direction on a belt
tile). An item is **delivered** the moment it steps onto a sink; it is **lost** if it leaves the grid
or enters an empty cell with no tile. You may place at most `B` belt tiles. Maximize the number of
delivered items.

This is the heuristic-optimization core of factory/automation games (Factorio-style belt routing):
an NP-hard congested-routing problem on a grid where the only lever is which cells carry a belt and
which way each belt points. There is no known optimum; the score is a continuous count judged on
local seeds.

## Input / output contract

- **Input (stdin)** — one instance:
  - line 1: `H W` (grid dimensions; rows `r in [0,H)`, cols `c in [0,W)`).
  - line 2: `nS nG B T` — number of sources, number of sinks, the tile budget, and the simulation
    horizon (number of ticks).
  - next `nS` lines: `r c d` — a source at `(r,c)` emitting in direction `d`
    (`0=Right(+c)`, `1=Down(+r)`, `2=Left(-c)`, `3=Up(-r)`).
  - next `nG` lines: `r c` — a sink at `(r,c)`.
  - All source and sink cells are pairwise distinct.
- **Output (stdout)** — one solution:
  - line 1: `K` — the number of belt tiles you place (`0 <= K <= B`).
  - next `K` lines: `r c d` — a belt tile at `(r,c)` pointing in direction `d`.
- **Time limit:** ~2 seconds per instance. **Memory:** 256 MB.

## Background

The per-cell out-direction map turns the grid into a **functional graph**: every used cell (source or
belt tile) has exactly one out-edge, so an item's whole future is determined by where it starts. An
item is delivered iff the forward walk from its source reaches a sink within `T` ticks; otherwise it
falls off the grid, runs into an unbuilt cell, or cycles forever. Because items do not collide
(belts carry abstract flow), each source's fate is independent given the layout — which is exactly
what makes incremental, component-local re-evaluation possible.

Two reference approaches frame the problem. The **straight-line** baseline lays, for each source, a
straight run of belts in its emission direction until it hits a sink or runs out of room — cheap but
almost never aligned with a sink, so it delivers very little. The **routing** view treats each
delivery as a shortest source-to-sink path; the difficulty is the shared **budget** `B`: routing
every source on its own private path costs far more tiles than `B`, so paths must **merge onto shared
trunks** to fit, and choosing which sources to route and how to share is the combinatorial heart.

## Evaluation settings

- **Score (per instance):** the number of delivered items under the submitted layout, computed by
  the frozen scorer `verify/score.py`, which: (1) parses the layout; (2) checks feasibility; (3)
  builds the per-cell direction map; (4) simulates each source's forward walk for up to `T` ticks and
  counts how many reach a sink.
- **Feasibility -> 0 floor.** The score is forced to `0` if the output is malformed (non-integer,
  truncated, or carrying extra tokens after the declared `K` tile triples), if `K < 0` or `K > B`
  (budget exceeded), if any tile is out of the grid, if any tile is placed on a source or sink cell,
  if two tiles share a cell, or if any direction code is outside `{0,1,2,3}`.
- **Instances** are produced by `verify/gen.py SEED`: a grid of `20..30` per side, `16..26` sources
  (each at a random distinct cell with a random emission direction), `2..4` sinks, a **tight** budget
  `B` of `12%..18%` of the cells (so private per-source paths cannot all fit — trunk sharing is
  forced), and a generous horizon `T = 2*H*W`.
- **Normalization / reporting:** the raw delivered count is reported and compared against the
  straight-line baseline (`verify/baseline.py`) on the same seeds; a higher mean is better. The seed
  set used for self-verification is seeds `1..20`.

## Code framework

A single self-contained C++17 program that reads stdin and writes a feasible layout to stdout. The
scaffold below reads the instance and emits an empty (always-feasible) layout; the heuristic goes
where marked.

```cpp
#include <bits/stdc++.h>
using namespace std;

static const int DR[4] = {0, 1, 0, -1};   // 0=R,1=D,2=L,3=U
static const int DC[4] = {1, 0, -1, 0};

int main() {
    int H, W, nS, nG, B, T;
    if (scanf("%d %d", &H, &W) != 2) return 0;
    scanf("%d %d %d %d", &nS, &nG, &B, &T);

    vector<int> srcR(nS), srcC(nS), srcD(nS);
    for (int i = 0; i < nS; i++) scanf("%d %d %d", &srcR[i], &srcC[i], &srcD[i]);
    vector<int> sinkR(nG), sinkC(nG);
    for (int i = 0; i < nG; i++) scanf("%d %d", &sinkR[i], &sinkC[i]);

    // role[r*W+c]: -1 empty, -2 sink, -3 source. tileDir[cell] in {0..3} or -1.
    // TODO: heuristic -- place at most B belt tiles so that as many sources as
    // possible reach a sink by following cell directions for up to T ticks.
    // Output a FEASIBLE layout: K <= B, no tile on a source/sink, no overlap.

    vector<array<int,3>> tiles;   // (r, c, dir)
    printf("%d\n", (int)tiles.size());
    for (auto &t : tiles) printf("%d %d %d\n", t[0], t[1], t[2]);
    return 0;
}
```
