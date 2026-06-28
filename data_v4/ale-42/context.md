# Interactive Adaptive Probing (find hidden hotspots)

## Research question

A hidden `H x W` grid carries a non-negative integer **reward field** `r(x, y) >= 0`. The field is
smooth: it is a sum of a few 2-D Gaussian **hotspots** of uneven height and spread sitting on a small
uniform background. A cell is **hot** iff `r(x, y) >= thr`. You control a **probing agent** with a
budget of `Q` probes. A probe placed at cell `(x, y)` **observes** the `(2*rad+1) x (2*rad+1)`
Chebyshev window around `(x, y)` (clipped to the grid) and returns a **noisy** local reading of each
observed cell — measurement noise with standard deviation `sigma` (relative to the reward scale). The
agent never sees the true field; it sees only noisy readings of the cells it probes near. After
spending its probes, the agent must commit a **report**: a set of cells it declares hot.

The objective is to **maximize the total true reward of the cells you report as hot, minus a penalty
for every reported cell that is in fact not hot**, subject to a hard rule: you may only report a cell
you actually **observed** (probed within `rad` of), and you may not exceed `Q` probes. Because `Q` is
small relative to the grid, the hot mass is concentrated in a few **unknown** locations, and each
reading is noisy, this is a sequential information-gathering problem with no exact answer: it is
judged by a continuous score. The only lever is the heuristic that decides **where to probe** and
**what to report**.

## Input / output contract

This is an interactive-style problem cast in the offline ALE-Bench single-file form: the full hidden
field is present in the instance, but the *contract* (enforced by the scorer) is that you may only
credit cells you observed through `<= Q` probes — so your program must behave like an agent that
discovers the field through probing, not one that reads it off.

- **Input (stdin).** The first line is
  `H W Q rad sigma thr penalty`
  with `40 <= H, W <= 70`, the probe budget `12 <= Q <= 24`, the window radius `rad in {1, 2}`, the
  measurement-noise std `sigma` (a real in `[0.20, 0.45]`), the hot threshold `thr`, and the
  false-positive `penalty` (both integers on the reward scale `SCALE = 1000`). Then follow `H` lines,
  the `x`-th being `W` integers `r[x][0] ... r[x][W-1]` — the hidden reward field
  (`0 <= r <= ~SCALE`). Row index `x in [0, H)`, column index `y in [0, W)`.
- **Output (stdout).** A token stream describing your probes then your report:
  - `P` — the number of probes you used (`0 <= P <= Q`);
  - then `P` pairs `px py` — your probe cells (row then column);
  - then `M` — the number of cells you report as hot;
  - then `M` pairs `rx ry` — your reported cells (row then column).
  Tokens are whitespace-separated; line breaks do not matter. Nothing may follow the last report
  pair.
- **Time limit:** about 2 seconds. **Memory:** 256 MB.

Example shape: with a `50 x 50` grid, `Q = 16`, `rad = 1`, a valid output begins with `16`, then 16
probe-cell pairs, then `M`, then `M` reported-cell pairs. Reporting nothing (`P` probes, then `0`) is
always feasible and scores `0`.

## Background

The structure is a classic **active sensing / Bayesian optimization** problem: a smooth latent field,
expensive noisy point queries, and a budget far smaller than the domain. Two reference points frame
the design.

- **Uniform-grid probing (the baseline).** Lay the `Q` probes on a regular lattice covering the grid
  as evenly as possible, observe their windows, and report exactly the observed cells whose reading
  says "hot". This guarantees coverage with no blind region and is the natural non-adaptive strategy.
  Its weakness is that it is **oblivious**: it spends the same effort on empty background as on a
  hotspot, so it under-samples the hot cores (which is where almost all the reward and almost all the
  decision difficulty lives) and over-samples the void. This is the strategy the scorer normalizes
  against.

- **Adaptive, belief-driven probing.** Because the field is **smooth**, a probe placed *near* a
  hotspot reads elevated values even several cells away. That means a sparse set of readings already
  tells you *roughly where the bumps are* — if you fuse the readings into a spatial belief instead of
  treating each cell in isolation. The established strong approach is to maintain a posterior belief
  over the field (a Gaussian-smoothed estimate that propagates each noisy reading to its
  neighbourhood) and choose probes by an acquisition rule that trades exploration (probe where the
  belief is uncertain) against exploitation (probe where the belief says "hot"). This locates and
  pins the hotspots with far fewer probes than a uniform sweep, and lets you denoise the hot cores by
  concentrating overlapping reads on them. The open design choices are how wide to smooth (too wide
  dilutes a hot core below `thr`; too narrow fails to localize), how to split the budget between a
  coverage sweep and adaptive zoom, and how to make the final hot/cold report decision given that a
  false positive is penalized.

## Evaluation settings

For a fixed seed the generator (`verify/gen.py`) produces one instance. A solver's output is scored
exactly as `verify/score.py` computes:

- **Feasibility / floor (any violation -> score 0).** The token stream must parse exactly: `P >= 0`,
  then `2*P` probe ints, then `M >= 0`, then `2*M` report ints, with nothing left over. Every probe
  and report cell must be inside the grid. `P <= Q` (the probe budget). The `M` reported cells must
  be pairwise **distinct**. Every reported cell must be **observed** — within Chebyshev radius `rad`
  of at least one declared probe (you cannot report a cell you never looked at). Any violation floors
  the score to `0`.

- **Objective (of a feasible report).** With `HOT(c) := (r(c) >= thr)`,
  `value = sum over reported c of r(c)  -  penalty * (number of reported c that are NOT hot)`.
  Reporting only true hot cells collects their full reward with no penalty; reporting an
  observed-but-cold cell costs `penalty`. The credited value is clamped at `0` (a net-negative report
  scores `0`, never below).

- **Normalized score.** The scorer recomputes a deterministic **uniform-grid probing** baseline: lay
  `Q` probes on a regular lattice (`rows x cols <= Q`, aspect-matched to the grid), observe their
  windows, and report every observed cell whose **true** reward is `>= thr` (the best report any
  reporter could make from that fixed coverage). Then
  `score = round(1_000_000 * max(0, value) / max(1, baseline_value))`.
  The baseline scores about `1_000_000`; an adaptive scheme that observes and confirms **more** of
  the hot mass than the fixed lattice scores higher; an infeasible output scores `0`.

**How instances are generated** (`verify/gen.py`, parameter = integer seed). `H, W` are drawn in
`[40, 70]`, the probe budget `Q` in `[12, 24]` (small relative to the grid, so where you probe is the
whole game), `rad in {1, 2}`, and `sigma in [0.20, 0.45]`. The field is a mixture of `3..7` Gaussian
hotspots of uneven height and spread placed anywhere, on a small background, scaled to integers in
`[0, SCALE]` with `SCALE = 1000`. `thr` is `0.55..0.72` of the peak (so only the hottest cores are
hot — a few dozen cells, worth far more than the rest), and `penalty` is `0.30..0.50` of `SCALE`. The
concentrated hot mass in a few unknown regions, plus the heavy measurement noise, is exactly the
regime where an adaptive, belief-driven probe schedule beats an oblivious uniform sweep.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes a feasible
solution to stdout. The scaffold below already emits a valid (empty, score-0) report; the method
replaces the TODO with the belief-grid construction, the adaptive probe schedule, and the
decision-theoretic report.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W, Q, rad, thr, penalty;
    double sigma;
    if (!(cin >> H >> W >> Q >> rad >> sigma >> thr >> penalty)) return 0;
    vector<vector<int>> grid(H, vector<int>(W, 0));
    for (int x = 0; x < H; x++)
        for (int y = 0; y < W; y++)
            cin >> grid[x][y];

    // A feasible fallback: use no probes and report nothing (scores 0 but is
    // always valid -- never crashes, never reports an unobserved cell).
    vector<pair<int,int>> probes;   // <= Q probe cells, each inside the grid
    vector<pair<int,int>> report;   // distinct cells, each within rad of a probe

    // TODO heuristic: maintain a Gaussian-smoothed belief over the field; lay a
    // coarse coverage sweep, then adaptively ZOOM probes onto the detected
    // belief peaks (the hotspots), denoising their cores with overlapping reads;
    // finally report a cell when its expected (penalty-aware) credit is positive.

    // emit: P, P probe cells, M, M report cells
    cout << probes.size() << "\n";
    for (auto &p : probes) cout << p.first << ' ' << p.second << "\n";
    cout << report.size() << "\n";
    for (auto &p : report) cout << p.first << ' ' << p.second << "\n";
    return 0;
}
```
