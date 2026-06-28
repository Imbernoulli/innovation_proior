# Cable Layout

## Research question

A circuit board exposes `n` **terminals** (pins) at fixed integer positions on a square grid, and
they must all be wired together into one electrically connected net using **axis-aligned
(rectilinear) copper wire** — every wire segment runs purely horizontally or purely vertically, as
on a real chip layer. The routing may introduce extra junction points (Steiner points) wherever
that helps wires share copper. Cost is the **total length of copper laid down**, so the task is to
connect all `n` terminals with the shortest possible rectilinear wire network.

Phrased structurally: choose a set of horizontal/vertical segments whose union is **connected and
spans every terminal** — a *Steiner tree in the L1 (Manhattan) metric*. Among all such layouts,
minimize the total routed wire length (collinear overlaps counted once, because copper laid twice
on the same track is still one wire). This is the **Rectilinear Steiner Minimal Tree (RSMT)**
problem: it is NP-hard, there is no known efficient exact solver at this scale, and the benchmark
scores a layout by *how short* its copper is rather than by matching a unique optimum.

## Input / output contract

- **Input (stdin):** the first line is `n SIDE`, where `n` is the number of terminals
  (`200 ≤ n ≤ 600`) and `SIDE` is the grid extent (so all coordinates lie in `[0, SIDE]`). Then `n`
  lines follow, line `i` (0-indexed) holding two integers `x_i y_i` with `0 ≤ x_i, y_i ≤ SIDE`. All
  terminal coordinates are **distinct**.
- **Output (stdout):** the first token is `m`, the number of wire segments. Then `m` lines follow,
  each holding four integers `x1 y1 x2 y2` describing one **axis-aligned** segment from `(x1,y1)` to
  `(x2,y2)`. A segment is legal iff `x1 == x2` (vertical) or `y1 == y2` (horizontal), it is
  non-degenerate (the two endpoints differ), and all four coordinates are in `[0, SIDE]`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A layout is **feasible** iff (a) the output parses as exactly `m` legal axis-aligned segments and
(b) the union of those segments forms a single connected network containing **all `n` terminals**
(two segments are electrically joined wherever they share any point — endpoint touch, T-junction,
or a proper interior crossing of a horizontal and a vertical wire). Anything else — a non-axis
segment, an out-of-grid coordinate, a parse error, a missing file, or terminals split across
components — is **infeasible**.

## Background

The wire network must be connected and span all terminals, with only horizontal/vertical copper and
free Steiner junctions allowed: that is precisely the Rectilinear Steiner Minimal Tree. Several
approaches sit on the table before committing:

- **Rectilinear minimum spanning tree (MST), L-routed.** Build the MST of the terminals under the
  Manhattan metric, then route each tree edge as an **L-shape** (one horizontal leg + one vertical
  leg). This is always feasible and is the classic *baseline*: the routed length equals the MST
  weight, and it is provably within a factor `3/2` of the RSMT optimum (Hwang's bound). It is the
  reference the scorer recomputes.
- **Hanan-grid Steiner construction.** **Hanan's theorem** says an optimal RSMT uses Steiner points
  only at intersections of the vertical and horizontal lines through the terminals — the *Hanan
  grid*. This collapses an infinite continuous search to a finite `O(n²)`-node grid and is the
  structural lever every strong RSMT heuristic exploits.
- **Overlap-sharing by L-shape choice.** Each MST edge has **two** L-routes of equal length (corner
  above-left vs. below-right), but they overlap differently with the rest of the tree. Because the
  cost is the *union* of copper, picking L-corners so that neighbouring routes run along the **same
  Hanan gridline** turns two separate wires into one shared trunk — free length. Choosing the
  corners well is the single biggest lever that pushes a layout below the MST length.
- **Borah–Owens–Irwin edge-based Steinerization.** After L-corners settle, reconnect a terminal to
  a nearby perpendicular trunk (creating a Hanan Steiner junction) when that shortens the net — the
  standard way an MST is refined into a genuine Steiner tree.

The decisive engineering lever is **incremental union evaluation on the Hanan grid**: flipping one
edge's L-corner only changes copper on that edge's own two gridlines, so its effect on the total
routed length is an `O(degree)` recompute over exactly those two lines — the full union is *never*
recomputed. That makes a simulated-annealing sweep over the per-edge corner choices cheap enough to
run tens of thousands of moves inside the time budget, while the tree topology (and hence
feasibility) never changes.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n ∈ [200, 600]` and places the terminals as a mixture of a few 2-D Gaussian "pin
  clusters" (the dense functional blocks of a board) plus a uniform background, clipped to the
  `[0, SIDE]²` grid with all coordinates distinct (`SIDE = 10 000`). Clustered layouts are exactly
  where Steiner sharing saves the most over a plain L-routed MST.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted segments.
  - **Feasibility floor:** if the output does not parse as `m` legal axis-aligned in-grid segments,
    or the wires do not connect all `n` terminals into a single component, the score is **`0`**.
  - Otherwise let `W` be the **total routed wire length** — the length of the geometric *union* of
    all segments (collinear overlaps counted once, so padding a wire with duplicate/overlapping
    copper cannot be gamed). Let `G` be the **rectilinear MST weight** over the terminals under the
    L1 metric (recomputed inside the scorer via Prim, so the reference is reproducible and
    independent of the solver). The score is

    ```
    score = round( 1 000 000 × G / W )      (feasible, W > 0)
    score = 0                                (infeasible)
    ```

    A higher score is better. The L-routed rectilinear MST scores essentially `1 000 000`; a genuine
    Steiner layout that shares trunk copper scores strictly more; a wasteful layout scores less but
    stays positive. (`n ≤ 1` is a degenerate full-credit case — no wire needed.)
- **Reported metric.** The mean score over a fixed seed set. A real Hanan-grid overlap-sharing
  solver should land a few percent above `1 000 000` (≈ 1.08–1.11× on these clustered instances);
  the trivial *star* layout — every terminal L-wired straight to terminal 0 — scores only ~100 000
  and is the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible set of
axis-aligned segments to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, side;
    if (scanf("%d %d", &n, &side) != 2) return 0;
    vector<long long> X(n), Y(n);
    for (int i = 0; i < n; i++) scanf("%lld %lld", &X[i], &Y[i]);

    // A feasible answer is ANY connected rectilinear network spanning all
    // terminals. The safe baseline: a rectilinear MST with every edge routed as
    // an L-shape -- always connected, always valid. Start there so we always
    // have something legal to print.

    // TODO heuristic: build the rectilinear (L1) MST; restrict Steiner points to
    // the Hanan grid; choose each edge's L-corner (above-left vs below-right) so
    // neighbouring routes share trunk copper, scored by an O(degree) incremental
    // union delta and driven by simulated annealing; optionally Steinerize edges
    // by reconnecting terminals to nearby trunks. Keep the network connected and
    // spanning throughout, all under a ~2s wall-clock budget.

    // Emit the deduplicated horizontal/vertical copper as `m` segments.
    vector<array<long long,4>> segs; // each {x1,y1,x2,y2}, axis-aligned
    // ... fill segs with a feasible layout ...

    string out = to_string((long long)segs.size()) + "\n";
    for (auto &s : segs)
        out += to_string(s[0]) + " " + to_string(s[1]) + " " +
               to_string(s[2]) + " " + to_string(s[3]) + "\n";
    fputs(out.c_str(), stdout);
    return 0;
}
```
