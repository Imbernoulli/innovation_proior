# Sensor Placement for Coverage + Connectivity

## Research question

You are given an `H x W` grid of cells. Cell `(i, j)` (row `i`, column `j`, both 0-based) has a
non-negative integer **demand** `d[i][j]`. You may deploy **at most `k` sensors** on distinct cells.
A sensor placed on cell `(i, j)` is a disk of **radius `r`** centred at that cell's centre; it
**covers** every cell `(a, b)` whose centre lies within Euclidean distance `r`, i.e.
`(a - i)^2 + (b - j)^2 <= r^2`. The **covered demand** is the sum of `d` over the *union* of all the
sensors' coverage disks (each covered cell counts once).

The sensors also have to talk to each other. Two placed sensors are **linked** when their centres lie
within distance `2r` — i.e. `(i1 - i2)^2 + (j1 - j2)^2 <= (2r)^2`, the case where their coverage disks
touch or overlap. The sensors and their links form a graph; let `C` be its number of connected
components. The objective you **maximize** is

```
objective = covered_demand - lam * max(0, C - 1),
```

where `lam >= 0` is a fixed per-extra-component penalty: one component is "free", and every additional
disconnected cluster costs `lam`. This couples two pulls that fight each other — coverage wants to plant
sensors on the separate demand hotspots, while connectivity wants them in one contiguous blob. It is the
kind of facility-location / wireless-sensor-network design problem that is NP-hard, has no known
closed-form optimum, and is judged by a continuous score.

## Input / output contract

- **Input (stdin).** The first line is `H W k r lam`
  (`26 <= H, W <= 40`, `8 <= k <= 16`, `2 <= r <= 4`, `40 <= lam <= 120`). Then `H` lines follow, the
  `i`-th being the `W` demand values `d[i][0] ... d[i][W-1]` (`0 <= d[i][j] <= 99`), whitespace-separated.
- **Output (stdout).** The first line is `s`, the number of sensors you place (`0 <= s <= k`). Then `s`
  lines follow, the `t`-th being `i j`: the row and column of sensor `t`'s cell
  (`0 <= i < H`, `0 <= j < W`). The positions must be **distinct**. Placing fewer than `k` sensors is
  allowed; `s = 0` is allowed but scores 0 (it covers nothing).
- **Time limit:** 2 seconds. **Memory:** 256 MB.

## Background

Two pieces of structure dominate the design, and two families of approach are on the table before
committing to one.

- The coverage term, taken alone, is a **monotone submodular** set function of the chosen sensor set:
  adding a sensor never decreases coverage, and the marginal coverage of a new sensor only shrinks as
  more sensors are already down (cells it would have covered may already be covered). The textbook
  result is that the **greedy** algorithm — repeatedly add the sensor of largest *marginal* coverage —
  is within a factor `1 - 1/e` of the optimal coverage. The open question is purely how to run that
  greedy fast: recomputing every candidate cell's marginal gain on every round is `O(#cells^2 *
  disk)`, which is wasteful.

- The connectivity term is **not** submodular and is not even monotone in a friendly way — it is a
  graph-component count that a single relocated or inserted sensor can collapse from many components to
  one. A pure coverage greedy ignores it entirely and tends to scatter sensors across distant hotspots,
  ending with `C > 1` and paying `lam * (C - 1)`.

The two approaches:

- **Grid-spaced placement.** Lay sensors on a regular lattice spaced `~2r` apart and take the first
  `k`. This is connected by construction (one component) but blind to where the demand actually is, so
  it leaves a lot of high-demand cells uncovered. (This is the normalization baseline.)
- **Submodular coverage + connectivity repair.** Run a fast lazy greedy for coverage, then *repair*
  connectivity by bridging components together. The non-obvious lever — named in the candidate — is the
  **composition**: use lazy-greedy (CELF) submodular coverage for the placement, then a Steiner-style
  repair that nudges sensors to connect components along a near-straight bridge path, caching marginal
  coverage gains and recomputing them only when invalidated.

## Evaluation settings

A deterministic local scorer (`verify/score.py`) reads the instance and a candidate solution and prints
an integer score; higher is better.

- **Feasibility (any violation floors the score to 0):** the output parses as `s` followed by exactly
  `s` well-formed `i j` lines with no trailing garbage; `0 <= s <= k` (placing **more** than `k`
  sensors floors the score); every `(i, j)` is in range; and all positions are **distinct**.
- **Objective of a feasible solution (higher is better).** `covered_demand` is the demand of the union
  of the coverage disks (each covered cell counted once); `C` is the number of connected components of
  the sensor graph under the `2r` link rule (`C = 0` when `s = 0`); and
  `objective = covered_demand - lam * max(0, C - 1)`.
- **Score (normalized, higher better).** The scorer recomputes the objective of a deterministic
  **grid-spaced placement** baseline (`baseline_objective`) and reports
  `score = round(1_000_000 * solver_objective / max(1, baseline_objective))`. The grid baseline scores
  about `1_000_000`; a better placement scores more. A feasible solution whose objective is `<= 0`
  scores 0 (it did no better than placing nothing). Infeasible output scores 0.
- **Instances.** `verify/gen.py <seed>` builds an `H x W` grid whose demand is a low uniform background
  plus a few 2-D Gaussian **hotspots** placed far enough apart that a pure coverage greedy tends to
  leave its chosen sensors in disconnected clusters. `r` is a small radius (`2..4`), `k` is sized to
  cover a good chunk of — but not all of — the high-demand area, and `lam` is on the scale of a single
  hotspot's coverable demand, so connecting clusters genuinely trades against grabbing one more hotspot.
  That is exactly the regime where the coverage-plus-connectivity-repair heuristic earns its keep.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, k, r, lam;
    if (scanf("%d %d %d %d %d", &H, &W, &k, &r, &lam) != 5) return 0;
    vector<int> d(H * W);
    for (int i = 0; i < H * W; i++) scanf("%d", &d[i]);

    // TODO: choose at most k DISTINCT cells for sensors (radius r) to maximize
    //   covered_demand(union of disks) - lam * max(0, C - 1)
    // where C is the number of connected components of the sensor graph under the
    // 2r link rule. Idea: lazy-greedy (CELF) submodular coverage to place sensors,
    // then a Steiner-style connectivity repair that bridges components together.

    // Print a feasible solution: s, then "i j" per sensor (distinct, in range).
    printf("0\n");
    return 0;
}
```
