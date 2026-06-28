# Continuous Facility Layout

## Research question

A factory planner must place `N` rectangular facility footprints (machines, storage
bays, work cells) on a flat shop floor of size `W × H`. Every footprint `i` has a fixed
width `w_i` and height `h_i`; the planner chooses the integer bottom-left corner
`(x_i, y_i)` of each one. Two requirements pull against each other:

- **No physical clash.** Two facilities that overlap on the floor cannot coexist there,
  so any overlapping area is heavily penalised.
- **Compactness.** Material-handling cost grows with how spread out the facilities are,
  so the layout should keep the footprints clustered tightly around a common centroid
  rather than flung to the far corners.

Formally, choose the `N` corners, keeping every rectangle fully inside the container, to
**minimise**

```
energy = OVERLAP_W   * (total pairwise overlap area)
       + DISPERSION_W * (sum over rects of squared distance of the rect centre
                         from the mean of all centres)
```

with `OVERLAP_W = 1.0` and `DISPERSION_W = 1e-4`. This is a continuous facility-layout
problem: NP-hard, no closed-form optimum, judged by a continuous energy. The only
decision variables are the `N` corner positions; the sizes and the container are fixed.

## Input / output contract

- **Input (stdin), the instance.**
  - Line 1: three integers `N W H` with `120 ≤ N ≤ 200`, and `W, H` the container
    dimensions (a few hundred each). The container is sized so the total rectangle area
    is roughly 35–55 % of `W·H` — there is slack, so an overlap-free layout exists, but a
    careless placement overlaps heavily.
  - Next `N` lines: integers `w_i h_i` (`4 ≤ w_i, h_i ≤ 120`), the size of rectangle `i`
    (1-based, in input order). Every rectangle individually fits: `W ≥ max_i w_i + 10`
    and `H ≥ max_i h_i + 10`.
- **Output (stdout), the solution.**
  - `N` lines; line `i` holds two integers `x_i y_i` — the bottom-left corner of
    rectangle `i`, in the **same order** as the input.
- **Time limit:** 2 seconds wall-clock. **Memory:** 256 MB.

## Background

Two reference approaches frame the problem before committing to one:

- **Blind simulated annealing.** Hold a layout; repeatedly pick a random rectangle, propose
  a uniformly random new legal position, and accept by the Metropolis rule under a cooling
  schedule. This is the textbook layout heuristic, but two things make the naive version
  weak. First, a uniformly random target position is almost never an improving move once
  the layout is half-decent, so the search wastes most of its steps. Second, evaluating a
  move's effect on the overlap term naively re-checks the moved rectangle against all `N−1`
  others — `O(N)` per step — and at `N ≈ 200` with the millions of steps annealing needs,
  that `O(N)` factor is the bottleneck.
- **Force / spring relaxation.** Treat overlaps as repulsive forces and compactness as an
  attractive spring, and integrate the system like a physics simulation. This proposes
  *good* moves but, as a pure gradient descent, settles into the nearest local minimum and
  cannot tunnel out of a tangled configuration.

The open question is how to combine the *directed* moves of force relaxation with the
*escape* ability of annealing, while paying far less than `O(N)` per move.

## Evaluation settings

- **Scoring (what the judge reports; higher is better).** Let the solution be feasible iff
  exactly `N` integer coordinate pairs are present and every rectangle stays fully inside
  the container (`0 ≤ x_i` and `x_i + w_i ≤ W`, and likewise for `y`). Then

  ```
  score = 0                                       if the solution is infeasible
  score = round( SCORE_SCALE / (1 + energy / N) ) otherwise,
  ```

  with `SCORE_SCALE = 1e9`. A lower `energy` yields a higher score; any malformed,
  wrong-length, or out-of-container output **floors the score to exactly 0**. The
  underlying objective is to **minimise `energy`**; the `1e9/(1 + energy/N)` wrapper just
  turns it into a bounded, maximise-style continuous score with a hard feasibility floor.
  The overlap area is the dominant term — driving overlap toward zero is worth far more
  than the small dispersion term — but a layout that achieves zero overlap by scattering
  rectangles to the corners is penalised by dispersion, so both matter.

- **Instances.** A frozen generator draws `N` rectangle sizes from a log-uniform side
  distribution and sizes a near-square container so the packing fraction lands in
  `[0.35, 0.55]`. Everything — `N`, `W`, `H`, every size — is a deterministic function of an
  integer seed. We report the mean score over a fixed seed set (seeds `1..20`). The trivial
  baseline is "place every rectangle's corner at `(0,0)`" (always feasible, maximally
  overlapping).

## Code framework

A single self-contained C++17 program that reads the instance on stdin and writes a
feasible solution on stdout within the time budget.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N; long long W, H;
    if (!(cin >> N >> W >> H)) return 0;
    vector<int> w(N), h(N);
    for (int i = 0; i < N; i++) cin >> w[i] >> h[i];

    // A feasible solution is ANY set of integer corners keeping each rectangle
    // inside the container. (0,0) is always legal because every rectangle fits.
    vector<int> X(N, 0), Y(N, 0);

    // TODO: heuristic. Minimise
    //   OVERLAP_W * (total pairwise overlap area)
    //   + DISPERSION_W * (sum of squared centre-to-mean-centre distances),
    // e.g. simulated annealing with FORCE-DIRECTED move proposals and a
    // SPATIAL HASH grid so each move's overlap delta is O(neighbours), not O(N).

    string out;
    for (int i = 0; i < N; i++) out += to_string(X[i]) + " " + to_string(Y[i]) + "\n";
    cout << out;
    return 0;
}
```
