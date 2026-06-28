# Facility Location with Opening Cost

## Research question

A logistics company is deciding where to open distribution depots. There are `F`
candidate depot sites; opening site `i` costs a fixed `open[i]` (lease,
build-out, staffing). There are `C` customers. Every customer is served by its
single **nearest open depot** at a service cost equal to the straight-line
(Euclidean) distance between them. You choose a **non-empty** subset `S` of the
candidate sites to open so as to **minimize the total cost**

```
total = sum_{i in S} open[i]  +  sum_{c=1..C} min_{i in S} euclid(client c, facility i).
```

This is the classic **Uncapacitated Facility Location Problem (UFLP)**, also
called the simple/discrete plant location problem: NP-hard, with no closed-form
optimum, judged by a continuous cost. The only lever is *which* sites are
opened; opening fewer sites saves opening cost but pushes customers onto farther
depots (more service cost), and opening more sites does the reverse. The trade-off
between the two cost terms is the whole game.

## Input / output contract

- **Input (stdin), the instance.**
  - Line 1: two integers `F C` with `60 ≤ F ≤ 120` and `400 ≤ C ≤ 700`.
  - Next `F` lines: three integers `fx fy fcost` — facility `i`'s coordinates
    (`0 ≤ fx, fy ≤ 10^6`) and its opening cost `fcost ≥ 0`. Facility `i` is the
    `i`-th of these lines (1-based).
  - Next `C` lines: two integers `cx cy` (`0 ≤ cx, cy ≤ 10^6`), the client
    coordinates.
- **Output (stdout), the solution.**
  - Line 1: an integer `M` — the number of opened facilities. It **must satisfy
    `1 ≤ M ≤ F`** (you must open at least one).
  - Next `M` lines: one integer each, a facility index in `[1, F]`. The `M`
    indices must be **pairwise distinct**. These are the opened sites.
- **Time limit:** 2 seconds wall-clock. **Memory:** 256 MB.

## Background

Three reference approaches frame the problem before committing to one:

- **Trivial extremes.** "Open every facility" minimizes service cost but pays
  every opening cost; "open only the single cheapest facility" minimizes opening
  cost but forces every customer onto one depot (huge service cost). Both are
  feasible and both are weak — the optimum opens an intermediate number of sites.
- **ADD / DROP / SWAP local search (the standard UFLP metaheuristic).** Start
  from some open set and repeatedly apply the best of: *add* a closed facility,
  *drop* an open one, or *swap* (drop one, add another). Each move's gain is the
  change in `open-cost + service-cost`. Written naively, evaluating a *drop*
  re-assigns every affected customer against all remaining open facilities —
  `O(C·|S|)` per candidate drop, `O(C·|S|·F)` per pass — which is too slow to
  sweep all candidates many times within 2 seconds at these sizes.
- **LP relaxation, then round.** UFLP has a strong linear-programming relaxation
  (and an equivalent Lagrangian relaxation of the assignment constraints) whose
  fractional open-indicators round to a much better starting set than either
  trivial extreme. The open question is how to get a principled initial set
  *and* keep each local-search move cheap.

The lever that resolves both halves is caching, for every client, its
**first- and second-nearest open facility**: with the second-nearest distance
`d2[c]` stored, a *drop* of facility `i` is evaluated in `O(C)` (each customer
served by `i` simply falls back to its cached `d2`), and an *add* is `O(C)`
(each customer keeps its better of `d1` and the new distance). That turns the
naive `O(C·|S|)` drop into `O(C)` and makes thousands of moves per second feasible.

## Evaluation settings

- **Scoring (what the judge reports; higher is better).** A solution is feasible
  iff `1 ≤ M ≤ F`, every printed index is an integer in `[1, F]`, and all `M`
  indices are distinct. Then

  ```
  score = 0                                    if the solution is infeasible
  score = round( 10^9 / (1 + total/C) )        otherwise,
  ```

  where `total` is the cost above and `C` the client count. A lower `total`
  yields a higher score; any malformed, empty, `M=0`, out-of-range, or
  duplicate-index output **floors the score to exactly 0**. The underlying
  objective is to **minimize `total`**; the `10^9/(1+total/C)` wrapper just turns
  it into a bounded, maximize-style continuous score with a hard feasibility
  floor. Because `S` must be non-empty, every feasible solution has a finite
  service cost (each client always has a nearest open depot).

- **Instances.** A frozen generator draws both facilities and clients from a
  shared random mixture of 2D-Gaussian "neighbourhood" clusters (clipped to the
  plane), plus ~5% uniform noise. Each facility's opening cost is a base level
  (sized so the optimum opens only a fraction of the sites) times a per-facility
  multiplier, keeping the opening-cost and service-cost terms in the same order
  of magnitude so the trade-off is genuinely live. Everything — `F`, `C`, every
  coordinate and cost — is a deterministic function of an integer seed. We report
  the mean score over a fixed seed set (seeds `1..20`), each rung run on the same
  instances. The trivial baseline is "open all facilities".

## Code framework

A single self-contained C++17 program that reads the instance on stdin and writes
a feasible solution on stdout within the time budget.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int F, C;
    if (!(cin >> F >> C)) return 0;
    vector<double> FX(F), FY(F), FCOST(F);
    for (int i = 0; i < F; i++) cin >> FX[i] >> FY[i] >> FCOST[i];
    vector<double> CX(C), CY(C);
    for (int c = 0; c < C; c++) cin >> CX[c] >> CY[c];

    // A feasible solution is ANY non-empty set of distinct facility indices.
    // Start from a valid set so we can always print something legal: open all.
    vector<char> open(F, 1);

    // TODO: heuristic. Improve `open` to minimize
    //   sum_{i in S} FCOST[i] + sum_c min_{i in S} dist(client c, facility i),
    // e.g. LP/Lagrangian-relaxation rounding for the initial set, then ADD/DROP/
    // SWAP local search with a first/second-nearest cache so each move is O(C).

    vector<int> opened;
    for (int i = 0; i < F; i++) if (open[i]) opened.push_back(i + 1);
    cout << opened.size() << "\n";
    for (int v : opened) cout << v << "\n";
    return 0;
}
```
