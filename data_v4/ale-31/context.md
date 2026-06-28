# Balanced Districting

## Research question

A planning region is given as an `H x W` grid of cells, each carrying an integer **population**
`p[r][c] >= 1`. The region must be split into exactly `K` **districts**. Every cell belongs to one
district, and each district has to be a single **4-connected** region (a contiguous blob of cells —
no island parcels). A good districting is one that is **population-balanced** and has **short
internal boundaries** (compact, non-fractal districts). Concretely the task is to minimize

```
cost = imbalance + LAMBDA * boundary
```

where, writing `avg = (total population) / K`,

- `imbalance = sum over districts d of | pop(d) - avg |` — the total L1 deviation of district
  populations from the perfectly-fair share, and
- `boundary = number of unordered 4-adjacent cell pairs whose two cells lie in different districts`
  — the total cut length between districts.

This is the kind of trade-off that appears in fair political redistricting, sales-territory design,
and image super-pixel segmentation: balance the load while keeping each piece geographically
contiguous and tidy. It is a constrained graph-partition problem — **NP-hard**, with no known
efficient exact solver at this scale — and the benchmark scores a partition by *how low* its cost
is, not by matching a unique optimum. `LAMBDA` is fixed at `100`.

## Input / output contract

- **Input (stdin):** the first line is `H W K` (`20 <= H, W <= 40`, `4 <= K <= 10`, and
  `K <= H*W`). Then `H` lines follow, line `r` holding `W` integers `p[r][0] ... p[r][W-1]`
  (`1 <= p[r][c]`), the population grid in row-major order.
- **Output (stdout):** `H*W` integers — the district id `d[r][c]` in `[0, K-1]` for every cell, in
  **row-major** order (any whitespace layout; the scorer reads tokens). A convenient layout is `H`
  lines of `W` ids each.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A partition is **feasible** iff (a) the output parses as exactly `H*W` integers each in `[0, K-1]`;
(b) every district id `0..K-1` is used by **at least one** cell (no empty district); and (c) every
district is a **single 4-connected region** (the cells of any given id form one orthogonally
connected blob). Anything else — a parse error, an id out of range, a missing or empty district, a
district split into two or more pieces — is **infeasible**.

## Background

The connectivity constraint is what makes this hard: without it, balancing `K` sums is an easy
assignment, but "each district is one contiguous blob" couples the cells geometrically and turns the
problem into constrained graph partition. Several approaches sit on the table before committing:

- **Stripe partition (the baseline).** Cut the grid into `K` contiguous horizontal bands of
  near-equal numbers of rows; each band is one district. Every band is a connected rectangle, so it
  is always feasible. But a stripe cuts straight through population hotspots and so is usually badly
  imbalanced — it is the reference the scorer recomputes, and the floor a real solver must beat.
- **Multi-source region growing.** Pick `K` spread-out **seed** cells and grow all districts
  simultaneously from a shared BFS frontier until every cell is claimed. By construction each
  district is one 4-connected region and the whole grid is covered — a feasible, already roughly
  compact and roughly balanced *starting* partition, far better than a stripe.
- **Local search by reassigning border cells.** From a feasible partition, repeatedly move a single
  cell that lies on the border between two districts from its current district (the **donor**) to a
  neighbouring district (the **receiver**). This is the natural neighbourhood: it changes the
  partition by one cell, and because the moved cell is adjacent to the receiver, the receiver can
  never become disconnected.

The decisive lever is making that border-flip local search both **cheap to evaluate** and **safe**:

- **O(1) incremental cost delta.** Moving one cell changes only two district populations, so the
  imbalance delta is read off from `|pop(donor) - avg|` and `|pop(receiver) - avg|` before and after
  — `O(1)`. The boundary delta depends only on the moved cell's `<= 4` neighbours — also `O(1)`. The
  full cost is *never* recomputed inside the search loop, which is what lets a simulated-annealing
  sweep run millions of moves inside the time budget.
- **Donor-only split guard (local bridge test).** The receiver is safe for free; the only way a
  border flip can break feasibility is by splitting the **donor** into two pieces when the cell is
  removed. That is checked locally: gather the donor-side neighbours of the moved cell and run a
  small bounded BFS inside the donor (excluding the cell) to confirm they still reach one another. If
  the bounded search can't confirm it within a capped budget, the move is conservatively **rejected**
  — so connectivity is a hard invariant that never breaks, and any time-limit cutoff still leaves a
  feasible partition in hand.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `H, W in [20, 40]` and `K in [4, 10]`, then builds a smooth low-frequency **population
  density** field — a flat background plus a few 2-D Gaussian "hot spots" — sampled on the grid,
  perturbed by mild per-cell noise and quantised to small positive integers. The non-uniform density
  is exactly what makes balancing non-trivial: a stripe cuts through the bumps and is wildly
  imbalanced, while a good partition has to bend its boundaries around the hotspots, fighting the
  boundary penalty.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted ids.
  - **Feasibility floor:** if the output does not parse as `H*W` ids each in `[0, K-1]`, or some
    district is empty, or some district is not a single 4-connected region, the score is **`0`**.
  - Otherwise compute `cost = imbalance + LAMBDA * boundary` as defined above (lower is better). Let
    `cost_ref` be the cost of the **stripe partition** (`K` horizontal bands), recomputed inside the
    scorer so the reference is reproducible and solver-independent. The score is

    ```
    score = round( 1 000 000 * cost_ref / cost )     (feasible, cost > 0)
    score = 0                                          (infeasible)
    ```

    A higher score is better. The stripe partition scores essentially `1 000 000`; a balanced,
    compact partition that bends around the hotspots scores strictly more; a feasible-but-worse
    partition scores less but stays positive. (The unreachable `cost == 0` perfectly-balanced
    zero-cut case is given a generous full-credit cap.)
- **Reported metric.** The mean score over a fixed seed set. A real multi-source-seed +
  boundary-flip-SA solver lands well above the stripe floor (roughly `2.0x`–`3.0x` of `1 000 000` on
  these clustered instances); the trivial stripe baseline scores exactly `1 000 000` and is the
  floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible
district id per cell (row-major) to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, K;
    if (scanf("%d %d %d", &H, &W, &K) != 3) return 0;
    int N = H * W;
    vector<long long> pop(N);
    for (int i = 0; i < N; i++) scanf("%lld", &pop[i]);

    // A feasible answer is ANY assignment of every cell to a district 0..K-1 such
    // that all K districts are non-empty and each is a single 4-connected region.
    // The safe construction: pick K spread-out seeds and grow all districts at
    // once from a shared BFS frontier -- always connected, always covers the grid.
    // Start there so we always have something legal to print.
    vector<int> assign(N, -1);

    // TODO heuristic: from the multi-source-BFS seeding, run a boundary-flip
    // simulated annealing: repeatedly move a border cell from its donor district
    // to a neighbouring receiver district, scoring the move by an O(1) imbalance
    // delta (two district populations change) plus an O(1) boundary delta (the
    // moved cell's <=4 neighbours), accepting by the SA rule. Guard feasibility
    // with a donor-only local bridge test (the receiver can never split); reject
    // any move that would disconnect the donor. Keep every district non-empty and
    // 4-connected throughout, all under a ~2s wall-clock budget.

    // Emit the district id of every cell in row-major order.
    string out;
    for (int i = 0; i < N; i++) {
        out += to_string(assign[i]);
        out += (((i + 1) % W) == 0) ? '\n' : ' ';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
