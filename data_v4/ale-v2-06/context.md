# Dense Weighted Independent Set

## Research question

You are given an undirected graph `G = (V, E)` on `n` vertices, where every vertex `i` carries a
positive integer **weight** `w_i`. A set `S ⊆ V` is an **independent set** if no edge of `G` has both
endpoints in `S` (the vertices of `S` are pairwise non-adjacent). The task is to choose an independent
set `S` that **maximizes the total weight**

```
weight(S) = Σ_{i ∈ S} w_i.
```

This is the **Maximum Weight Independent Set (MWIS)** problem. It is NP-hard and inapproximable in the
worst case; on the *dense* graphs used here (average degree a large fraction of `n`) the maximum
independent set is small, and *which* few vertices to keep is a genuinely hard combinatorial choice.
There is no known efficient exact solution at this scale, so the benchmark scores a submission by *how
much weight* its independent set collects rather than by matching a unique optimum. The empty set is a
legal (weight-0) answer, so a real solver must do strictly better than collecting nothing — and, as the
scoring makes precise, strictly better than a strong greedy.

## Input / output contract

- **Input (stdin):**
  - the first line holds two integers `n m` — the number of vertices and the number of undirected
    edges (`600 ≤ n ≤ 1200`);
  - the second line holds `n` integers `w_0 w_1 … w_{n−1}`, the vertex weights (`1 ≤ w_i ≤ 1000`);
  - then `m` lines follow, each holding two integers `a b` (`0 ≤ a, b < n`, `a ≠ b`) denoting an
    undirected edge `{a, b}`. No edge is repeated; there are no self-loops.
- **Output (stdout):**
  - the first token is `k`, the size of the chosen set (`0 ≤ k ≤ n`);
  - then `k` lines, each one vertex id in `{0, …, n−1}`. Order is irrelevant; the listed ids are the
    chosen independent set. `k = 0` (with no ids) is legal and denotes the empty set, weight `0`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff (1) the output is a single integer `k` followed by exactly `k` integer
ids, all in `{0, …, n−1}` and pairwise **distinct**, with the declared `k` matching the number of ids,
**and** (2) the set is **independent** — no edge of `G` joins two chosen ids. Anything else — a wrong
count, a repeated or out-of-range id, a header disagreeing with the listed ids, a stray token, a
missing file, or an edge inside the chosen set — is **infeasible** and scores `0`.

## Background

Stripped of the surface story, the structure is: pick `S ⊆ V` with no internal edge, maximizing
`Σ_{i∈S} w_i`. Several approaches sit on the table before committing:

- **Greedy by weight (GWMIN).** Repeatedly take the heaviest remaining vertex whose neighbours are all
  still available, then forbid its neighbours; repeat. Always feasible, fast, and a sensible reference.
  But it is myopic: a *heavy hub* — a high-weight vertex with many neighbours — is taken first and
  *blocks an entire independent neighbourhood* whose vertices, though each lighter than the hub, sum to
  far more. Greedy never recovers from that early commitment. (This is the scorer's reference baseline.)
- **Greedy by weight/degree.** Order by `w_i / (deg_i + 1)` to discount hubs. Better on some instances,
  but still a one-shot construction with no way to undo a bad early pick, and it underweights vertices
  that are heavy *and* central but happen to be the right choice.
- **LP relaxation / rounding.** Solve the fractional clique-constrained or edge-constrained relaxation
  and round. On dense graphs the relaxation is loose and the rounding still needs a repair pass to fix
  the conflicts it introduces; it is not the strongest practical lever here.
- **A tightness-based local search (the established strong approach).** Maintain, for every vertex,
  `tight[v]` = the number of its neighbours currently in `S`. A non-solution vertex is **free** iff
  `tight[v] = 0` (it can be added without breaking independence). Inserting or removing a vertex
  changes only its neighbours' tightness, so every move is `O(deg(v))` — never `O(n)`. On top of greedy
  construction one runs two move classes: a **(0,1)-add** (insert any free vertex — always improves),
  and the decisive **(1,2)-swap** — *remove one solution vertex `x` and add two non-adjacent vertices
  that become free* whenever `w_u + w_v > w_x`. The (1,2)-swap is the move a "fix the set, then tweak"
  search cannot reach: it trades a single blocking vertex for a pair, which is exactly how a heavy hub
  gets unwound in favour of the independent neighbourhood it was suppressing. Wrapping local search in
  iterated local search / simulated annealing — a forced-insertion perturbation (push a random vertex
  in, evict its solution-neighbours) followed by re-descent, accepted by a cooling rule — lets the
  search cross the plateaus and escape the deep local optima that dense MWIS is full of.

The decisive accelerators are the **tightness counter** (so add/remove and the freeness test are
`O(deg)`), a **CSR adjacency** (cache-friendly neighbour scans), and the observation that in a dense
graph the (1,2)-swap candidate lists are *short* (a non-solution vertex with exactly one
solution-neighbour is rare), so the swap scan is cheap even though it is the most powerful move.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n ∈ [600, 1200]` and a dense edge probability `p ∈ [0.30, 0.55]`, then builds an
  Erdős–Rényi-style backbone scaled by a per-vertex *popularity* multiplier (so some vertices are hubs,
  some are sparse). It then plants a few **independent pockets** — small vertex subsets made mutually
  non-adjacent and given *moderate* weights — and gives hubs a weight **bonus**. The point of this
  design is to make "take the heaviest first" a genuine trap: a heavy hub blocks a planted independent
  pocket whose aggregate weight beats the hub, so a one-shot greedy leaves weight on the table exactly
  where the (1,2)-swap local search shines.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted solution.
  - **Feasibility floor:** if the output is not a single `k` followed by exactly `k` distinct in-range
    ids (header matching the count), **or the chosen set is not independent**, the score is **`0`**.
  - Otherwise let `W` be the submitted set's total weight (`Σ` of chosen weights; `0` for `k = 0`). Let
    `W_base` be the total weight of the scorer's own deterministic **GWMIN greedy** independent set
    (heaviest-available-first, then forbid its neighbours), recomputed inside the scorer so the
    reference is reproducible and independent of the solver, and let `W_ref = max(W_base, 1)`. The score
    is

    ```
    score = round( 1 000 000 × W / W_ref )     (feasible, independent), clamped to ≥ 0
    score = 0                                    (infeasible / not independent)
    ```

    A higher score is better. The greedy baseline scores exactly `1 000 000`; an independent set heavier
    than greedy scores strictly more; a lighter (but still independent) one scores less but never below
    `0`. The empty set (`W = 0`) scores `0`.
- **Reported metric.** The mean score over a fixed seed set. A genuine tightness-based local search
  should land well above `1 000 000` (≈ `1.3–1.8 ×` on these dense instances); the empty set and any
  non-independent output score `0` and are the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible solution
to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long m;
    if (scanf("%d %lld", &n, &m) != 2) return 0;
    vector<int> w(n);
    for (int i = 0; i < n; i++) scanf("%d", &w[i]);
    // read edges into a CSR adjacency (omitted here)
    for (long long e = 0; e < m; e++) { int a, b; scanf("%d %d", &a, &b); /* ... */ }

    // A feasible answer is ANY set of pairwise NON-ADJACENT distinct ids. The empty
    // set (k = 0, weight 0) is always legal, so hold it as a safety net.
    vector<int> chosen;   // the chosen independent set

    // TODO heuristic: greedy GWMIN construction (heaviest free vertex first); maintain
    // tight[v] = #solution-neighbours so the freeness test and every add/remove are
    // O(deg); run (0,1)-add + (1,2)-swap local search (the swap is the key move); wrap
    // in iterated local search / SA with a forced-insertion perturbation under a ~1.85s
    // budget. Keep a valid independent set at all times and print the best one seen.

    string out;
    out += to_string((int)chosen.size()); out += "\n";
    for (int v : chosen) { out += to_string(v); out += "\n"; }
    fputs(out.c_str(), stdout);
    return 0;
}
```
