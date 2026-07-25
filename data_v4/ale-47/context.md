# Knapsack with Synergies

## Research question

A planner must choose a subset of `n` **items** to pack under a single **weight budget** `W`. Each
item `i` has a positive weight `w_i` and a non-negative linear value `v_i`, exactly as in an ordinary
0/1 knapsack. What lifts this out of the ordinary is a second, **pairwise** reward: for certain
unordered pairs `{i, j}` there is a **synergy bonus** `b_{ij} > 0` that is collected **only when both
`i` and `j` are selected**. The total reward of a chosen subset `S` is therefore

```
obj(S) = Σ_{i∈S} v_i  +  Σ_{ {i,j} : i∈S and j∈S } b_{ij}.
```

The goal is to choose `S` with total weight `Σ_{i∈S} w_i ≤ W` that **maximizes `obj(S)`**. The
pairwise term makes the objective **non-separable** — the marginal worth of an item depends on which
other items are already in the bag — so this is the **Quadratic Knapsack Problem (QKP)**. QKP is
strongly NP-hard, has no efficient exact solver at the scales here, and the benchmark scores a subset
by *how large its objective is* rather than by matching a unique optimum: a continuous-score heuristic
optimization problem.

## Input / output contract

- **Input (stdin):**
  - Line 1: `n W` — the number of items (`400 ≤ n ≤ 900`) and the integer weight budget `W`.
  - Next `n` lines: line `i` (0-indexed) holds `w_i v_i`, the weight (`1 ≤ w_i`) and linear value
    (`0 ≤ v_i`) of item `i`.
  - Next line: `p` — the number of synergy pairs.
  - Next `p` lines: each holds `i j b`, an **unordered** pair `{i, j}` with `0 ≤ i < j < n` and a
    positive integer bonus `b`, meaning `b_{ij} = b` is earned iff both `i` and `j` are selected. Each
    unordered pair appears at most once.
- **Output (stdout):** the first token is `k`, the size of the chosen subset. Then `k` integers
  follow (whitespace-separated, any layout): the **distinct** indices of the selected items, each in
  `[0, n)`. The empty subset is printed as `k = 0`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A subset is **feasible** iff (a) the output parses as `k` followed by exactly `k` integers, (b) every
index lies in `[0, n)` and the `k` indices are **distinct**, and (c) the total selected weight is
`≤ W`. The empty subset is feasible (objective `0`). Anything else — a parse error, a repeated index,
an out-of-range index, trailing junk, or total weight exceeding `W` — is **infeasible**.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n ∈ [400, 900]`, draws item weights in `[1, 100]` and **modest** linear values in `[0, 40]`
  (so the quadratic term dominates the decision), and sets `W` to a tight fraction (`≈ 0.18–0.32`) of
  the total weight. Crucially the synergy graph has **community structure**: items are partitioned
  into a few latent clusters, and synergy pairs are drawn far more often *within* a cluster (with
  large bonuses) than *across* clusters (with small bonuses).
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted subset `S`.
  - **Feasibility floor:** if the output does not parse as `k` indices, repeats an index, names an
    out-of-range index, has trailing tokens, or the total selected weight exceeds `W`, the score is
    **`0`**.
  - Otherwise compute `obj(S) = Σ_{i∈S} v_i + Σ_{ {i,j}⊆S } b_{ij}`. Let `G` be the objective of the
    **synergy-blind value/weight-ratio greedy**, recomputed inside the scorer (sort by `v_i/w_i`
    descending with deterministic tie-breaks, add each item that fits, then add whatever synergy
    bonuses happen to be realized among the chosen items). The score is

    ```
    score = round( 1 000 000 × obj(S) / G )      (feasible, G > 0)
    score = round( 1 000 000 × obj(S) )          (feasible, G = 0, degenerate)
    score = 0                                     (infeasible)
    ```

    A higher score is better. The ratio-greedy reference scores exactly `1 000 000`; a synergy-aware
    subset that harvests more bonus scores strictly more; a weaker feasible subset scores less but
    stays positive.
- **Reported metric.** The mean score over a fixed seed set. The empty subset scores `0` and is the
  floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible subset to
stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long W;
    if (scanf("%d %lld", &n, &W) != 2) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) scanf("%lld %lld", &w[i], &v[i]);
    long long p; scanf("%lld", &p);
    vector<vector<pair<int,long long>>> adj(n);          // incident-synergy lists
    for (long long e = 0; e < p; e++) {
        int a, b; long long bonus; scanf("%d %d %lld", &a, &b, &bonus);
        adj[a].push_back({b, bonus});
        adj[b].push_back({a, bonus});
    }

    // A feasible answer is ANY subset whose total weight is <= W; the empty set
    // (k = 0) is always feasible. Start there so we always have something legal
    // to print, then improve.

    // TODO heuristic: build a feasible subset (total weight <= W), then improve
    // it to push obj(S) as high as possible within the ~2s budget, always
    // retaining the best feasible subset found so far.

    vector<int> chosen;          // indices of the selected items
    // ... fill `chosen` with a feasible subset (total weight <= W) ...

    string out = to_string((long long)chosen.size()) + "\n";
    for (size_t k = 0; k < chosen.size(); k++)
        out += to_string(chosen[k]) + (k + 1 == chosen.size() ? "\n" : " ");
    if (chosen.empty()) out += "\n";
    fputs(out.c_str(), stdout);
    return 0;
}
```
