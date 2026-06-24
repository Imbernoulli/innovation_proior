# Total pumping work in a rooted irrigation tree

## Research question

A farm's irrigation network is a rooted tree of `n` junctions. Water is pumped in at the root and
flows down toward the field outlets. Each junction `i` (other than the root) is connected to its
parent by exactly one pipe segment; pushing one unit of water through that segment costs `cost[i]`
units of energy. Every **outlet** — a junction with no children — must receive exactly one unit of
water. To deliver one unit to an outlet, that unit must travel down every pipe on the root-to-outlet
path, paying each segment's per-unit cost once.

Output the **total pumping work**: the sum, over all outlets, of the energy spent moving that
outlet's one unit of water from the root down to it. Equivalently (this equivalence is the crux of
the efficient solution), it is the sum over every pipe of `cost[pipe] * (number of outlets fed
through that pipe)`, because a pipe carries one unit for each outlet in the subtree hanging below it.

This is a textbook DFS subtree-aggregation, but the headline trap is scale: the per-pipe terms and
their running total grow far past what a 32-bit integer can hold, so the data types must be chosen
deliberately.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 2*10^5`), the number of junctions, numbered
  `1..n`. Then follow `n` lines; line `i` (for `i = 1..n`) contains two integers `p_i` and `c_i`:
  - `p_i` is the parent of junction `i`, or `-1` if junction `i` is the root.
  - `c_i` (`0 <= c_i <= 10^6`) is the per-unit cost of the pipe from `i` up to its parent. For the
    root, `c_i` is present in the input but irrelevant (the root has no pipe above it).
  Exactly one junction has `p_i = -1`. The parent relation forms a single rooted tree; the root id
  is not necessarily `1`, and a parent may appear later in the list than its child.
- Output (stdout): a single line with the total pumping work.
- Time limit: 1 second. Memory: 256 MB.

Example: the tree with root `1`, children `2` (cost `5`) and `3` (cost `3`), where `2` has children
`4` (cost `7`) and `5` (cost `2`), and `3` has child `6` (cost `4`), has outlets `4, 5, 6` with
root-path costs `5+7=12`, `5+2=7`, `3+4=7`, for a total of `26`.

## Background

Once the tree is built, every plausible method is a depth-first traversal that aggregates something
over subtrees; the only real questions are *what* to aggregate and *how big it gets*.

- **Per-outlet path summation.** For each outlet, walk up to the root summing pipe costs. This is the
  literal definition and is obviously correct, but a single root-to-outlet walk can be `O(n)` long
  and there can be `O(n)` outlets, so a degenerate tree (a long chain with a bush of leaves) makes it
  `O(n^2)` — too slow at `n = 2*10^5`. It is the right *reference* implementation, not the right
  *submission*.
- **Subtree-outlet-count DFS.** Compute, for each junction `v`, the number of outlets in its subtree
  `leaves[v]`; then each pipe `(v -> parent)` is traversed once per outlet below `v`, contributing
  `cost[v] * leaves[v]`. Summing those products over all non-root junctions is the answer in one
  linear pass. The open questions are the exact aggregation order (a correct post-order) and the
  integer width, since both the products and the running sum overflow 32 bits.

## Evaluation settings

Judged on hidden tests covering: small hand-checkable trees; a single junction (`n = 1`, no pipes,
answer `0`); deep chains of length `2*10^5` (which break recursive DFS via stack overflow and break
`O(n^2)` path-walking via time limit); wide "broom" trees (a short stem feeding `~2*10^5` outlets,
which maximize both the per-pipe outlet counts and the total, so the answer comfortably exceeds
`2^31`); zero-cost pipes; roots placed at a non-first junction id.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> par(n + 1, 0);
    vector<long long> cost(n + 1, 0);
    int root = -1;
    vector<vector<int>> children(n + 1);
    for (int i = 1; i <= n; i++) {
        int p; long long c;
        cin >> p >> c;
        par[i] = p;
        cost[i] = c;
        if (p == -1 || p == 0) root = i;
        else children[p].push_back(i);
    }

    // TODO: DFS-aggregate outlet counts per subtree, then sum cost[v] * leaves[v]
    //       over non-root v. Mind the integer width and the recursion depth.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
