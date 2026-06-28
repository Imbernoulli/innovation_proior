# Cut-to-Quarantine: minimum edge cut to isolate marked nodes from the root

## Research question

You manage a network shaped as a tree of `n` facilities, rooted at facility `1`. Each of the
`n - 1` tree edges has a positive **cut cost**: the price of physically severing that link.

Then `q` independent **quarantine orders** arrive. Each order names a set `S` of *infected*
facilities. For that order you must cut a set of edges so that **no facility in `S` remains
connected to the root** (facility `1`), and you want to do it as cheaply as possible. Report,
for each order, the minimum total cut cost.

The orders are independent: cuts made for one order do not carry over to the next.

The hard part is the *aggregate* scale. A single order may name many facilities, and the only
bound you are given is on the **total** size of all the sets, `sum |S| <= 2*10^5`. A solution that
spends time proportional to the whole tree on *every* order (for instance, a fresh tree DP over all
`n` nodes per order) costs `O(q * n)`, which is far too slow when both `q` and `n` are large. The
research question is how to make each order cost time proportional to `|S|` (up to a logarithmic
factor), not to `n`.

## Input / output contract

- Input (stdin):
  - The first line contains `n` (`1 <= n <= 2*10^5`).
  - Each of the next `n - 1` lines contains three integers `u v w`
    (`1 <= u, v <= n`, `u != v`, `1 <= w <= 10^9`): an undirected tree edge between `u` and `v`
    with cut cost `w`. The edges always form a tree.
  - The next line contains `q` (`1 <= q <= 2*10^5`).
  - Each of the next `q` lines describes one order: first an integer `k` (`0 <= k`), the size of
    `S`, then `k` distinct facility ids, each in `2..n` (the root, facility `1`, is never marked —
    a node cannot be quarantined from itself). The sum of all `k` over the whole input satisfies
    `sum k <= 2*10^5`.
- Output (stdout): `q` lines; line `i` is the minimum total cut cost for order `i`. If `k = 0`,
  the answer is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Because a single answer can be the sum of up to `2*10^5` edges each of cost up to `10^9`, the
answer can reach `~2*10^14` and must be printed as a 64-bit integer.

### Worked sample

Input:

```
7
1 2 2
1 3 4
3 4 1
1 5 3
5 6 2
5 7 5
3
1 4
2 4 6
2 6 7
```

Output:

```
1
3
3
```

The tree rooted at `1` has children `2` (edge cost 2), `3` (cost 4), `5` (cost 3); node `3` has
child `4` (cost 1); node `5` has children `6` (cost 2) and `7` (cost 5).

- Order `{4}`: the path root → 4 is `1-3` (cost 4) then `3-4` (cost 1); cutting the cheaper link
  `3-4` isolates `4`. Answer `1`.
- Order `{4, 6}`: isolate `4` via `3-4` (cost 1) and `6` via `5-6` (cost 2). Answer `1 + 2 = 3`.
- Order `{6, 7}`: both lie under node `5`. Cutting the single shared edge `1-5` (cost 3) severs
  both at once, which beats cutting `5-6` and `5-7` separately (`2 + 5 = 7`). Answer `3`.

The third order is the interesting one: the cheapest plan cuts an edge that is *above* both marked
nodes, on the part of the tree they share.

## Background

This is a minimum **edge cut** that separates a chosen set of terminals from a fixed root, on a
tree. On a tree the cut has clean structure: severing a single edge disconnects its entire subtree
from everything above it, and the cut cost is just the sum of the cut edges. Equivalently, by
max-flow / min-cut duality it is the min cut between the root (source) and the marked nodes (sinks)
in the tree seen as a flow network — but one does not need general max flow here; the tree
structure admits a direct dynamic program.

The naive correct method is a post-order DP over the **whole** tree: `f[v]` = minimum cost to cut
all marked nodes inside the subtree of `v` away from `v`. For a child `c` reached by an edge of
cost `w`, the contribution is `w` if `c` is marked (the link to a marked node must be severed
somewhere on the way down, and cutting `w` is one option) and otherwise `min(w, f[c])` (cut here,
or solve the subproblem below). The answer is `f[root]`. This is `O(n)` per order, hence `O(q n)`
overall — correct but too slow.

The decisive observation is that an order with marked set `S` only "bends" the tree at `O(|S|)`
places: the marked nodes themselves and the branch points where their root-paths split — that is,
the pairwise lowest common ancestors. Everything between two consecutive bend points is a straight
chain, and on a chain the cheapest way to cut it is simply to remove its minimum-weight edge. So
the whole order can be answered on a compressed tree of `O(|S|)` nodes whose edges carry the
minimum edge weight of the chain they stand for.

## Evaluation settings

Judged on hidden tests covering: single-node trees and empty orders (`k = 0`, answer `0`); deep
"bamboo" paths (worst-case ancestor depth); wide stars; orders whose cheapest cut is a shared edge
high in the tree (so per-terminal cutting is suboptimal); many small orders versus a few huge
orders; and full-scale inputs with `n = 2*10^5`, `q` up to `2*10^5`, `sum |S| = 2*10^5`, and edge
costs near `10^9` so the answer overflows 32 bits. Correctness is exact (the printed integers must
match) and the time limit rules out any `O(q n)` approach.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    // read n-1 weighted tree edges
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        // TODO: store the tree
    }

    int q;
    cin >> q;
    while (q--) {
        int k;
        cin >> k;
        vector<int> S(k);
        for (auto &x : S) cin >> x;

        // TODO: minimum total cut cost so that no node of S reaches the root (node 1).
        long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
