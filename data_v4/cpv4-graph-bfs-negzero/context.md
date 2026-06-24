# Brightest ripple layer of a broadcast graph

## Research question

A signal is broadcast from a single transmitter node `s` in an undirected, unweighted network of `n`
nodes and `m` cables. The signal floods outward one hop per tick: every node first lit at tick `d`
sits on **ripple layer `d`**, where `d` is its shortest hop-distance from `s`. Each node `i` carries a
fixed integer **brightness** `w[i]` that may be **negative, zero, or positive** (a node can absorb
light rather than emit it). The brightness of a ripple layer is the sum of `w[i]` over all nodes whose
shortest distance from `s` equals exactly `d`. Nodes the signal never reaches (no path from `s`) belong
to no layer and contribute nothing.

Output the brightness of the **single brightest ripple layer**. Layer `0` always exists because it
contains the transmitter `s` itself, so there is always at least one layer to compare — and because
brightnesses can all be negative, the answer is **not** floored at `0`: it can be negative.

This is the layered-BFS subproblem that appears inside diffusion, broadcast-scheduling, and
shortest-path-tree analyses. Getting the one-source version exactly right — including the all-negative,
single-node, and unreachable-node corners — is the whole point.

## Input / output contract

- Input (stdin):
  - The first line holds three integers `n m s` (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`,
    `1 <= s <= n`): node count, cable count, and the transmitter node (nodes are `1`-indexed).
  - The second line holds `n` integers `w[1..n]` (`-10^9 <= w[i] <= 10^9`), the brightnesses.
  - Each of the next `m` lines holds two integers `u v` (`1 <= u, v <= n`, `u != v`) describing an
    undirected cable between `u` and `v`. There are no self-loops; parallel cables may appear and are
    harmless.
- Output (stdout): a single line with the brightness of the brightest ripple layer.
- Time limit: 1 second. Memory: 256 MB.

Example: for the graph below with `s = 1`

```
6 6 1
3 -5 0 7 2 -1
1 2
1 3
2 4
3 4
4 5
5 6
```

the layers are `{1}` (sum `3`), `{2,3}` (sum `-5+0=-5`), `{4}` (sum `7`), `{5}` (sum `2`),
`{6}` (sum `-1`), so the answer is `7`.

## Background

The signal spreads by hop count on an unweighted graph, which is exactly the structure a
breadth-first search exposes: BFS from `s` labels every reachable node with its shortest hop-distance,
and nodes dequeued in order fall into contiguous distance layers `0, 1, 2, ...`. Two design questions
have to be settled before committing to code:

- **Which layer is "brightest"?** We must group nodes by their *shortest* distance (not just any path
  length) and sum brightnesses per group, then take the maximum group-sum. Because a layer can be a
  single zero-brightness node, a layer-sum of `0` does **not** mean the layer is empty — emptiness must
  be tracked by distance, never inferred from the sum.
- **What is the base case for the maximum?** The running maximum cannot start at `0`. If every reachable
  node is negative, the true brightest layer is itself negative, and a `0` seed would silently return
  `0` — a wrong answer. The maximum must start below every attainable layer-sum.

## Evaluation settings

Judged on hidden tests covering: all-positive brightnesses, brightnesses mixing negatives and zeros,
**all-negative** graphs (the answer must be the least-negative layer, not `0`), single-node graphs
(`n = 1`, `m = 0`: the answer is `w[s]` itself, possibly negative or zero), graphs with **unreachable
components** (those nodes must be excluded, not counted as a phantom distance-`0` or distance-`-1`
layer), zero-brightness layers (a real layer whose sum is `0`), and large `n = m = 2*10^5` with
`|w[i]|` near `10^9` (a single layer can sum to `~2*10^14`, overflowing a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i];

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // TODO: BFS shortest hop-distances from s, sum brightness per distance layer,
    //       and report the maximum layer-sum (answer may be negative).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
