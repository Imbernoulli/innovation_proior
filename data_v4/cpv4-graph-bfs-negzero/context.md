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

## Evaluation settings

Judged on hidden tests covering: all-positive brightnesses, brightnesses mixing negatives and zeros,
all-negative graphs, single-node graphs (`n = 1`, `m = 0`), graphs with unreachable components,
zero-brightness layers, and large `n = m = 2*10^5` with `|w[i]|` near `10^9`.

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
