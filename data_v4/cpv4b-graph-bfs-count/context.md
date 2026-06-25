# Counting the fastest subway routes

## Research question

A city subway network has `n` stations (numbered `1..n`) connected by `m` bidirectional tracks; every
track takes exactly one minute to ride. A commuter starts at station `1` and wants to reach station `n`
in the fewest minutes possible. Among all the *fastest* itineraries, how many **distinct routes** are
there?

A route is identified by the **sequence of stations** it visits. So two itineraries that pass through
the same stations in the same order are the *same* route even if the city happened to lay several
parallel tracks between two stations; conversely two routes are different the moment their station
sequences differ. The number of fastest routes can be astronomically large, so report it **modulo
`1_000_000_007`**. If station `n` is unreachable from station `1`, the answer is `0`.

This is shortest-path *counting* on an unweighted graph. The arithmetic — accumulating counts across
BFS layers, collapsing parallel tracks, and folding everything modulo a prime — is exactly where a
double-count slips in unnoticed, so getting the layer bookkeeping right is the whole game.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`2 <= n <= 2*10^5`, `0 <= m <= 4*10^5`). Each of the next `m` lines has two integers `u v`
  (`1 <= u, v <= n`) describing a bidirectional track between stations `u` and `v`. Tracks may be
  **parallel** (the same pair more than once) and may be **self-loops** (`u == v`).
- Output (stdout): a single line with the number of distinct fastest routes from station `1` to
  station `n`, taken modulo `1_000_000_007`. Print `0` if `n` is unreachable.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with `n = 7` and tracks
`1-2, 1-3, 2-4, 3-4, 4-5, 4-6, 5-7, 6-7`, the fastest trip takes 4 minutes and there are `4`
distinct routes (`1-2-4-5-7`, `1-2-4-6-7`, `1-3-4-5-7`, `1-3-4-6-7`), so the answer is `4`.

## Background

Because every track costs the same one minute, the fastest trip is a shortest path in an unweighted
graph, which BFS finds in `O(n + m)`. Counting the shortest paths is the layered refinement of BFS:
a station `v` at distance `d` is reached by a fastest route through exactly those neighbours `u` whose
distance is `d - 1`, and the number of fastest routes to `v` is the sum of the counts at those
predecessors. Two design questions are open before committing:

- **What exactly is a predecessor?** Only a neighbour strictly one layer *closer* to the source
  contributes. A neighbour in the *same* layer, or a neighbour *farther* away, must not be added in —
  that is the spot where a double-count is born.
- **How do parallel tracks and self-loops interact with the route definition?** Since a route is a
  station sequence, multiplicity of tracks between the same pair must be collapsed and self-loops
  ignored, or the count is inflated.

## Evaluation settings

Judged on hidden tests covering: a direct `1-n` edge, disconnected graphs (answer `0`), graphs with
parallel tracks and self-loops, "diamond" graphs whose counts multiply (so the answer overflows 64-bit
and the modulus is genuinely exercised), grids/lattices with binomial route counts, and large
instances at `n = 2*10^5`, `m = 4*10^5`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        // TODO: build a simple-graph adjacency (collapse parallel tracks, drop self-loops),
        // then BFS from station 1 counting shortest routes to station n modulo MOD.
    }

    long long answer = 0;
    cout << answer << "\n";
    return 0;
}
```
