# Total relay cost of a broadcast over a fiber network

## Research question

A national broadcaster operates a network of `n` relay cities numbered `1..n`, connected by `m`
bidirectional fiber links (each link joins two cities and carries a signal in one hop). A live
broadcast originates at city `1`. The signal floods outward link by link: a city is *reached* as soon
as the broadcast arrives, and the number of **hops** it took to get there is the city's *depth* — the
minimum number of links on any path from city `1`.

Each city `v` has an *importance* `w[v]` (a non-negative integer). Relaying the broadcast at city `v`
costs `depth(v) * w[v]`: cities farther from the source are more expensive to keep in sync, weighted
by how important they are. The origin city `1` has depth `0`, so it costs nothing. Cities that the
broadcast can **never reach** (no path from city `1`) are simply never lit and contribute nothing.

Output the **total relay cost** summed over every city the broadcast reaches. This is single-source
shortest path on an *unweighted* graph (BFS gives the depths), followed by a weighted sum. The whole
point is the aggregate: with many cities, large depths, and large importances, the running total grows
far beyond what a 32-bit integer can hold.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `m` (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`).
  - The second line has `n` integers `w[1..n]` (`0 <= w[v] <= 10^6`).
  - Each of the next `m` lines has two integers `a b` (`1 <= a, b <= n`) describing a bidirectional
    link between cities `a` and `b`. Self-loops (`a == b`) and repeated links may appear and are
    harmless.
- Output (stdout): a single line with the total relay cost over all cities reachable from city `1`.
- Time limit: 1 second. Memory: 256 MB.

Example: with `6` cities, importances `[3, 5, 2, 8, 1, 4]`, and links `1-2, 1-3, 2-4, 3-4, 4-5`,
city `6` is unreachable. Depths are `1:0, 2:1, 3:1, 4:2, 5:3`, so the cost is
`0*3 + 1*5 + 1*2 + 2*8 + 3*1 = 0 + 5 + 2 + 16 + 3 = 26`.

## Background

Two design questions sit in front of the implementation:

- **Which shortest-path engine?** The links are unweighted (every hop counts as `1`), so the depth of
  a city is the minimum hop count from city `1`. Breadth-first search computes exactly that in
  `O(n + m)`: it visits cities in non-decreasing depth order, fixing each city's depth the first time
  it is dequeued. Dijkstra would also work but is overkill (and slower by a log factor) when all edges
  have equal weight; plain BFS is the right tool.

- **How large can the answer get, and what type holds it?** A reachable city's depth is at most
  `n - 1`, and importances reach `10^6`. A single term `depth(v) * w[v]` can therefore be on the
  order of `2*10^5 * 10^6 = 2*10^11`, already past the 32-bit signed range of about `2.1*10^9`. The
  *sum* over up to `2*10^5` cities can reach roughly `2*10^16`. That comfortably exceeds a 32-bit
  `int` (by seven orders of magnitude) yet still fits a 64-bit `long long` (whose ceiling is about
  `9.2*10^18`). So the answer must be accumulated — and each product must be formed — in 64-bit
  arithmetic. Storing depths themselves in `int` is fine (they are at most `n - 1`); the danger is the
  *multiplication and the running total*.

## Evaluation settings

Judged on hidden tests covering: a single city (`n = 1`, `m = 0`, answer `0`); graphs that are fully
connected, a long path, a star, a grid, and several disconnected components (so unreachable cities are
exercised); importances that are all zero, mixed, and uniformly `10^6`; self-loops and duplicate
links; and large instances `n = m = 2*10^5` whose total cost lands near `2*10^16` — far past 32-bit
range, so any `int` accumulator or `int` product is a silent wrong answer that nonetheless passes the
tiny sample.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<long long> w(n + 1);
    for (int v = 1; v <= n; v++) cin >> w[v];

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    // TODO: BFS from city 1 to get each reachable city's hop-depth, then sum
    //       depth(v) * w[v] over all reachable cities (mind the magnitude).
    long long total = 0;

    cout << total << "\n";
    return 0;
}
```
