# Cheapest metro trip with line-transfer surcharges

## Research question

A city's transit network is a directed graph on `n` stations. Each ride is a directed edge `(u, v)`
operated by some **line** `c` (an integer id, think "Red line", "Blue line", ...) and costs a fare
`w`. You start at station `1` and want to reach station `n`. The twist is the fare card rule: every
time you board an edge whose line **differs from the line of the edge you just rode**, you pay a fixed
**transfer surcharge** `S` on top of the fare. Boarding your very first edge is never a transfer
(there is no previous line yet), and *arriving* at `n` costs nothing extra. Riding two consecutive
edges of the same line is free of surcharge even across an intermediate station.

The cost of a route is `(sum of fares of the edges used) + S * (number of line changes along it)`.
Output the minimum possible cost of any route from `1` to `n`, or `-1` if `n` is unreachable.

This is a shortest-path problem, but the cost of extending a route depends on *which line you arrived
on*, not just on where you are.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `m`, `S`
  (`2 <= n <= 2*10^5`, `0 <= m <= 2*10^5`, `0 <= S <= 10^9`).
  Then `m` lines follow, each `u v c w` describing a directed edge from `u` to `v` on line `c`
  with fare `w` (`1 <= u, v <= n`, `1 <= c <= 2*10^5`, `0 <= w <= 10^9`). Self-loops and parallel
  edges (including parallel edges on different lines) may occur.
- Output (stdout): a single line with the minimum total cost from station `1` to station `n`, or
  `-1` if no route exists.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with `S = 3` and edges `1->2` (Red, 4), `2->5` (Red, 9), `1->3` (Blue, 2), `3->4` (Blue, 2),
`4->5` (Red, 1), `2->4` (Blue, 1), the cheapest route to station `5` is
`1 ->(Blue,2) 3 ->(Blue,2) 4 ->(Red,1) 5`, costing `2 + 2 + 1 + 3 = 8` (one Blue->Red change), which
beats the all-Red route `1->2->5` at `4 + 9 = 13`.

## Evaluation settings

Judged on hidden tests covering: tiny graphs with multiple lines connecting the same pair of stations
at different fares; `S = 0`; disconnected targets (answer `-1`); self-loops and parallel edges on
different lines between the same pair; single-line graphs (no transfer ever); and large
`n, m = 2*10^5` with fares and `S` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    long long S;
    if (!(cin >> n >> m >> S)) return 0;

    vector<vector<array<long long,3>>> edges(n + 1); // edges[u] = {(v, line, fare), ...}
    for (int i = 0; i < m; i++) {
        long long u, v, c, w;
        cin >> u >> v >> c >> w;
        edges[u].push_back({v, c, w});
    }

    // TODO: minimum cost from station 1 to station n, charging S on each line change.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
