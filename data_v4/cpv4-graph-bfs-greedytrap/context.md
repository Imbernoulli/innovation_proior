# Fewest line transfers across a metro network

## Research question

A metro network has `n` stations (numbered `0 .. n-1`) and `m` bidirectional connections. Each
connection joins two stations `u` and `v` and is operated by a **line** with an integer label `c`
(think of it as the line's colour). Many connections may share the same line label, and the same pair
of stations may be joined by several connections on different lines.

You board at station `0` and want to reach station `n-1`. You travel by riding a sequence of
connections (a *walk* — you may revisit stations and connections). Riding a connection on line `c`
immediately after a connection on line `c'` is free if `c == c'` (you stay on the same train) and
costs **one transfer** if `c != c'` (you change trains). Boarding the very first connection of the
whole trip is free.

Output the minimum number of transfers needed to get from station `0` to station `n-1`. If `0 == n-1`
output `0`. If station `n-1` is unreachable, output `-1`.

This is a shortest-path question in disguise, but the cost lives on *line changes*, not on edges or
stations, which is exactly where an edge-counting intuition goes wrong.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`). Each of the next `m` lines has three integers
  `u v c` (`0 <= u, v <= n-1`, `0 <= c <= 10^9`) describing one bidirectional connection between
  stations `u` and `v` on line `c`. Self-loops (`u == v`) and duplicate connections may appear.
- Output (stdout): a single line with the minimum number of transfers, or `-1` if station `n-1`
  cannot be reached from station `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with stations `0..4` and connections
`(0,2,line 1), (2,4,line 2), (0,1,line 5), (1,3,line 5), (3,4,line 5)`,
the answer is `0`: ride `0 -> 1 -> 3 -> 4` entirely on line 5, changing trains zero times — even
though that path is longer (3 connections) than the 2-connection route `0 -> 2 -> 4`, which would
cost 1 transfer.

## Background

The cost being minimized is the number of line changes along a walk, not the number of connections.
Two families of approach are on the table before committing to one:

- **Plain BFS / greedy by connections.** Run an ordinary breadth-first search over stations, or a
  greedy that keeps extending the current line as far as it can and only switches when forced. Both
  are `O(n + m)` and trivial to write; the open question is whether minimizing the number of
  connections (or locally avoiding switches) actually minimizes the number of *transfers*.
- **Shortest path on an augmented state.** Recognize that "free to stay on a line, pay to switch"
  is a `0/1` edge-cost structure, and search a state space that knows which line you are currently
  riding. This is the `0-1`-BFS / Dijkstra family; the open question is the exact state and how to
  keep the state space from exploding on high-degree stations.

## Evaluation settings

Judged on hidden tests covering: trips where the short (few-connection) route needs more transfers
than a longer same-line route; lines whose connections form several disconnected segments that merely
share a label; high-degree hub stations entered on many different lines; self-loops and duplicate
parallel connections; `start == destination` (`n = 1`); unreachable destinations (`-1`); and the
maximum `n = m = 2*10^5` for performance.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<int> eu(m), ev(m);
    vector<long long> ec(m);
    for (int i = 0; i < m; i++) cin >> eu[i] >> ev[i] >> ec[i];

    // TODO: compute the minimum number of line transfers from station 0 to station n-1,
    //       or -1 if n-1 is unreachable. Boarding the first line is free.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
