# Counting fastest routes through a transit map (BFS path-count, mod 1e9+7)

## Research question

A city transit map has `n` stations (numbered `1..n`) joined by `m` bidirectional, equal-cost hops.
Each hop is a concrete connection (a train line, a road) — so two stations may be joined by several
*distinct* hops. A commuter wants to travel from station `s` to station `t` using the fewest hops
possible. Among all itineraries that achieve that minimum hop count, how many *distinct* ones are
there? An itinerary is the **sequence of hops** taken, so two itineraries differ if at any step they
use a different connection (even between the same pair of stations). Because the count can be
astronomically large, report it **modulo `1_000_000_007`**.

This is the counting variant of single-source shortest paths on an unweighted graph: not "how far"
but "how many ways, at that distance." It is the engine behind reliability/diversity questions on
networks (how many equally-good routes exist), and it is exactly the kind of problem where a tiny
bookkeeping slip — accumulating from a same-distance neighbour, adding back toward the source along
an undirected edge, or fumbling the `s == t` / unreachable corners — silently produces a wrong number
that still *looks* plausible.

## Input / output contract

- Input (stdin): the first line has four integers `n m s t`
  (`1 <= n <= 2*10^5`, `0 <= m <= 4*10^5`, `1 <= s,t <= n`).
  Then `m` lines follow, each `u v` (`1 <= u,v <= n`) describing one bidirectional hop. The graph may
  contain self-loops (`u == v`) and parallel edges. A self-loop is a hop from a station to itself; it
  never shortens any route. Parallel edges are distinct hops, so an itinerary that uses one of two
  parallel `u-v` connections differs from the itinerary that uses the other.
- Output (stdout): a single line with the number of minimum-hop itineraries from `s` to `t`, taken
  modulo `1_000_000_007`. If `t` is unreachable from `s`, print `0`. If `s == t`, the unique empty
  itinerary (zero hops) counts, so print `1`.
- Time limit: 1 second. Memory: 256 MB.

Example: for the map below (`s = 1`, `t = 6`)

```
6 9 1 6
1 2
1 3
2 4
3 4
3 5
4 6
5 6
2 3
4 5
```

the answer is `3`: the minimum hop count is `3`, achieved by `1-2-4-6`, `1-3-4-6`, and `1-3-5-6`.
The edges `2-3` and `4-5` join stations at the same distance from `s` and lie on no shortest route.

## Evaluation settings

Judged on hidden tests covering: `s == t` (answer `1`); `t` unreachable (answer `0`); a single
station with no edges; chains with a unique route (answer `1`); diamond and layered graphs whose
counts multiply; graphs with self-loops, parallel edges, and abundant same-distance edges; and
large sparse graphs at `n = 2*10^5`, `m = 4*10^5`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    if (!(cin >> n >> m >> s >> t)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        // TODO: build adjacency (mind self-loops; keep parallel edges as distinct hops)
    }

    // TODO: BFS from s computing dist[] and ways[] (count of minimum-hop itineraries, mod MOD).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
