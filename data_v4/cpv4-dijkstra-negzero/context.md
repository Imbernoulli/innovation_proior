# Weakest-link routing in a relay network (max-min / widest path)

## Research question

A communication backbone has `n` relay stations and `m` directed links. Each directed link
`u -> v` carries an integer **margin** `w` — the signal headroom that link contributes. A margin can
be **positive** (the link is healthy), **zero** (exactly at threshold), or **negative** (the link is
degraded and eats into headroom). A *route* from the source station `s` to some station `v` is any
sequence of links `s -> ... -> v`, and the **reliability** of a route is the **minimum margin over
its links** — the route is only as good as its weakest link.

For the fixed source `s`, compute for **every** station `v` the maximum achievable reliability, i.e.
the maximum over all routes `s -> v` of the minimum margin along the route. This is the *widest path*
/ *max-min* / *bottleneck-shortest-path* objective, and because margins are routinely negative or
zero, the corners (a station reachable only through degraded links, an unreachable station, the
source itself) are exactly where the sign handling has to be exact.

## Input / output contract

- Input (stdin): the first line is three integers `n m s` (`1 <= n <= 2*10^5`,
  `0 <= m <= 5*10^5`, `1 <= s <= n`). Then `m` lines follow, each `u v w`
  (`1 <= u, v <= n`, `-10^9 <= w <= 10^9`) describing a directed link `u -> v` with margin `w`.
  Self-loops (`u == v`) and multiple links between the same pair may appear.
- Output (stdout): `n` lines. Line `v` (for `v = 1..n`) holds:
  - the integer maximum reliability of the best route `s -> v`, if `v != s` and `v` is reachable; or
  - the literal `INF` for the source line `v == s` (the empty route uses no link, so its minimum
    margin is `+infinity`); or
  - the literal `UNREACHABLE` if no route `s -> v` exists.
- Time limit: 2 seconds. Memory: 256 MB.

Example: see the worked sample in the reasoning. A reachable station whose best route still bottoms
out at margin `-2` prints `-2`; a station with **no** route prints `UNREACHABLE` — these are different
outcomes and must not collapse to the same value.

## Background

The objective is *max over paths of (min edge weight)*. Two facts shape the approach:

- **Simple paths suffice.** Appending a cycle to a route only adds links, and adding links can only
  lower (never raise) the running minimum. So an optimal route can always be taken simple, and the
  answer is well defined even though the graph may contain cycles of negative margin.
- **The relaxation is monotone in the metric, regardless of edge sign.** If the best bottleneck to
  `u` is `best[u]`, then routing one more link `u -> v` of margin `w` yields the candidate
  `min(best[u], w)`. Because `min` is monotone and the value of a route never *increases* as it is
  extended, the greedy "finalize the currently-widest unfinished node" argument of Dijkstra carries
  over verbatim from `+` to `min` and from `min`-of-distances to `max`-of-bottlenecks. The sign of
  `w` is irrelevant to this argument — what matters is monotonicity, not non-negativity. That is the
  crucial difference from ordinary shortest paths, where negative edges break Dijkstra.

Two families of approach are on the table before committing:

- **Max-min Dijkstra (max-heap).** Maintain `best[v]` = widest known bottleneck to `v`, start the
  source at `+infinity`, and repeatedly extract the unfinished node of largest `best`, relaxing its
  out-links with `min`. `O((n + m) log n)`. The open questions are the *base case* (what is the
  source's bottleneck, and what value marks "not reached yet") and the *output mapping* of the three
  outcomes — these are precisely where negative/zero margins bite.
- **Iterated relaxation (Bellman-Ford style).** Relax all edges `n - 1` times with the `min`/`max`
  rule. `O(n m)`, simpler to argue but far too slow at `m = 5*10^5`.

## Evaluation settings

Judged on hidden tests covering: all-positive margins; graphs mixing negatives, zeros, and
positives; **all-negative** graphs (reachable nodes have negative answers, which must not be confused
with unreachable); graphs with **zero** margins (a reachable node with bottleneck `0` must print `0`,
not `UNREACHABLE`); the **empty** edge set `m = 0` (only the source is reachable); `n = 1`; self-loops
and parallel edges; and large `n = 2*10^5`, `m = 5*10^5` with `|w|` up to `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;

    vector<vector<pair<int,long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    // TODO: max-min Dijkstra. best[v] = max over routes s->v of the minimum margin.
    //       Decide the source base case and the sentinel for "not reached", then
    //       print INF for the source, UNREACHABLE for nodes with no route, the
    //       integer bottleneck otherwise.

    return 0;
}
```
