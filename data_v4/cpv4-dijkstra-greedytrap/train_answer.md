**Problem.** A directed graph on `n` stations; each edge `(u, v)` has a *line* id `c` and a fare `w`.
A route from `1` to `n` costs the sum of fares plus a fixed surcharge `S` for every place where two
consecutive edges have different lines (the first boarding is free, arrival is free). Read
`n, m, S` and the `m` edges from stdin; print the minimum route cost, or `-1` if `n` is unreachable.

**Why the obvious method is wrong.** All weights are non-negative, so Dijkstra applies — the trap is
choosing the wrong notion of "node". Plain Dijkstra with one label per station (`dist[v]` = cheapest
arrival) silently assumes the cheapest way to *reach* a station summarizes the past. It does not,
because the next surcharge depends on the *line you arrived on*. Counterexample with `S = 5`: edges
`1->2`(Red, 1), `1->2`(Blue, 3), `2->3`(Blue, 1). Station-only Dijkstra sets `dist[2] = 1` on Red,
then pays a transfer into the Blue edge: `1 + 1 + 5 = 7`. But arriving more *expensively* on Blue
(`3`) lets `2->3` continue Blue-to-Blue for free: `3 + 1 + 0 = 4 < 7`. The cheapest arrival blocked
the free continuation, so a station alone is not enough state.

**Key idea — Dijkstra over `(station, arriving line)`.** Make the state the pair `(u, lc)` where `lc`
is the line of the edge by which you reached `u`. Use a start sentinel `lc = 0` (real lines are
`>= 1`) meaning "no line ridden yet". Transition from `(u, lc)` along `(u, v, c, w)`:

```
new cost = dist[(u, lc)] + w + ( (lc != 0 && c != lc) ? S : 0 )   into state (v, c)
```

Run Dijkstra (min-heap) on these states. Because every reachable state is `(v, c)` for some edge,
there are `O(m)` states, so this is `O((n + m) log m)`.

**Correctness.** On the expanded state graph all edge costs are non-negative (fares `>= 0`, surcharge
`>= 0`), so Dijkstra is valid: the first time a state whose station is `n` is popped, its cost is the
global minimum over all ways to *arrive* at `n` — and arriving costs nothing extra, so that is the
answer. If the heap empties without popping any `n`-state, `n` is unreachable and the answer is `-1`.
The state correctly captures the future: the only thing a future edge's surcharge depends on is the
arriving line, which is exactly the second component of the state.

**Pitfalls.**
1. *State too coarse.* One label per station is wrong (the counterexample above: `7` vs the true
   `4`). The line you arrived on must be part of the state.
2. *Taxing the first edge.* Guard the surcharge with `lc != 0`; the sentinel `0` compares unequal to
   every real line, so a bare `c != lc` charges a phantom `+S` on the first boarding (turns the clean
   answer `4` into `9`).
3. *Stale heap entries.* Old, worse labels for a state remain in the heap; on pop, skip any triple
   whose stored cost no longer equals `best[u][lc]`. Without the skip, stale labels re-expand
   adjacency lists and can TLE. (They never change the answer, only the running time.)
4. *Overflow.* With `m` up to `2*10^5` and fares/`S` up to `10^9`, costs reach `~4*10^14`; use
   `long long`. An `int` is a silent wrong-answer.

**Edge cases.** `n` unreachable -> `-1`; `S = 0` -> ordinary shortest path; single line everywhere ->
no transfer ever; self-loops add only non-negative cost and never help; parallel edges on different
lines between the same pair are distinct states and are handled.

**Complexity.** `O((n + m) log m)` time, `O(n + m)` space (one cost per reachable `(station, line)`).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    long long S;
    if (!(cin >> n >> m >> S)) return 0;

    // edges[u] = list of (v, color, fare)
    vector<vector<array<long long,3>>> edges(n + 1);
    // colors are 1..C; we compress per-node arrival operator state.
    // State = (node, last_color). last_color = 0 means "no edge used yet" (start).
    // We index states. To keep it bounded we only create states reachable through edges.

    // Collect all colors to map them; but a state's "last_color" is the color of the
    // edge we arrived on. We use the raw color value as part of the key.
    // dist over (node, color). Use a hash map keyed by node*BIG + color, but colors
    // can be large; instead store per node a map<color,dist>.
    for (int i = 0; i < m; i++) {
        long long u, v, c, w;
        cin >> u >> v >> c >> w;
        edges[u].push_back({v, c, w});
    }

    // best[node] : map from last_color -> min cost arriving at node having last
    // traversed an edge of that color.
    vector<unordered_map<long long,long long>> best(n + 1);

    // Priority queue of (cost, node, last_color).
    priority_queue<array<long long,3>, vector<array<long long,3>>, greater<array<long long,3>>> pq;

    // Start at node 1 with last_color = 0 (a sentinel meaning "no operator yet").
    // 0 is never used as a real color (real colors are >= 1).
    best[1][0] = 0;
    pq.push({0, 1, 0});

    long long answer = -1;

    while (!pq.empty()) {
        auto top = pq.top(); pq.pop();
        long long d = top[0], u = top[1], lc = top[2];
        auto it = best[u].find(lc);
        if (it == best[u].end() || it->second != d) continue; // stale
        if (u == n) { answer = d; break; }
        for (auto &e : edges[u]) {
            long long v = e[0], c = e[1], w = e[2];
            // surcharge only if we already used an edge (lc != 0) and color changes.
            long long nd = d + w + ((lc != 0 && c != lc) ? S : 0);
            auto vit = best[v].find(c);
            if (vit == best[v].end() || nd < vit->second) {
                best[v][c] = nd;
                pq.push({nd, v, c});
            }
        }
    }

    cout << answer << "\n";
    return 0;
}
```
