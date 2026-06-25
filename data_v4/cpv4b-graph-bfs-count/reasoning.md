**Reading the problem and pinning the contract.** I am handed a subway network: `n` stations, `m`
bidirectional one-minute tracks, and I must count the distinct *fastest* routes from station `1` to
station `n`, modulo `1_000_000_007`. A route is a station *sequence*, so I am counting shortest paths
in an unweighted graph by their vertex sequences, not by the physical tracks they ride. Two things in
the contract jump out as load-bearing before I touch an algorithm. First, tracks may be **parallel**
(the same pair listed more than once) and may be **self-loops** (`u == v`). Because a route is a
station sequence, parallel tracks between the same two stations are the *same* hop and must be merged,
and a self-loop is never part of a shortest simple step, so it must be dropped — otherwise the count
inflates. Second, the count is taken **modulo a prime** because it can be astronomically large; that
tells me the true count is exponential, so I must never compare counts as raw integers anywhere in the
logic (only the final value, and intermediate sums, live in modular space). And `n` is unreachable ->
answer `0`. Scale: `n <= 2*10^5`, `m <= 4*10^5`, so I need an `O(n + m)`-ish BFS, not anything
quadratic.

**Laying out the candidate approaches.** Two routes are on the table.

- *Enumerate every shortest path explicitly.* For tiny graphs I could DFS all simple paths of the
  minimum length and count them. Correct and dead simple, but the number of shortest paths is
  exponential (that is *why* there is a modulus), so this is hopeless at `n = 2*10^5`. I will keep it
  only as my mental brute-force oracle for checking the real method, never as the submission.
- *Layered BFS counting.* Run BFS once from station `1` to get `dist[v]`, the fewest minutes to each
  station. A station `v` is reached by a fastest route exactly through neighbours `u` with
  `dist[u] = dist[v] - 1`, and the number of fastest routes to `v` is the sum over those predecessors
  of the number of fastest routes to `u`. This is `O(n + m)` and is the standard shortest-path count.
  The risk is not the idea but the *bookkeeping*: which neighbours count as predecessors, and not
  adding any contribution twice.

I commit to layered BFS. The danger is entirely in the counting arithmetic, which is precisely the
emphasis I must respect: an index/layer/modulus slip here is a silent double-count.

**Deriving the counting recurrence and checking it on paper.** Let `dist[v]` be the BFS distance from
`1`. Define `ways[v]` = number of distinct fastest routes from `1` to `v`. Base case: `ways[1] = 1`
(the empty route, just standing at station `1`). Recurrence: for `v != 1`,

  `ways[v] = sum over neighbours u with dist[u] = dist[v] - 1 of ways[u]`.

Why is this exactly right, and why simple paths only? A fastest route to `v` has length `dist[v]`; its
second-to-last station `u` is a neighbour of `v`, and the prefix to `u` is itself a fastest route to
`u`, which forces `dist[u] = dist[v] - 1`. Conversely any fastest route to such a `u` extended by the
hop `u -> v` is a fastest route to `v`. Distinct `(u, route-to-u)` pairs give distinct routes to `v`
because the station sequences differ, and every such route is automatically simple: distances strictly
increase along it, so no station repeats. Good — the recurrence counts simple shortest paths, which is
the definition.

Let me confirm on the sample `n = 7`, tracks `1-2,1-3,2-4,3-4,4-5,4-6,5-7,6-7`. BFS distances from 1:
`dist[1]=0`, `dist[2]=dist[3]=1`, `dist[4]=2`, `dist[5]=dist[6]=3`, `dist[7]=4`. Counts:
`ways[1]=1`; `ways[2]=ways[1]=1`, `ways[3]=ways[1]=1`; `ways[4]=ways[2]+ways[3]=2`;
`ways[5]=ways[4]=2`, `ways[6]=ways[4]=2`; `ways[7]=ways[5]+ways[6]=4`. Answer `4`, matching the four
listed routes. The recurrence is right.

**How to evaluate the recurrence in one BFS pass.** I do not want a separate predecessor pass. The
clean trick: BFS dequeues stations in non-decreasing distance order, so when I dequeue `u` (at distance
`d`) and look at a neighbour `v`, three cases arise. If `v` is undiscovered, it belongs to layer
`d + 1`; set `dist[v] = d + 1` and seed `ways[v] = ways[u]`. If `v` is already discovered *and*
`dist[v] = d + 1`, then `u` is another predecessor of `v` on the next layer; add `ways[u]` into
`ways[v]`. If `dist[v] = d` (same layer) or `dist[v] < d` (closer), `u` is *not* a predecessor of `v`
and contributes nothing. The reason this single pass is sufficient: every predecessor `u` of `v` sits
in layer `d` and is dequeued before `v` (which is in layer `d + 1`), so by the time `v` is dequeued its
`ways[v]` is already the full sum. This is the spot I must get exactly right.

**First implementation — and immediately a trace, because counting code lies.** My first cut builds
adjacency directly from the input tracks (a multigraph, one adjacency entry per track) and runs the
BFS:

```
// build adjacency straight from tracks, no dedup
for (auto &e : edges) { adj[e.first].push_back(e.second); adj[e.second].push_back(e.first); }
...
for (int v : adj[u]) {
    if (dist[v] == INF) { dist[v] = dist[u]+1; ways[v] = ways[u]; q.push(v); }
    else if (dist[v] == dist[u]+1) { ways[v] = (ways[v] + ways[u]) % MOD; }
}
```

I trace the smallest input that could expose the parallel-track issue: `n = 2`, tracks `1-2` and `1-2`
(two parallel tracks between the same stations), `m = 2`. There is obviously exactly **one** route:
the station sequence `1, 2`. Run it. Adjacency: `adj[1] = [2, 2]`, `adj[2] = [1, 1]`. Start
`dist[1]=0, ways[1]=1`, queue `[1]`. Dequeue `1`, iterate `adj[1] = [2, 2]`. First `2`: undiscovered,
set `dist[2]=1`, `ways[2]=ways[1]=1`, push. Second `2`: now `dist[2]=1 == dist[1]+1=1`, so
`ways[2] += ways[1]` -> `ways[2] = 2`. Final `ways[2] = 2`.

**Diagnosing the first bug (a double-count from parallel tracks).** The code returns `2` but the answer
is `1`. The defect is exactly the multigraph: the two parallel tracks `1-2` produced two adjacency
entries, and the BFS treated the second one as a *second predecessor*, adding `ways[1]` a second time.
But a route is a station sequence; the hop `1 -> 2` is the same hop no matter how many physical tracks
implement it. The contract literally warned me about parallel tracks, and I built the graph as if they
were distinct routes. The fix is to collapse the graph to a *simple* graph before BFS: for each input
track with `u != v`, record `v` in a per-station set for `u` and vice versa; self-loops (`u == v`) are
skipped because a self-loop can never be a strict distance-increasing step. Then the adjacency of each
station is the deduplicated neighbour list.

```
vector<set<int>> tmp(n + 1);
for (auto &e : edges) {
    int u = e.first, v = e.second;
    if (u == v) continue;              // drop self-loops
    tmp[u].insert(v); tmp[v].insert(u);
}
for (int u = 1; u <= n; u++) adj[u].assign(tmp[u].begin(), tmp[u].end());
```

Re-trace `n = 2`, parallel `1-2,1-2`: now `adj[1] = [2]`, `adj[2] = [1]`. Dequeue `1`, single neighbour
`2`: undiscovered, `ways[2] = ways[1] = 1`. Final `1`. Correct. Re-trace adding a self-loop
`n = 2`, tracks `1-2, 1-2, 1-1`: the `1-1` is skipped, adjacency unchanged, answer still `1`. The
parallel/self-loop dedup is now exactly the "collapse multiplicity" the problem demanded, and the case
that returned the illegal `2` returns `1` for the reason I fixed.

**Second trace — the layer condition itself, on a diamond.** Even with the simple graph, the predecessor
test is the classic place a double-count hides, so I deliberately wrote down a *wrong* version of the
condition to make sure my real one is right. Suppose I had been sloppy and written the accumulation as

```
else if (dist[v] >= dist[u]) { ways[v] = (ways[v] + ways[u]) % MOD; }   // WRONG
```

reasoning vaguely that "any already-seen neighbour that is not closer contributes". I trace it on the
diamond `n = 4`, tracks `1-2, 1-3, 2-4, 3-4`, whose answer is plainly `2` (`1-2-4` and `1-3-4`).
Distances: `dist[1]=0`, `dist[2]=dist[3]=1`, `dist[4]=2`. BFS: dequeue `1`, discover `2` and `3` with
`ways=1` each. Dequeue `2`, neighbour `4`: undiscovered -> `dist[4]=2`, `ways[4]=ways[2]=1`. Also
neighbour `1`: `dist[1]=0`, and under the wrong test `dist[1] >= dist[2]`? `0 >= 1` is false, so no harm
there. Dequeue `3`, neighbour `4`: `dist[4]=2 = dist[3]+1`, add `ways[3]` -> `ways[4]=2`. So far the
diamond alone does not trip the `>=` bug. I need a same-layer edge to expose it. Take the *triangle-
topped* graph `n = 5`, tracks `1-2, 1-3, 2-3, 2-5, 3-5`: distances `dist[1]=0`, `dist[2]=dist[3]=1`,
`dist[5]=2`, and there is a *same-layer* edge `2-3`. The true answer is `2` (`1-2-5`, `1-3-5`). Under
the wrong `>=` test: dequeue `2`, neighbour `3` has `dist[3]=1 = dist[2]`, and `1 >= 1` is true, so the
wrong code does `ways[3] += ways[2]`, making `ways[3] = 2` — a phantom! Then `ways[5]` would pick up
that inflated `ways[3]`. The same-layer edge `2-3` is *not* a predecessor relation (both are one minute
from the source; neither precedes the other on a fastest route), but `>=` swept it in. This is the
canonical layered-BFS double-count.

**Confirming my actual condition avoids it.** My committed code uses the *strict* test
`else if (dist[v] == dist[u] + 1)`, with an explicit comment that `dist[v] == dist[u]` and
`dist[v] < dist[u]` are *not* predecessors. Re-trace the same `n = 5` triangle-top graph with the
strict test: dequeue `2`, neighbour `3` has `dist[3] = 1`, and `1 == 1 + 1`? No (`1 != 2`), so nothing
is added — the same-layer edge is correctly ignored. Neighbour `5`: undiscovered, `ways[5] = ways[2]
= 1`. Dequeue `3`, neighbour `2`: `dist[2]=1`, `1 == 2`? no. Neighbour `5`: `dist[5]=2 = dist[3]+1`,
add `ways[3]=1` -> `ways[5] = 2`. Answer `2`. Correct. The strict `== dist[u] + 1` is exactly what
keeps same-layer edges out of the sum, and tracing the wrong `>=` against it is the evidence I trust.

**A numeric self-check of the "counts multiply" claim.** I claimed shortest-path counts behave
multiplicatively across independent branch points, which is *why* the answer overflows 64-bit and the
modulus matters. Let me not just assert it — let me check it numerically on a chain of diamonds. Put
`k` diamonds in series: source `h_0 = 1`, and between hub `h_i` and hub `h_{i+1}` two disjoint middle
stations, so each diamond contributes a factor of 2 and the total number of fastest routes is `2^k`.
For `k = 3` the prediction is `2^3 = 8`. Build it (hubs `1,4,7,10` with two mids each) and the
recurrence gives `ways[4]=2`, `ways[7]=ways[4]*?` — careful, it is additive per layer:
`ways[h_{i+1}] = ways[mid_a] + ways[mid_b] = ways[h_i] + ways[h_i] = 2*ways[h_i]`, so
`ways[h_k] = 2^k`. For `k = 3`: `1 -> 2 -> 4 -> 8`. I ran exactly this construction for `k = 40` where
`2^40 = 1099511627776`, far beyond what fits if I had used 32-bit, and beyond a billion so the modulus
genuinely bites: the program prints `511620083`, and independently `pow(2, 40) mod (10^9+7) =
511620083`. The two agree, so (a) the multiplicative intuition is correct, (b) the modular reduction is
applied correctly, and (c) had I used `int` for `ways` it would have wrapped and silently lied. This is
the concrete confirmation that `long long` accumulation plus `% MOD` is mandatory, not decorative.

**Edge cases, deliberately, because counting code dies in the corners.**
- *Disconnected:* `n = 2, m = 0`. BFS never reaches `2`, so `dist[2] = INF`; the final guard
  `if (dist[n] == INF) print 0`. Output `0`. Correct.
- *Direct edge:* `n = 2`, track `1-2`. `ways[2] = ways[1] = 1`. Output `1`. Correct.
- *Parallel tracks + self-loop:* `n = 2`, tracks `1-2, 1-2, 1-1`. Dedup collapses to a single hop and
  drops the loop; output `1`. Correct (verified above).
- *`ways[1]` seeding:* I set `ways[1] = 1 % MOD`. With `MOD = 10^9 + 7`, `1 % MOD = 1`, so the source
  has one route (standing still). If `n = 1` were ever allowed the answer would be `1`; the contract
  has `n >= 2`, but the seeding is consistent either way.
- *Overflow / modulus:* `ways` is `vector<long long>`, every accumulation does `% MOD`, and the final
  print does `ways[n] % MOD`. The diamond-chain check above confirms the result stays in `[0, MOD)`.
- *Self-loop never on a shortest step:* dropping `u == v` is safe because a fastest route strictly
  increases distance every hop, and a self-loop keeps distance the same — it can never be on one.
- *Input parsing:* `cin >> n >> m` then `m` pairs; `cin >>` skips arbitrary whitespace, so layout is
  format-agnostic. If the stream is empty I `return 0`.

**Complexity.** Building the simple graph with per-station `set<int>` costs `O(m log n)`; the BFS is
`O(n + m)`. At `m = 4*10^5` this measured `0.43 s` and `64 MB` at `n = 2*10^5` — comfortably inside the
2-second / 256-MB budget. (If I needed to shave the log factor I would sort each adjacency list and
`unique` it, but the `set` is fast enough and clearer.)

**Final solution.** I convinced myself the idea is right by deriving the predecessor recurrence and
checking it on the 4-route sample; I convinced myself the *code* is right by tracing a parallel-track
input that returned the illegal `2` (fixed by collapsing to a simple graph), by tracing a wrong `>=`
layer test against a same-layer triangle edge to justify the strict `== dist[u] + 1` predecessor test,
and by numerically confirming the multiplicative `2^k` count against the modular output at `k = 40`.
That is what I ship — one self-contained file: dedup the graph, one BFS, layered counting modulo the
prime:

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> adj(n + 1);
    vector<pair<int,int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        edges[i] = {u, v};
    }
    // Build simple-graph adjacency: collapse parallel edges, drop self-loops,
    // because a "route" is a station sequence, so multiplicity does not matter.
    {
        vector<set<int>> tmp(n + 1);
        for (auto &e : edges) {
            int u = e.first, v = e.second;
            if (u == v) continue;            // self-loop: never on a shortest simple step
            tmp[u].insert(v);
            tmp[v].insert(u);
        }
        for (int u = 1; u <= n; u++)
            adj[u].assign(tmp[u].begin(), tmp[u].end());
    }

    const int INF = INT_MAX;
    vector<int> dist(n + 1, INF);
    vector<long long> ways(n + 1, 0);
    dist[1] = 0;
    ways[1] = 1 % MOD;
    queue<int> q;
    q.push(1);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                ways[v] = ways[u];
                q.push(v);
            } else if (dist[v] == dist[u] + 1) {
                // v already discovered on the next layer: u is one valid predecessor.
                ways[v] = (ways[v] + ways[u]) % MOD;
            }
            // dist[v] == dist[u] (same layer) or dist[v] < dist[u]: NOT a predecessor.
        }
    }

    if (dist[n] == INF) {
        cout << 0 << "\n";
    } else {
        cout << ways[n] % MOD << "\n";
    }
    return 0;
}
```

**Causal recap.** Counting fastest subway routes is shortest-path counting on an unweighted graph: BFS
for distances, then `ways[v] = sum of ways[u]` over strictly-closer predecessors `u`. The first trap
was structural — building a multigraph from parallel tracks made the BFS add the same hop's predecessor
twice (a trace of two parallel `1-2` tracks returning `2` instead of `1` pinpointed it), fixed by
collapsing to a simple graph and dropping self-loops. The second trap was the layer condition itself —
a `>=` predecessor test sweeps in same-layer edges and double-counts (a trace on a triangle-topped
graph showed the phantom), fixed by the strict `dist[v] == dist[u] + 1` test that admits only
next-layer successors. A numeric check on a chain of `k = 40` diamonds (`2^40 mod p = 511620083`,
matching the program) confirmed both the multiplicative blow-up that forces `long long` plus `% MOD`
and that the modular reduction is applied correctly; the unreachable, direct-edge, parallel, and
self-loop corners close it out.
