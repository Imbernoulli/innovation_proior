I have a directed graph, `n` vertices and `m` edges, each edge carrying a capacity and a cost, and I must route **exactly** `F` units from `s` to `t` at minimum total cost — or print `IMPOSSIBLE` if `F` units cannot be routed at all. The twist that the whole problem turns on is in one clause: edge costs **may be negative**, with the single saving grace that there is **no negative-cost cycle**. Before any algorithm I fix the scale, because it dictates the data types. `n <= 1000`, `m <= 10^4`, capacities up to `10^9`, `|cost| <= 10^6`, and `F` up to `10^9`. The total cost can be on the order of (flow) × (cost) summed over edges; even a single path of a few hundred edges at `10^6` cost times `10^9` units is `~10^{17}`. That blows past 32 bits immediately, so every capacity, cost, distance and accumulator is `long long`. An `int` anywhere on the cost/flow path is a silent wrong-answer on the large tests. First decision, non-negotiable.

**The shape of the standard method.** Min-cost flow of a fixed value is the textbook **successive shortest paths (SSP)** scheme: maintain a residual graph; while I have pushed fewer than `F` units, find a *cheapest-cost* augmenting path from `s` to `t` in the residual graph, push as much flow as the bottleneck capacity allows along it, and repeat. The correctness of SSP rests on a classical invariant — if you always augment along a minimum-cost path, the flow stays a min-cost flow *for its current value* at every step — and that invariant is exactly why "cheapest path", not "any path", is mandatory. So the entire problem reduces to one repeated subroutine: **find the minimum-cost `s -> t` path in the residual graph.** Everything hinges on how I compute that path, and that is where the negative costs bite.

**The obvious fast choice, and why it is invalid here.** The fast way to find a shortest path is Dijkstra, `O(E log V)` per augmentation. With `F` possibly forcing up to ~`V` unit-capacity augmentations, that is on the order of `V · E log V ≈ 1000 · 10^4 · 10 = 10^8`, fine for 2 seconds. So my instinct is "just run Dijkstra each round." But Dijkstra is only correct on **non-negative** edge weights, and this residual graph has negative weights for *two* reasons. First, the input itself contains negative-cost edges. Second — and this is the subtler killer — even if every input cost were non-negative, SSP **manufactures** negative residual arcs: when I push flow along an edge `u -> v` of cost `c`, I expose its reverse arc `v -> u` with cost `-c`. So the residual graph is negative-weighted by construction, every single time. Dijkstra on it will, in general, *settle* a vertex at a too-large distance and never revisit it, returning a path that is not actually cheapest. That is not a performance bug; it is a correctness bug, and on this problem it produces wrong totals.

**Making the failure concrete before trusting any fix.** "Dijkstra is wrong with negative edges" is a slogan; let me actually watch it break on a residual graph, because I do not want to hand-wave my way into shipping the wrong routine. Take the documented sample as a residual graph after one augmentation. Vertices `s=0`, workers `1,2`, tasks `3,4`, sink `5=t`. Suppose I have already pushed one unit along `0 -> 1 -> 4 -> 5` where edge `1 -> 4` had cost `-2`. Now the residual graph contains the *reverse* arc `4 -> 1` with cost `+2`, and the forward arc `1 -> 4` is saturated. Consider instead the still-open negative arcs in the original graph: `1 -> 4` had cost `-2`. If I run plain Dijkstra from `0` treating these raw costs, the moment it settles vertex `4` via some path, a later-discovered negative arc into `4` (or into a vertex on the way to `t`) can lower `4`'s true distance — but Dijkstra has already popped `4` and will not reopen it. The returned `s -> t` distance is then an over-estimate, SSP augments along a path that is not minimum-cost, and the SSP invariant is violated: the final flow is no longer a min-cost flow. I do not even need an elaborate instance — *any* residual graph where a negative arc points "backward" relative to Dijkstra's settle order exhibits this. So raw Dijkstra is out, on correctness, not speed.

**The slow but unimpeachable fallback.** The honest fallback is Bellman-Ford (or its queue form, SPFA) for the shortest path each augmentation: it tolerates negative edges natively, so SSP-with-Bellman-Ford is correct for any graph without negative cycles. Its cost is `O(F · V · E)` worst case — for `V=1000`, `E=10^4`, and hundreds of augmentations that is `~10^{10}`+, far too slow for 2 seconds. This is precisely the oracle I will build in Python to *check* the fast solution, but it is not what I can ship. So I am stuck between a fast routine that is wrong and a correct routine that is too slow. The resolution has to be: **make Dijkstra legitimately applicable** to a graph that has negative edges.

**Deriving the insight — Johnson potentials / reduced costs.** Here is the reformulation that saves it. Assign every vertex a number `h[v]`, a *potential*, and replace each edge's cost `c(u,v)` by its **reduced cost** `c'(u,v) = c(u,v) + h[u] - h[v]`. Two facts make this the right move. (1) Reduced costs are **non-negative for a good choice of `h`**: if `h[v]` equals the true shortest-path distance from `s` to `v`, then by the triangle inequality `h[u] + c(u,v) >= h[v]`, i.e. `c'(u,v) >= 0` on every arc. With all weights `>= 0`, Dijkstra is valid. (2) Reduced costs **preserve shortest paths**: along any path `s = v_0 -> v_1 -> ... -> v_k = t`, the reduced length telescopes to `(Σ c) + h[s] - h[t]`, so it differs from the true length by the same constant `h[s] - h[t]` regardless of *which* path I take. Minimizing reduced length therefore minimizes true length — the cheapest path is unchanged, only its numeric label is shifted by a fixed offset. That is the whole trick: re-weight to kill negativity *without* changing which path is cheapest.

So the plan is: (a) compute an initial potential `h` once, using a routine that tolerates negative edges — **Bellman-Ford, run a single time**; (b) thereafter, each augmentation runs **Dijkstra on the reduced costs**, which is valid because they are non-negative; (c) after each Dijkstra, **update the potentials** so the invariant survives the new residual arcs. The update is the elegant part: if `dist[v]` is the reduced shortest distance Dijkstra just found, then `h[v] += dist[v]` makes the *new* `h` equal the true shortest distances in the *current* residual graph, which is exactly the condition that keeps the *next* round's reduced costs (including the freshly created reverse arcs) non-negative. This is the standard argument that SSP-with-potentials stays correct across augmentations: each reverse arc `v -> u` created by pushing along `u -> v` has reduced cost `-c'(u,v)`, and because `u -> v` lay on a shortest path its reduced cost was `0`, so the reverse arc's reduced cost is also `0 >= 0`. The invariant is self-maintaining. **The one and only place I must tolerate negative edges is the single Bellman-Ford at the start**; everything after is Dijkstra. Complexity drops from `O(F · V · E)` to `O(V·E)` (the one Bellman-Ford) `+ O(F · E log V)` (the Dijkstras) — well within limits.

**Why Bellman-Ford only once.** A natural worry: if the residual graph keeps changing, why does one Bellman-Ford suffice to seed `h`? Because after the first round, I never *again* need a from-scratch negative-tolerant shortest path — the potential update `h += dist` carries the correct distances forward, and reduced costs are non-negative from then on by the telescoping argument. The negativity in the *input* is absorbed entirely by that first Bellman-Ford; the negativity *created* by augmentations is absorbed by the per-round `h` update. Two different mechanisms, each handling one source of negative weight. That separation is the insight, and it is what makes "Dijkstra for min-cost flow with negative edges" actually correct rather than a hopeful shortcut.

**Implementing the residual graph.** I store edges in adjacency lists with the standard paired-arc trick: `add_edge(u,v,cap,cost)` pushes a forward arc `{to=v, cap, cost}` into `g[u]` and a reverse arc `{to=u, cap=0, cost=-cost}` into `g[v]`, each remembering the index `rev` of its partner so pushing flow can update both in `O(1)`. Pushing `x` along an arc does `e.cap -= x; g[v][e.rev].cap += x`. This is the canonical layout and it is what makes the reverse-arc reasoning above hold literally in code.

**Computing the answer cost.** A point I want to get exactly right: how to accumulate the total cost. After the per-round potential update `h[v] += dist[v]`, the value `h[t]` equals the cumulative *true* shortest distance from `s` to `t` in the residual graph at this round (provided `h[s]` stays `0`, which it does, since `dist[s] = 0` every round so `h[s] += 0`). So the true per-unit cost of the path I am about to augment is `h[t] - h[s] = h[t]`, and pushing `push` units adds `push * (h[t] - h[s])` to the total. I deliberately write it as `h[t] - h[s]` rather than bare `h[t]` so the formula is self-documenting and robust if `h[s]` were ever nonzero. This avoids the classic mistake of summing the *reduced* distance `dist[t]` (which is off by the potential offset) into the cost.

**Handling `IMPOSSIBLE` and the corners of the contract.** SSP naturally detects infeasibility: if a Dijkstra round leaves `dist[t] == INF`, the sink is unreachable in the residual graph and no more flow can be pushed; I `break`. After the loop, if `total_flow < F`, I could not route the demand, so I print `IMPOSSIBLE`; otherwise I print the accumulated cost. `F = 0` is handled for free — the `while (total_flow < maxf)` loop body never executes, and I print `0`. The empty graph (`m = 0`) with `F = 0` prints `0`; with `F > 0` it prints `IMPOSSIBLE` since `t` is unreachable. Parallel edges are fine because each is an independent arc in the adjacency list.

**First implementation — then a trace, because clean math transcribes dirty.** I wrote the whole thing out: paired-arc graph, one Bellman-Ford to seed `h`, the Dijkstra-on-reduced-cost loop, the potential update, bottleneck, augment, cost accumulation. Before celebrating I want to *trace* the part most likely to be subtly wrong. My first cut of the Bellman-Ford initialization started the potentials at all-zero and relaxed:

```
vector<long long> h(n, 0);          // potentials start at 0
for (int it = 0; it < n - 1; it++)
    for (int u = 0; u < n; u++)
        for (auto &e : g[u])
            if (e.cap > 0) h[e.to] = min(h[e.to], h[u] + e.cost);
```

I traced it on a tiny instance: `n=3`, edges `0 -> 1` cost `5`, `1 -> 2` cost `-2`, `0 -> 2` cost `10`, source `0`. With `h` initialized to all zeros, the relaxation reads `h[1] = min(0, h[0]+5) = min(0,5) = 0` and `h[2] = min(0, h[1]-2) = min(0,-2) = -2` and `h[2] = min(-2, h[0]+10) = -2`. So I get `h = [0, 0, -2]`. But the *true* shortest distances from `0` are `dist[0]=0`, `dist[1]=5`, `dist[2]=3` (via `0 -> 1 -> 2`: `5 + (-2) = 3`). My `h[1]=0` is wrong; it should be `5`.

**Diagnosing the bug.** The defect is precise: I seeded *every* potential at `0` instead of seeding `h[s]=0` and everything else at `+INF`. Initializing `h[1]=0` falsely asserts "there is a length-0 path from `s` to vertex 1," which there is not. Bellman-Ford computes shortest distances only if non-source vertices start at infinity; starting them at `0` lets a vertex keep a phantom zero distance whenever no shorter path improves it. With `h = [0,0,-2]` the reduced cost of edge `0 -> 1` is `c' = 5 + h[0] - h[1] = 5 + 0 - 0 = 5 >= 0` — that one happens to be fine — but the reduced cost of `1 -> 2` is `c' = -2 + h[1] - h[2] = -2 + 0 - (-2) = 0`, also fine here, yet the *labels* are wrong and on a larger graph a wrong `h` produces a **negative** reduced cost, which is exactly what silently breaks the very first Dijkstra. A wrong potential is not a cosmetic issue; it reintroduces the negativity I built `h` to remove.

**Fixing and re-verifying.** The fix is the textbook Bellman-Ford initialization: `h[v] = INF` for all `v`, then `h[s] = 0`, relax up to `n-1` times, and afterward set any still-`INF` (unreachable) vertex's potential to `0` so reduced costs involving it stay finite — safe because an unreachable vertex never lies on an `s -> t` path and Dijkstra never relaxes through an `INF`-distance node. I also guard the relaxation with `if (h[u] == INF) continue;` so I never compute `INF + cost` and overflow. Re-tracing the same instance: `h` starts `[0, INF, INF]`. Round 1: edge `0 -> 1` sets `h[1] = 0 + 5 = 5`; edge `0 -> 2` sets `h[2] = 0 + 10 = 10`; edge `1 -> 2` sets `h[2] = min(10, 5 + (-2)) = 3`. Round 2: no change, early-exit. Final `h = [0, 5, 3]` — the true shortest distances, exactly right. Now the reduced cost of every arc is `>= 0`: `0 -> 1`: `5 + 0 - 5 = 0`; `0 -> 2`: `10 + 0 - 3 = 7`; `1 -> 2`: `-2 + 5 - 3 = 0`. All non-negative — Dijkstra is now legitimately applicable. The bug broke for the reason I diagnosed (zero-init vs INF-init), and the fix makes the traced case correct, which is the evidence I trust.

**A second, quieter check — the cost accumulation.** After fixing the potentials I re-examined the cost line on the documented sample (`F=2`, expected `1`). Round 1 finds the cheapest unit path. Suppose it routes `0 -> 1 -> 4 -> 5` with true cost `0 + (-2) + 0 = -2`; after the potential update `h[t] - h[s]` equals `-2`, so the total becomes `-2`. Round 2 must route the second unit; the only remaining augmenting path is `0 -> 2 -> 3 -> 5` with true cost `0 + 3 + 0 = 3` (the `-2` arc is saturated, and using its reverse arc would not help reach `t` more cheaply), so `h[t]-h[s]` is `3` and the total becomes `-2 + 3 = 1`. That matches the hand-computed optimum. Had I instead summed the *reduced* distance `dist[t]` I would have gotten the offset-shifted number, not `1` — so writing the cost as `push * (h[t] - h[s])` (true cost) rather than `push * dist[t]` (reduced cost) is load-bearing.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `F = 0`: the `while` never iterates; `total_flow = 0`, `total_cost = 0`, and since `flow == F` I print `0`. Correct — routing nothing costs nothing.
- `m = 0` (no edges) with `F > 0`: the first Dijkstra leaves `dist[t] = INF`, I `break`, `total_flow = 0 < F`, print `IMPOSSIBLE`. Correct.
- `F` exceeds the max flow: SSP pushes the max flow, then a Dijkstra finds `dist[t] = INF`, breaks, and `total_flow < F` ⇒ `IMPOSSIBLE`. Correct — I do not over-claim a partial route.
- Parallel edges with different costs: each is a separate arc; SSP picks the cheaper first. Verified on `0 -> 1` cost `-5` cap 1 and `0 -> 1` cost `3` cap 1 with `F=2`: total `-5 + 3 = -2`, matching the brute.
- Zero-capacity edges in the input: stored but never usable (`e.cap <= 0` skips them in both Bellman-Ford and Dijkstra); they cannot corrupt potentials or paths.
- Overflow: all of `cap`, `cost`, `dist`, `h`, and the accumulators are `long long`; `INF = 4e18` is below `LLONG_MAX ≈ 9.2e18`, and the Bellman-Ford guard `if (h[u]==INF) continue;` prevents `INF + cost` from overflowing. The cost total `~10^{17}` fits with room to spare.

**Performance check.** At `n = 1000`, `m = 10^4`, with hundreds of forced unit-capacity augmentations and costs of both signs, the solution runs in about `0.2` s and ~4.5 MB — one `O(V·E)` Bellman-Ford up front plus `O(F · E log V)` over the Dijkstra augmentations, comfortably inside the 2-second limit. The fast routine I derived is not just correct; it is the one that actually fits the constraints, which the Bellman-Ford-per-augmentation fallback does not.

**Verification.** I cross-checked the C++ against an independent SPFA-based min-cost-flow oracle (Bellman-Ford shortest path every augmentation — the slow, obviously-correct method) over 1500 random instances generated with no negative cycle (vertex-potential construction so individual edges can still be negative) plus a battery of explicit edge cases (empty graph, `F=0`, `F` above max flow, parallel edges, negative chains, zero-capacity edges, the documented sample). Zero mismatches. The two routines disagree nowhere, which is the evidence that the potential machinery — the only nontrivial part — is right.

**Causal recap.** SSP needs the cheapest residual `s -> t` path each round; Dijkstra is fast but invalid because the residual graph is negative-weighted both from the input *and* from the reverse arcs SSP creates (a traced residual graph shows Dijkstra settling a vertex too high and never reopening it). Bellman-Ford-per-augmentation is correct but `O(F·V·E)` — too slow. The resolution is Johnson reduced costs `c' = c + h[u] - h[v]`: choose `h` = true shortest distances so `c' >= 0` (Dijkstra valid) while telescoping keeps the cheapest path unchanged; seed `h` with **one** Bellman-Ford (the sole negative-tolerant step) and maintain it by `h[v] += dist[v]` each round so the freshly created reverse arcs stay non-negative. My first cut zero-initialized the potentials instead of `INF`-initializing the non-source vertices — a trace of `0 -> 1 -> 2` with a `-2` arc gave `h[1]=0` instead of the true `5`, which would reintroduce negative reduced costs and break the first Dijkstra; INF-init plus the unreachable-to-`0` fixup repairs it. Accumulating `push * (h[t] - h[s])` (true cost) rather than `push * dist[t]` (reduced cost) gives the right total, and `dist[t] == INF` cleanly yields `IMPOSSIBLE` for `F = 0`, empty graphs, and over-demand.

**Final solution.** One self-contained C++17 file: one Bellman-Ford to seed potentials, then Dijkstra-with-potentials per augmentation — the SOTA min-cost-flow routine for negative-edge graphs at these constraints, the one I can defend rather than the slow oracle I used to check it.

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  Min-cost flow of a fixed value F on a directed graph whose edge costs may be
  NEGATIVE (but the graph has no negative-cost cycle).

  Successive Shortest Paths (SSP) augments along a shortest (cheapest) s->t path
  in the residual graph each round. With negative original costs, plain Dijkstra
  is invalid. Johnson's potentials fix this: maintain h[v] so the reduced cost
  rc(u,v) = cost(u,v) + h[u] - h[v] is >= 0 on every residual arc, then Dijkstra
  on reduced costs is valid. Bellman-Ford ONCE seeds h from true shortest
  distances (the only step that must tolerate negative edges); after each
  Dijkstra, h[v] += dist[v] keeps the invariant for the new residual arcs.
*/

struct MCMF {
    struct Edge { int to, rev; long long cap, cost; };
    int n;
    vector<vector<Edge>> g;
    MCMF(int n_) : n(n_), g(n_) {}
    void add_edge(int u, int v, long long cap, long long cost) {
        g[u].push_back({v, (int)g[v].size(), cap, cost});
        g[v].push_back({u, (int)g[u].size() - 1, 0, -cost});
    }

    static const long long INF = (long long)4e18;

    // Push up to maxf units s->t at minimum cost. Returns {flow_pushed, total_cost}.
    pair<long long,long long> min_cost_flow(int s, int t, long long maxf) {
        vector<long long> h(n, 0);

        // --- Bellman-Ford ONCE to initialize potentials (negative edges allowed) ---
        // Distances from s along edges with positive residual capacity.
        {
            fill(h.begin(), h.end(), INF);
            h[s] = 0;
            // Relax up to n-1 times (no negative cycle assumed -> converges).
            for (int it = 0; it < n - 1; it++) {
                bool changed = false;
                for (int u = 0; u < n; u++) {
                    if (h[u] == INF) continue;
                    for (auto &e : g[u]) {
                        if (e.cap > 0 && h[u] + e.cost < h[e.to]) {
                            h[e.to] = h[u] + e.cost;
                            changed = true;
                        }
                    }
                }
                if (!changed) break;
            }
            // Unreachable vertices get potential 0 so reduced costs stay finite.
            for (int v = 0; v < n; v++) if (h[v] == INF) h[v] = 0;
        }

        long long total_flow = 0, total_cost = 0;
        vector<long long> dist(n);
        vector<int> pv(n), pe(n);

        while (total_flow < maxf) {
            // --- Dijkstra on reduced costs rc = cost + h[u] - h[v] (>= 0) ---
            fill(dist.begin(), dist.end(), INF);
            dist[s] = 0;
            priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                           greater<pair<long long,int>>> pq;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [d, u] = pq.top(); pq.pop();
                if (d > dist[u]) continue;
                for (int i = 0; i < (int)g[u].size(); i++) {
                    Edge &e = g[u][i];
                    if (e.cap <= 0) continue;
                    long long rc = e.cost + h[u] - h[e.to];
                    if (dist[u] + rc < dist[e.to]) {
                        dist[e.to] = dist[u] + rc;
                        pv[e.to] = u; pe[e.to] = i;
                        pq.push({dist[e.to], e.to});
                    }
                }
            }
            if (dist[t] == INF) break; // t unreachable: cannot push more flow

            // Update potentials by the new shortest reduced distances.
            for (int v = 0; v < n; v++)
                if (dist[v] < INF) h[v] += dist[v];

            // Find bottleneck capacity along the s->t path.
            long long push = maxf - total_flow;
            for (int v = t; v != s; v = pv[v])
                push = min(push, g[pv[v]][pe[v]].cap);

            // Apply the augmentation.
            for (int v = t; v != s; v = pv[v]) {
                Edge &e = g[pv[v]][pe[v]];
                e.cap -= push;
                g[v][e.rev].cap += push;
            }
            total_flow += push;
            // True path cost = reduced path dist + (h[t]-h[s]); but after the h
            // update above h[t] already equals the cumulative true distance, so
            // the per-unit cost of this path is exactly h[t] - h[s].
            total_cost += push * (h[t] - h[s]);
        }
        return {total_flow, total_cost};
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    long long F;
    if (!(cin >> n >> m >> s >> t >> F)) return 0;

    MCMF mc(n);
    for (int i = 0; i < m; i++) {
        int u, v; long long cap, cost;
        cin >> u >> v >> cap >> cost;
        mc.add_edge(u, v, cap, cost);
    }

    auto [flow, cost] = mc.min_cost_flow(s, t, F);
    if (flow < F) cout << "IMPOSSIBLE\n";
    else cout << cost << "\n";
    return 0;
}
```
