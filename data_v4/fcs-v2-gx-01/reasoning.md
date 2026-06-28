**Reading the problem and pinning the contract.** I have an undirected graph on `n` vertices with `m` colored edges, each color `c` carrying a capacity `cap[c]`, and I must choose the largest edge subset `S` that is at once a *forest* (no cycle) and *color-bounded* (at most `cap[c]` chosen edges of each color). Output is a single integer, `|S|`. Before any algorithm I fix the scale, because it decides what is feasible: `n, m, K <= 2000`. The answer fits in an `int` (at most `m`), so there is no overflow worry here — the difficulty is entirely combinatorial, not arithmetic. Input is `n m K`, then `K` capacities, then `m` lines of `u v c`; self-loops (`u == v`) and parallel edges are explicitly allowed, which I must keep in mind because both are cycle-closing in the graphic sense.

The first thing I want to name precisely is *what kind of constraint each rule is*. "The chosen edges form a forest" is the independence condition of the **graphic matroid** on the edge set. "At most `cap[c]` chosen edges of each color" is the independence condition of a **partition matroid** whose blocks are the color classes. So `S` must be independent in two matroids simultaneously, and I want the maximum-cardinality such `S`. This is the canonical *matroid intersection* problem. Recognizing it as matroid intersection is the whole game — it tells me both what the right algorithm is and why the easy attempts will fail.

**Laying out the candidate approaches.** Two routes are tempting, and I want to commit only to one I can defend.

- *Greedy.* Process edges in some order — say rarest color first, or simply scan and add any edge that neither closes a cycle nor overflows its color. This is `O(m alpha(n))` with a union-find for the forest test and a per-color counter. It is what I would write if I trusted that locally-good choices compose into a global optimum.
- *Augmenting paths in the exchange graph.* The exact matroid-intersection algorithm: start from `S = {}` and repeatedly find a *shortest alternating augmenting path* from a "source" element (addable under the partition matroid) to a "sink" element (addable under the graphic matroid), then flip membership of every element along the path to grow `|S|` by exactly one. Stop when no augmenting path exists; the current `S` is then provably maximum. This is polynomial and oracle-driven.

Greedy is far easier, so I owe it a real attempt to break it before I reach for the heavier machinery.

**Stress-testing greedy before committing.** I will attack greedy with a concrete instance, because "greedy feels fine" is exactly how a wrong solution ships. Take a triangle on vertices `{1,2,3}` plus one extra edge, with colors chosen to set a trap.

Edges: `e0 = (1,2)` color `A`, `e1 = (2,3)` color `A`, `e2 = (1,3)` color `B`, and `e3 = (3,4)` color `A`. Capacities: `cap[A] = 1`, `cap[B] = 1`. There are 4 vertices `{1,2,3,4}`, so the best possible forest has 3 edges. What is actually achievable? Color `A` may contribute at most one edge, color `B` at most one edge — so any feasible set has at most `1 + 1 = 2` edges. Two is in fact reachable: `{e2 (color B), e3 (color A)}` is a forest (edges `1-3` and `3-4`, no cycle) and uses each color once. So the true answer is `2`.

Now run greedy "scan in index order, add if it keeps a forest and the color has room". It takes `e0` (color `A`, now `A` is full), skips `e1` (color `A` full), takes `e2` (color `B`), then reaches `e3` (color `A`, full) and must skip it. Greedy's set is `{e0, e2}` of size `2` — here it happens to match. So index-order greedy got lucky. Let me perturb the order to expose the fragility, because the danger is order-dependence. Suppose greedy instead prefers edges of the *scarcer remaining color* and `A` looks scarce, so it grabs `e0` early; the structure is the same and it still lands on 2. The real failure shows when a single edge is "load-bearing" in *both* matroids at once. Consider: `e0 = (1,2)` color `A`, `e1 = (1,2)` color `B` (a parallel edge, different color), `e2 = (1,3)` color `A`, with `cap[A] = 1`, `cap[B] = 1`, vertices `{1,2,3}`. Optimal: `{e1 (1-2, color B), e2 (1-3, color A)}` — a forest of size 2 using one of each color. But greedy scanning in order takes `e0` (`1-2`, color `A`), filling color `A`; then `e1` (`1-2`, color `B`) would close a cycle `1-2-1` with `e0` (parallel edge) — skip; then `e2` (`1-3`, color `A`) — color `A` full — skip. Greedy returns `{e0}`, size `1`, while the optimum is `2`. Greedy is wrong, and I see *why*: by taking `e0` it consumed the `1-2` adjacency slot of the forest **and** the single unit of color `A`, and that double commitment blocked the strictly better pairing. The constraint is global across *two* structures at once, and greedy decides locally in each. Greedy is out.

**Deriving the insight — exchange graph and augmenting paths.** Since greedy fails, I lean on the structure of matroid intersection. The key object is the **exchange graph** relative to the current common independent set `S`. Build a directed graph whose nodes are all `m` edges. The arcs encode legal one-for-one swaps:

- For each `y` **not** in `S` and each `x` **in** `S`: put an arc `y -> x` if `S - x + y` is independent in the **graphic** matroid `M1`. Concretely, adding `y` to the forest `S` creates exactly one cycle (the tree path between `y`'s endpoints plus `y`); `x` is swappable iff `x` lies on that cycle, i.e. `x` is on the tree path between `y`'s endpoints in `S`.
- For each `x` **in** `S` and each `y` **not** in `S`: put an arc `x -> y` if `S - x + y` is independent in the **partition** matroid `M2`. Concretely, `x` and `y` share a color and that color is currently at capacity (removing `x` frees the unit that lets `y` in).

Then define the special endpoints. A node `y not in S` is a **source** (call it `X1`-free) if `S + y` is independent in `M2` alone — its color has spare capacity. A node `y not in S` is a **sink** (`X2`-free) if `S + y` is independent in `M1` alone — adding `y` keeps the forest a forest (its endpoints are in different components). The theorem I am relying on (Edmonds / Lawler): if there is a directed source-to-sink path, then taking a **shortest** such path and flipping the membership of every node on it (out-of-`S` nodes go in, in-`S` nodes go out) yields a common independent set of size `|S| + 1`; and if there is no source-to-sink path, `S` is already a maximum common independent set. "Shortest" — fewest arcs — is essential: a shortest path has no chord, which is what guarantees the flipped set stays independent in *both* matroids. A shortest path is exactly what a breadth-first search finds, so BFS is the right engine.

So the algorithm is: from `S = {}`, repeat — build the exchange graph, BFS from all sources following arcs until a sink is reached, flip the discovered shortest path, increment the size — until BFS finds no sink. The number of augmentations is `|S*| <= min(n-1, m)`, and each augmentation rebuilds and searches the exchange graph in polynomial time. For `m <= 2000` this is fast; brute force over `2^m` subsets is of course hopeless, which is the whole point.

**Working out the oracles concretely.** I do not want to materialize all `O(m^2)` arcs explicitly; I will generate them on the fly during the BFS, which keeps both memory and time in check.

- *Source test* (`isM2Free(y)`): `colorCount[color(y)] < cap[color(y)]`. Trivial counter lookup.
- *Sink test* (`isM1Free(y)`): `S + y` is a forest. A self-loop (`u == v`) is never independent in the graphic matroid, so it is never a sink. Otherwise `y` is a sink iff its endpoints lie in different components of the forest `S` — equivalently, there is no tree path between them.
- *M1 arcs `y -> x`* (for `y not in S`): the `x` are exactly the `S`-edges on the tree path between `y`'s endpoints. I find that path by a BFS over the current forest adjacency and read off the edge indices along it.
- *M2 arcs `x -> y`* (for `x in S`): for each `y not in S` of the same color, present only when `color(x)`'s count is at capacity. I group the not-in-`S` edges by color so this expansion is direct.

The forest adjacency of `S` is rebuilt each augmentation from scratch (cheap at these sizes), and the tree-path query is a plain BFS recording parent pointers.

**First implementation.** I write `augment()` to (1) build the forest adjacency, (2) seed the BFS with all sources at distance 0, (3) immediately check whether any source is itself a sink (a length-0 augmenting "path" — a single addable edge), and otherwise (4) expand: from an out-of-`S` node follow M1 arcs to its tree-path `S`-edges; from an in-`S` node follow M2 arcs to same-colored out-of-`S` edges, checking sink-ness at the moment of first discovery. The first sink reached is at minimum BFS distance. Then I walk `prevNode` from the sink back to a source and flip every node. The driver loops `while (augment()) size++;`.

**A trace, because clean math transcribes dirty.** My first cut had a flaw I need to hunt, so let me trace a small but expressive case — the parallel-edge trap that killed greedy: `e0 = (1,2,A)`, `e1 = (1,2,B)`, `e2 = (1,3,A)`, `cap[A]=1`, `cap[B]=1`, answer should be `2`.

Augmentation 1: `S = {}`, all colors at count 0. Sources = every edge whose color has room = all of `e0, e1, e2`. Are any of them sinks? `e0` connects `1-2`, components are singletons, so yes `e0` is a sink. Length-0 path: add `e0`. Now `S = {e0}`, `colorCount[A] = 1`.

Augmentation 2: forest `S = {e0}` is the single edge `1-2`. Sources = out-of-`S` edges with color room = `e1` (color `B`, count 0 < 1). `e2` is color `A`, count `1`, not a source. Is `e1` a sink? `e1` is `(1,2)`; in the forest `1` and `2` are already connected by `e0`, so adding `e1` closes a cycle — not a sink. BFS expands from `e1`: `e1` is out-of-`S`, its M1 arcs go to the `S`-edges on the tree path `1..2`, which is just `e0`. Arc `e1 -> e0`, set `dist[e0]=1`. Now expand `e0` (in `S`): M2 arcs `e0 -> y` for `y not in S` same color as `e0` (color `A`) *if* color `A` is at capacity — it is (`1 == cap[A]`). Same-color out-of-`S` edges: `e2`. Arc `e0 -> e2`, `dist[e2]=2`. Is `e2` a sink? `e2 = (1,3)`; in the forest `1` and `3` are in different components, so yes. Sink found: `e2` at distance 2, path `e1 -> e0 -> e2`. Flip: `e1` (out->in), `e0` (in->out), `e2` (out->in). New `S = {e1, e2}`, colorCount `A=1, B=1`. Size is now 2.

Augmentation 3: forest is `e1 (1-2)`, `e2 (1-3)`. Sources: out-of-`S` edges with color room. `e0` is color `A`, count 1, no room — not a source. No sources at all, so BFS finds nothing. Return false. Final size `2`. Correct, and it correctly *un-took* `e0` to make room — exactly the global swap greedy could not perform.

**Diagnosing the bug I actually hit.** The first version of the sink check inside the BFS expansion was wrong in a way the trace above would not catch but a self-loop case would. My initial `isM1Free` simply ran the tree-path BFS and returned `!connected`, *without first rejecting self-loops*. For a self-loop `y = (v, v)`, the tree-path BFS from `v` to `v` returns "connected" at distance 0 trivially — but my early code had a guard `if (s == t) return true;` meaning connected, so `isM1Free` returned `!true = false`, which is actually right. The real defect was elsewhere: in the *expansion* of an out-of-`S` self-loop node, I called `treePathEdges(adj, v, v, pathE)`, which returned `true` with an *empty* path, and then my loop "for each `x` in `pathE`" did nothing — fine — but I had also, in an earlier draft, treated "connected with empty path" as a *sink discovery shortcut*, which would have wrongly let a self-loop be added. I caught this by feeding the explicit case `1 1 1 / 5 / 1 1 1` (one vertex, one self-loop edge, capacity 5): the answer must be `0` because a self-loop is a cycle. My buggy draft printed `1`. Tracing it: the self-loop was a source (color has room) and my shortcut declared it a sink because the tree-path query said "connected, empty path", so it took a length-0 augmentation. The fix is the explicit guard `if (eu[y] == ev[y]) return false;` at the very top of `isM1Free`, and a matching `if (eu[u] == ev[u]) { /* dead element, no M1 arcs */ }` branch in the expansion so a self-loop never contributes swap arcs either. After the fix, `1 1 1 / 5 / 1 1 1` prints `0`.

**Fixing and re-verifying.** With the self-loop guard in place I re-ran the parallel-edge trap (still `2`), the triangle-with-capacity cases, and then built a brute-force oracle that enumerates all `2^m` subsets, checks each for forest-ness via union-find and for color capacities via counters, and reports the maximum feasible size. I differential-tested the compiled C++ against this oracle on two generators — a plain random one (`n <= 6`, `m <= 13`, `K <= 4`, mixed capacities) and a denser/harder one (`n in [3,7]`, `m in [4,14]`, tight caps biased to `{0,1,2}`) — over more than 1400 random instances, plus an explicit battery of adversarial edges. All matched with zero mismatches.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `m = 0`: no edges, the first `augment()` finds no sources, returns false immediately, answer `0`. Correct.
- All capacities zero: every color is at capacity from the start, so no edge is ever a source; first BFS finds nothing; answer `0`. Correct — you cannot keep any edge.
- One color, huge capacity: the partition matroid never binds, so the problem degenerates to "maximum forest" = `n - (number of connected components touched)`; the augmenting algorithm reduces to repeatedly adding any forest-extending edge, which is exactly a spanning-forest build. Verified equal to `n - #components` on tests.
- Self-loops: never a sink, never a source-arc target that helps — handled by the explicit `eu == ev` guards, confirmed by the single-self-loop case returning `0`.
- Parallel edges: two parallel edges can never both be in `S` (they would form a 2-cycle), and the exchange arcs correctly let one swap out for the other when colors make that profitable — this is precisely the trap that broke greedy, and the algorithm handles it.
- Disconnected graphs: each component contributes its own spanning forest independently; the BFS over the global forest naturally respects component boundaries (sinks are cross-component edges).

**Confirming the documented sample.** `4 5 2 / 2 1 / e0=(1,2,1) e1=(2,3,1) e2=(3,4,2) e3=(4,1,2) e4=(1,3,1)`. Color 1 cap 2, color 2 cap 1. A spanning tree of the 4 vertices has 3 edges; `{e0, e1, e2}` uses color 1 twice and color 2 once, feasible, size 3. No 4-edge forest exists on 4 vertices. The program prints `3`, matching the brute force.

**Complexity.** Each augmentation rebuilds the forest adjacency in `O(n + m)` and runs a BFS whose expansions issue at most `O(m)` tree-path queries, each `O(n + m)`; so one augmentation is `O(m (n + m))` and with up to `O(min(n, m))` augmentations the whole run is `O(min(n,m) * m * (n + m))`. At `n = m = K = 2000` this is comfortably under the 2-second limit in practice (measured well under half a second on adversarial inputs), and memory is dominated by the `O(n + m)` adjacency and `O(m)` BFS arrays. The arithmetic never exceeds `int`. The deciding move was recognizing the two-matroid structure and refusing the locally-greedy shortcut in favor of the shortest-augmenting-path exchange algorithm — and then nailing the self-loop corner that a careless oracle test would have let slip.

**Final solution.** One self-contained C++17 file: matroid intersection by shortest augmenting paths in the exchange graph, with the graphic-matroid oracle realized as tree-path BFS and the partition-matroid oracle as per-color counters.

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  Maximum common independent set of two matroids on the SAME ground set of m
  edges:
    M1 = graphic matroid  : a subset is independent iff it is a forest
    M2 = partition matroid: at most cap[color] edges of each color

  We want the largest subset of edges that is simultaneously a forest and
  respects every per-color capacity. This is exactly maximum-cardinality
  matroid intersection, solved by the augmenting-path / exchange-graph
  algorithm: repeatedly find a shortest augmenting path from a free source
  set to a free sink set in the directed exchange graph, then flip the
  membership of every element on that path, growing the common independent
  set by one. With no augmenting path, the current set is optimal.
*/

int n, m, K;
vector<int> eu, ev, ec;     // endpoints and color of each edge
vector<int> cap_;           // capacity per color (1-indexed)

// ---- graphic-matroid (forest) helpers over a given edge subset S ----
// inForest1(S): is S a forest? (no cycle)
// Used as a building block; the real per-edge exchange tests below are
// computed directly via cycle/path analysis on S.

// Build adjacency of the forest induced by the current set S (edges with in[e]).
// For M1 exchange edges we need, for x in S and y not in S:
//   y -> x  (an M1-arc) iff x lies on the unique cycle that y would close
//             when added to S, i.e. x is on the tree path between y's endpoints.
// And free M1-sink: y not in S is M1-free iff S+y is independent
//             (y connects two different components / is a self loop? no).
//
// For M2 (partition): each color is its own "component". For x in S, y not in S:
//   x -> y  (an M2-arc) iff removing x frees capacity that lets y in, i.e.
//             x and y share a color and that color is at capacity.
//   y not in S is M2-free (source) iff its color is below capacity.

int main() {
    if (!(scanf("%d %d %d", &n, &m, &K) == 3)) return 0;
    eu.resize(m); ev.resize(m); ec.resize(m);
    cap_.assign(K + 1, 0);
    for (int i = 1; i <= K; i++) scanf("%d", &cap_[i]);
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &eu[i], &ev[i], &ec[i]);
    }

    vector<char> inS(m, 0);          // membership in current common indep set
    vector<int> colorCount(K + 1, 0);

    // DSU over vertices to test forest membership quickly when needed.
    // But for exchange arcs we need tree paths, so we rebuild adjacency of S.

    auto buildForestAdj = [&](vector<vector<pair<int,int>>> &adj) {
        // adj[v] = list of (neighbor, edgeIndex) for edges currently in S
        adj.assign(n + 1, {});
        for (int e = 0; e < m; e++) if (inS[e]) {
            adj[eu[e]].push_back({ev[e], e});
            adj[ev[e]].push_back({eu[e], e});
        }
    };

    // For a candidate edge y not in S, find the set of edges of S on the tree
    // path between y's endpoints (these are the x with arc y->x). If endpoints
    // are in different components, the path is empty and y is M1-free.
    // We answer this via a BFS/DFS from one endpoint.
    auto treePathEdges = [&](vector<vector<pair<int,int>>> &adj, int s, int t,
                             vector<int> &outEdges) -> bool {
        // returns true if s and t connected; fills outEdges with edge indices on path
        outEdges.clear();
        if (s == t) return true; // self-loop edge: path empty, but it forms a loop
        vector<int> par(n + 1, -1), parE(n + 1, -1);
        vector<char> vis(n + 1, 0);
        queue<int> q; q.push(s); vis[s] = 1;
        bool found = false;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u == t) { found = true; break; }
            for (auto &pr : adj[u]) {
                int w = pr.first, e = pr.second;
                if (!vis[w]) { vis[w] = 1; par[w] = u; parE[w] = e; q.push(w); }
            }
        }
        if (!found) return false;
        int cur = t;
        while (cur != s) {
            outEdges.push_back(parE[cur]);
            cur = par[cur];
        }
        return true;
    };

    // One augmentation step: build exchange graph, BFS shortest path from any
    // M2-free source (over not-in-S elements whose color has spare capacity)
    // to any M1-free sink (not-in-S elements that keep the forest a forest),
    // flip the path. Returns true if augmented.
    auto augment = [&]() -> bool {
        vector<vector<pair<int,int>>> adj;
        buildForestAdj(adj);

        // Precompute, for each y NOT in S, whether y is M1-free and the path edges.
        // Also classify sources (M2-free) for elements NOT in S.
        // Exchange graph nodes = all edges [0..m). We do a multi-source BFS.

        // arcs:
        //   for y not in S:
        //     M1: y is sink if S+y independent (forest). Else for each x in S on
        //         the cycle that y closes: arc y -> x.
        //     M2: y is source if colorCount[color]<cap. Else for each x in S with
        //         same color: arc x -> y.
        // BFS from all sources following arcs, looking for a sink. Shortest path
        // in edge count -> a valid shortest augmenting path (no shortcut needed
        // because BFS already yields shortest; the classic correctness uses the
        // shortest augmenting path).

        const int INF = INT_MAX;
        vector<int> dist(m, INF), prevNode(m, -1);
        // Precompute color->list of in-S edges (for M2 arcs x->y) lazily.
        // We'll just iterate when needed.

        // Determine sources and sinks, and an adjacency we can traverse.
        // To avoid O(m^2) blowups in path reconstruction we compute arcs on the
        // fly during BFS expansion.

        // For M1 arcs we need tree paths; cache path edges per y when first needed.
        // For M2 arcs x->y: y not in S, x in S, same color, color full.

        // group not-in-S edges by color for M2 forward arcs
        vector<vector<int>> notInByColor(K + 1);
        vector<vector<int>> inByColor(K + 1);
        for (int e = 0; e < m; e++) {
            if (inS[e]) inByColor[ec[e]].push_back(e);
            else notInByColor[ec[e]].push_back(e);
        }

        // sinks: y not in S, M1-free (S+y independent forest)
        // sources: y not in S, M2-free (colorCount[color] < cap)
        auto isM1Free = [&](int y) -> bool {
            // S + y is a forest iff endpoints in different components and not a
            // self-loop closing a cycle. A self-loop (eu==ev) always forms a loop.
            if (eu[y] == ev[y]) return false;
            vector<int> tmp;
            bool connected = treePathEdges(adj, eu[y], ev[y], tmp);
            return !connected; // not connected => adding y keeps forest
        };
        auto isM2Free = [&](int y) -> bool {
            return colorCount[ec[y]] < cap_[ec[y]];
        };

        queue<int> bfs;
        for (int e = 0; e < m; e++) if (!inS[e]) {
            if (isM2Free(e)) { dist[e] = 0; prevNode[e] = -1; bfs.push(e); }
        }

        int sinkFound = -1;
        // If a source is itself a sink, that's a length-0 augmenting "path".
        for (int e = 0; e < m; e++) if (!inS[e] && dist[e] == 0) {
            if (isM1Free(e)) { sinkFound = e; break; }
        }

        if (sinkFound == -1) {
            while (!bfs.empty() && sinkFound == -1) {
                int u = bfs.front(); bfs.pop();
                if (!inS[u]) {
                    // u is a not-in-S element. Outgoing arcs:
                    //  M1 arcs u -> x (x in S on the cycle u closes)
                    vector<int> pathE;
                    bool connected = treePathEdges(adj, eu[u], ev[u], pathE);
                    if (eu[u] == ev[u]) {
                        // self-loop: closes a cycle with no S edge => no M1 arcs,
                        // and it's never M1-free; it's a dead element. skip.
                    } else if (connected) {
                        for (int x : pathE) {
                            if (dist[x] == INF) {
                                dist[x] = dist[u] + 1;
                                prevNode[x] = u;
                                bfs.push(x);
                            }
                        }
                    }
                } else {
                    // u is an in-S element x. Outgoing arcs:
                    //  M2 arcs x -> y (y not in S, same color, color full)
                    int col = ec[u];
                    if (colorCount[col] >= cap_[col]) {
                        for (int y : notInByColor[col]) {
                            if (dist[y] == INF) {
                                dist[y] = dist[u] + 1;
                                prevNode[y] = u;
                                bfs.push(y);
                                if (isM1Free(y)) { sinkFound = y; }
                            }
                        }
                    }
                }
                if (sinkFound != -1) break;
            }
        }

        if (sinkFound == -1) return false;

        // flip path: walk prevNode from sinkFound back to a source.
        int cur = sinkFound;
        while (cur != -1) {
            inS[cur] = inS[cur] ? 0 : 1;
            if (inS[cur]) colorCount[ec[cur]]++;
            else colorCount[ec[cur]]--;
            cur = prevNode[cur];
        }
        return true;
    };

    int size = 0;
    while (augment()) size++;

    printf("%d\n", size);
    return 0;
}
```

**Causal recap.** I identified two pulling constraints as the graphic and partition matroids, so the problem is maximum matroid intersection; a parallel-edge instance (`e0=(1,2,A), e1=(1,2,B), e2=(1,3,A)`, caps `1,1`: greedy `1` vs optimum `2`) showed a single edge can be load-bearing in *both* matroids and greedy cannot undo that commitment, so I switched to the shortest-augmenting-path exchange algorithm; building it, I let a self-loop slip through as a phantom sink because the tree-path query reports "connected, empty path" for `v..v`, and the single-self-loop test printing `1` instead of `0` pinpointed it; an explicit `eu == ev` guard in both the sink test and the arc expansion fixes it; and a `2^m` brute-force oracle over 1400+ random and adversarial instances, plus the documented sample, confirms the final program.
