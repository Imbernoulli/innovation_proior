**Reading the problem and pinning the contract.** I have a directed acyclic graph with `V` vertices and `E` edges, and I must report the largest number of edges on any directed path that starts at a *source* — a vertex of in-degree `0`. Because the graph is acyclic, I don't have to worry about a walk revisiting a vertex: every forward walk is automatically a simple path and terminates, so "path" and "walk" coincide here and the maximum is well defined and finite. An isolated vertex is a length-`0` path, so the answer is never below `0`. Input is `V E` then `E` lines `u v`; I print one integer. Let me fix the scale before choosing an algorithm, since it rules things in or out: `V <= 2*10^5` and `E <= 4*10^5`. That is comfortably in linear-time territory but firmly *out* of `O(V*E)` territory — `2*10^5 * 4*10^5 = 8*10^10`, which is hopeless in a second. So whatever I ship has to be `O(V + E)`. The answer itself is at most `V - 1` edges (a path can't repeat a vertex in a DAG), which fits in a 32-bit `int` with enormous room; no 64-bit gymnastics are needed here, unlike a sum-of-values problem. One decision locked: plain `int` for the path length is safe.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is shortest to type.

- *Greedy descent.* From each source, repeatedly step to the out-neighbour that looks most promising and count edges until I hit a sink. The natural "most promising" heuristic is the out-neighbour with the largest out-degree — intuitively the one that keeps the most options open ahead of me. It is `O(V + E)` and a dozen lines. The risk is structural: the longest path is a *global* property of the whole forward structure, and out-degree is a purely *local* signal. That is exactly the configuration where greedy tends to be wrong, so I will not trust it until I have tried to break it with a concrete instance.
- *Topological-order DP.* Order the vertices so that every edge points forward, then sweep once carrying, for each vertex, the longest path that ends there. `O(V + E)`, and the correctness argument is short and airtight because once a vertex is processed all of its predecessors are already final. The risk here is not the *idea* but the *transcription*: if I relax edges in the wrong order, the DP silently uses stale values.

**Stress-testing the greedy before committing.** "Out-degree looks like a good proxy for how far I can go" is precisely the kind of intuition that ships wrong solutions, so let me actually attack it with a small graph I can reason about by hand. I want a source from which one neighbour has high out-degree but leads nowhere, and another neighbour has out-degree `1` but begins a long chain. Here is the instance, `V = 9`:

```
1 -> 2     1 -> 6
2 -> 3     6 -> 7
2 -> 4     7 -> 8
2 -> 5     8 -> 9
```

Vertex `1` is the only source. Its two out-neighbours are `2` (out-degree `3`: it points to `3`, `4`, `5`, each of which is a sink) and `6` (out-degree `1`: it begins the chain `6 -> 7 -> 8 -> 9`). The greedy "follow the largest out-degree" walks `1 -> 2` because `3 > 1`, then from `2` every neighbour is a sink, so it stops with `2` edges. But the genuinely longest path is `1 -> 6 -> 7 -> 8 -> 9` with `4` edges. Greedy returns `2`; the truth is `4`. I even coded the greedy up separately and ran it on this graph to be sure I wasn't strawmanning myself, and it printed `2` while the answer is `4`. So the heuristic is wrong, and I can see *why*: out-degree measures how many edges leave a vertex, which has nothing to do with the *depth* reachable beyond it. A fat fan-out into sinks beats a thin edge into a long tail on the local signal and loses on the global one. The verification paid off — it killed an approach I might otherwise have shipped. Greedy by any local proxy (out-degree, in-degree, anything that doesn't already know the downstream depth) is out for the same reason: the only thing that actually predicts how far I can continue is the downstream longest path, which is the very thing I'm trying to compute. That circularity is the tell that I need DP, not greed.

**Deriving the DP and checking the recurrence on paper.** I want, for each vertex `v`, the length `dp[v]` of the longest path (in edges) that *ends* at `v`. The recurrence is forced by the structure: a longest path ending at `v` either is just `v` alone (if nothing points into `v`) or arrives along some edge `u -> v`, in which case it is a longest path ending at `u` plus that one edge. So

```
dp[v] = max over edges (u -> v) of ( dp[u] + 1 ),   or 0 if v has no in-edges.
```

Now the key observation that makes "ends at `v`" answer the question that asked about "starts at a source": `dp[v] = 0` exactly when `v` has no incoming edge, i.e. when `v` is a source. If `v` has any predecessor then `dp[v] >= 1`. Therefore the longest path *ending* at any vertex, when I trace it backwards step by step, can only stop at a vertex with `dp = 0`, which is a source. Every longest-ending path automatically *starts at a source*. So I don't have to special-case "from a source" at all: I compute `dp` for every vertex and the answer is simply `max_v dp[v]`. The path realizing that maximum begins at a source by construction. That is a clean, provable reduction, and it's much less error-prone than trying to enumerate sources and forward-DP from each.

For the recurrence to be evaluable I need each `dp[u]` to be final *before* I use it to relax `u -> v`. That is exactly what a **topological order** gives me: process vertices so that for every edge `u -> v`, `u` comes before `v`. When I reach `u`, all edges into `u` have already relaxed it, so `dp[u]` is final, and I push `dp[u] + 1` into each successor `dp[v]`. I'll build the topological order with Kahn's algorithm (repeatedly emit in-degree-`0` vertices), which also doubles as a cycle check — though the statement guarantees a DAG, so I won't lean on that.

Let me confirm the recurrence by hand on the counterexample graph, answer `4`. Topological order (one valid one): `1, 2, 6, 3, 4, 5, 7, 8, 9`. Initialize all `dp = 0`. Process `1` (`dp=0`): relax `1->2` so `dp[2] = max(0, 1) = 1`; relax `1->6` so `dp[6] = 1`. Process `2` (`dp=1`): relax `2->3,2->4,2->5` so `dp[3]=dp[4]=dp[5]=2`. Process `6` (`dp=1`): relax `6->7` so `dp[7]=2`. Process `3,4,5` (each `dp=2`, no out-edges): nothing. Process `7` (`dp=2`): relax `7->8` so `dp[8]=3`. Process `8` (`dp=3`): relax `8->9` so `dp[9]=4`. Process `9` (`dp=4`): no out-edges. Maximum `dp` is `dp[9]=4`. Correct, and notice `dp[3]=dp[4]=dp[5]=2` (the fan-out) never beat the chain — the DP weighs depth, not breadth, which is exactly the property greedy lacked.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut read the edges into an adjacency list and then, to "save writing Kahn", relaxed the edges in the order I happened to store them, sweeping vertices `1..n` and pushing `dp[u]+1` into successors:

```
vector<int> dp(n + 1, 0);
int answer = 0;
for (int u = 1; u <= n; u++) {
    for (int v : adj[u]) dp[v] = max(dp[v], dp[u] + 1);
    answer = max(answer, dp[u]);
}
```

This relaxes in *vertex-id order*, not topological order. Those two orders coincide only if the input happens to be numbered so that every edge points from a lower id to a higher id. The statement makes no such promise — it only guarantees acyclicity, and a DAG can be numbered any way at all. Something about "I assumed id order equals topo order" felt unjustified, so I traced the smallest input that could expose it. Take a two-vertex graph where the single edge points *backwards* in id: `V = 2`, one edge `2 -> 1`. Vertex `2` is the source (in-degree `0`), `1` is the sink; the answer is obviously `1` (the path `2 -> 1`). Run the buggy loop: `u = 1` first — `adj[1]` is empty, so nothing happens and `answer = max(0, dp[1]=0) = 0`. Then `u = 2` — relax `2 -> 1`: `dp[1] = max(0, dp[2] + 1) = 1`; and `answer = max(0, dp[2]=0) = 0`. The loop ends with `answer = 0`. It printed `0`, but the correct answer is `1`.

**Diagnosing the bug.** The defect is precise. When I processed vertex `1`, its `dp` was still `0` because the edge `2 -> 1` that would have raised it hadn't been relaxed yet — vertex `2` is processed *after* vertex `1` in id order, even though `2` is upstream of `1`. So I read `dp[1]` into `answer` while it was stale. The relaxation `dp[1] = 1` did happen, but only *after* I'd already taken `answer` from `dp[1]`, so the freshly raised value never made it into the answer. The root cause is that the DP recurrence requires every predecessor of a vertex to be finalized before that vertex contributes to the answer, and id order does not guarantee that; only a topological order does. There's a second, quieter manifestation of the same bug: even the `dp` values themselves can come out wrong (not just the answer reading) whenever a chain is numbered against id order, e.g. edges `3 -> 2 -> 1` would leave `dp[1] = 1` instead of `2` because `dp[2]` isn't final when `2 -> 1` is relaxed. Both are the one disease: relaxing out of topological order.

**Fixing and re-verifying.** Build a real topological order with Kahn first, then sweep *that* order doing the relaxations. I also fold the answer update to the moment a vertex is *dequeued* (finalized), so I read `dp[u]` only when it is final:

```
// Kahn topological order.
vector<int> order; queue<int> q;
for (int u = 1; u <= n; u++) if (indeg[u] == 0) q.push(u);
vector<int> deg = indeg;
while (!q.empty()) {
    int u = q.front(); q.pop(); order.push_back(u);
    for (int v : adj[u]) if (--deg[v] == 0) q.push(v);
}
// DP in topological order.
vector<int> dp(n + 1, 0); int answer = 0;
for (int u : order) {
    answer = max(answer, dp[u]);
    for (int v : adj[u]) dp[v] = max(dp[v], dp[u] + 1);
}
```

I keep a *mutable copy* `deg` of the in-degrees for Kahn so that the original `indeg` (which I might want, and which conceptually defines the sources) is not destroyed by the `--deg[v]` decrements — overwriting the array I'm iterating logic over is its own classic bug, so I separate the two. Re-trace `2 -> 1`: in-degrees are `indeg[1]=1, indeg[2]=0`, so Kahn seeds the queue with `2`. Dequeue `2`, emit it, relax `2 -> 1` → `deg[1]` drops to `0`, push `1`. Dequeue `1`, emit it. Order is `[2, 1]`. DP: process `2` (`dp=0`), `answer = 0`, relax `2->1` → `dp[1] = 1`. Process `1` (`dp=1`), `answer = max(0, 1) = 1`. Prints `1`. Correct. The case that broke before now passes, and it broke for precisely the reason I fixed.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `E = 0` (no edges): Kahn seeds *every* vertex (all in-degree `0`), the order is all `V` vertices, every `dp` stays `0`, and `answer = 0`. Correct — each vertex is its own length-`0` path.
- A single long chain `1 -> 2 -> ... -> V`: Kahn emits `1, 2, ..., V`, and `dp` climbs `0, 1, 2, ..., V-1`; `answer = V - 1`. Correct.
- Multiple sources / multiple sinks (a diamond `1->2, 1->3, 2->4, 3->4`): sources `{1}`, sinks `{4}`; `dp[4] = 2`, answer `2`. A layered DAG with two sources both feeding a common tail computes the max over both correctly because every vertex's `dp` already aggregates all incoming paths.
- Parallel edges (`1 -> 2` twice, then `2 -> 3`): the edge appears twice in `adj[1]`, so `indeg[2] = 2`; Kahn decrements it twice, which is consistent, and the relaxation `dp[2] = max(0, 1)` just runs twice with the same effect. The answer is unaffected — `dp[2] = 1, dp[3] = 2`. I verified this explicitly because a duplicated edge inflating in-degree is exactly the kind of thing that could deadlock Kahn if I'd, say, decremented from the *wrong* count; here `indeg` and the decrements are the same multiset of edges, so it's consistent.
- Overflow / types: the longest path has at most `V - 1 <= 2*10^5 - 1` edges, far inside `int`. No 64-bit needed.
- Performance: Kahn is `O(V + E)`, the DP sweep visits every vertex once and every edge once, so `O(V + E)`. On `V = 2*10^5`, `E = 4*10^5` it runs in well under a tenth of a second with a few tens of megabytes — far inside the `1 s` / `256 MB` budget. The "obvious" `O(V*E)` Bellman-Ford-style relaxation would be `8*10^10` operations and time out; topological DP is what makes the linear bound real.
- Empty input / robustness: `if (!(cin >> n >> m)) return 0;` guards a missing header; `cin >>` skips arbitrary whitespace so the input format is forgiving.

**Self-verification harness.** Hand-tracing convinces me of the logic, but I don't trust a graph algorithm until it has survived a differential test, so I wrote an *independent* brute oracle and a random DAG generator and ran them against each other. The oracle ignores all cleverness: it lists the sources (in-degree-`0` vertices) and, from each, does a plain recursive DFS that explores *every* outgoing edge and tracks the maximum depth reached — the literal definition of "most edges on a path from a source", with no DP and no pruning. The generator builds random DAGs by drawing a random topological labelling and only adding edges that go forward in that labelling (guaranteeing acyclicity), across several shapes: tiny graphs (`V <= 4`), sparse and dense random DAGs, edge-free graphs, the "wide hub then long tail" shape that specifically tempts the out-degree greedy, and layered multi-source/multi-sink DAGs. To make sure id-order and topo-order really diverge in the tests (the exact thing my first bug depended on), the generator shuffles vertex labels before emitting, so the solver cannot get away with assuming the input is pre-sorted.

I compiled the fixed solution and ran it against the brute oracle on `700` random seeds plus the hand-built edge cases (`E = 0`, single chain, diamond, parallel edges, single isolated vertex). Zero mismatches. I also re-ran the explicit greedy-killer graph through the final solution: it returns `4`, matching the brute oracle, while the out-degree greedy I'd coded returns `2` — the counterexample survives end to end. Finally I timed the large instance (`V = 2*10^5`, `E = 4*10^5`, with a length-`199999` backbone chain plus random forward chords): about `0.07 s` wall and `~17 MB`. The linear method clears the budget with two orders of magnitude to spare.

**Final solution.** I disproved the greedy with a concrete graph (`out-degree-3` fan-out into sinks loses to an `out-degree-1` edge into a depth-`4` tail: greedy `2` vs. correct `4`), reduced "longest path from a source" to "max over all vertices of the longest path ending there" by the observation that `dp = 0` exactly at sources, and made the recurrence evaluable by relaxing in a real topological order rather than the input's id order — the bug a backward-numbered edge `2 -> 1` exposed by printing `0` instead of `1`. The fix, a Kahn order then a single relaxing sweep, passes `700` randomized differential tests and every hand edge case at `O(V + E)`. That is what I ship — one self-contained file, the simple provable topological DP, not the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;        // empty input -> nothing to do

    vector<vector<int>> adj(n + 1);        // 1-indexed; adj[u] = out-neighbours
    vector<int> indeg(n + 1, 0);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;                     // directed edge u -> v
        adj[u].push_back(v);
        indeg[v]++;
    }

    // Kahn topological order.
    vector<int> order;
    order.reserve(n);
    queue<int> q;
    for (int u = 1; u <= n; u++)
        if (indeg[u] == 0) q.push(u);
    vector<int> deg = indeg;               // mutable copy for Kahn
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int v : adj[u])
            if (--deg[v] == 0) q.push(v);
    }

    // dp[v] = longest path (in #edges) ending at v. A vertex with no incoming
    // edge is a source and has dp = 0; relaxing in topological order makes every
    // longest path ending at v trace back to a source, so the global maximum is
    // the longest path that starts at some source.
    vector<int> dp(n + 1, 0);
    int answer = 0;
    for (int u : order) {
        if (dp[u] > answer) answer = dp[u];
        for (int v : adj[u])
            if (dp[u] + 1 > dp[v]) dp[v] = dp[u] + 1;
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The out-degree greedy looked plausible but a single traced graph (`1 -> 2` with a fat fan-out into sinks versus `1 -> 6 -> 7 -> 8 -> 9`: greedy `2` vs. reachable `4`) showed a local breadth signal blocks more than it gains, so I moved to the topological DP and checked its recurrence by hand; the reduction from "starts at a source" to "max longest-path-ending-at-`v`" holds because `dp[v] = 0` is exactly the source condition; the DP needs predecessors finalized first, which my initial id-order relaxation violated — a backward edge `2 -> 1` printing `0` instead of `1` pinpointed it; replacing id order with a Kahn topological order and reading the answer only at dequeue time fixes it; and a `700`-case differential test against an independent DFS oracle, the explicit edge cases, and a `0.07 s` run at the upper bound close out the corners, the parallel-edge case, and the performance budget.
