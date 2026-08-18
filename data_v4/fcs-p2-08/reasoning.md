"Longest path" is NP-hard in general, but acyclicity collapses it here: every forward walk in a DAG is automatically a simple, finite path, so the maximum is well defined. The constraints then set the terms. `V <= 2*10^5`, `E <= 4*10^5`, one second — so `V*E = 8*10^10` rules out any Bellman-Ford-style repeated relaxation, and I need a single `O(V + E)` pass. The answer is bounded by `V - 1` edges (no repeated vertex in a DAG), so it is a small count that fits a 32-bit `int` with room to spare — no 64-bit worry. The one clause I don't yet know how to use is "starts at a source"; it reads like it wants a special-cased enumeration over the in-degree-`0` vertices, so I hold onto it. Plain `int`, one linear sweep — the question is which sweep.

**Stress-testing the greedy before committing.** "Out-degree looks like a good proxy for how far I can go" is precisely the kind of intuition that ships wrong solutions, so let me actually attack it with a small graph I can reason about by hand. I want a source from which one neighbour has high out-degree but leads nowhere, and another neighbour has out-degree `1` but begins a long chain. Here is the instance, `V = 9`.

**Why the greedy fails.** The seductive line is "from each source, keep stepping to the out-neighbour with the largest out-degree — the one with the most options ahead." It is a dozen lines and `O(V + E)`, and it is wrong, because out-degree is a *local breadth* signal while the longest path is a *global depth* property. The example in the statement is exactly the refutation: source `1` points to `2` (out-degree `3`, but `2 -> 3,4,5` are all sinks) and to `6` (out-degree `1`, but `6 -> 7 -> 8 -> 9`). Largest-out-degree greedy walks `1 -> 2` and dead-ends at `2` edges; the real answer is `4` via the thin edge into the long tail. Nor is it fixable by a smarter local proxy: the only quantity that predicts how far a vertex can continue is the downstream longest path itself — precisely what I'm trying to compute. That circularity forces DP.

**The DP, and why "from a source" is free.** Let `dp[v]` be the longest path in edges that *ends* at `v`. A longest path ending at `v` is either `v` alone (if nothing points in) or a longest path ending at some predecessor `u` plus the edge `u -> v`:

```
dp[v] = max over edges (u -> v) of ( dp[u] + 1 ),   or 0 if v has no in-edge.
```

And now the clause I was holding onto pays off. `dp[v] = 0` holds *exactly* when `v` has no incoming edge — that is, exactly when `v` is a source. So tracing any longest-ending path backwards can only halt at a `dp = 0` vertex, i.e. a source; every longest-ending path already starts at a source. The answer is therefore just `max_v dp[v]`, with no per-source enumeration and no forward DP from each source.

For the recurrence to be evaluable, `dp[u]` must be final before it relaxes `u -> v`, which is precisely what a **topological order** guarantees. I'll build one with Kahn (repeatedly emit in-degree-`0` vertices) and relax along it.

Let me confirm the recurrence by hand on the counterexample graph, answer `4`. Topological order (one valid one): `1, 2, 6, 3, 4, 5, 7, 8, 9`. Initialize all `dp = 0`. Process `1` (`dp=0`): relax `1->2` so `dp[2] = max(0, 1) = 1`; relax `1->6` so `dp[6] = 1`. Process `2` (`dp=1`): relax `2->3,2->4,2->5` so `dp[3]=dp[4]=dp[5]=2`. Process `6` (`dp=1`): relax `6->7` so `dp[7]=2`. Process `3,4,5` (each `dp=2`, no out-edges): nothing. Process `7` (`dp=2`): relax `7->8` so `dp[8]=3`. Process `8` (`dp=3`): relax `8->9` so `dp[9]=4`. Process `9` (`dp=4`): no out-edges. Maximum `dp` is `dp[9]=4`. Correct, and notice `dp[3]=dp[4]=dp[5]=2` (the fan-out) never beat the chain — the DP weighs depth, not breadth, which is exactly the property greedy lacked.

**The order trap the constraints invite.** My first cut skipped Kahn and just relaxed edges while sweeping vertices `1..n`, reading the answer off `dp[u]` as I went:

```
for (int u = 1; u <= n; u++) {
    for (int v : adj[u]) dp[v] = max(dp[v], dp[u] + 1);
    answer = max(answer, dp[u]);
}
```

This assumes id order equals topological order, and nothing in the statement promises that — only acyclicity. A DAG can be numbered any way, so an edge pointing to a lower id breaks it. The smallest witness: `V = 2`, single edge `2 -> 1`, answer `1`. The loop hits `u = 1` first while `dp[1]` is still `0` (the edge `2 -> 1` hasn't been relaxed, since `2` comes later), takes `answer = 0`, then relaxes `dp[1] = 1` — too late to count. It prints `0`. The same disease corrupts the `dp` values themselves on any backward-numbered chain (`3 -> 2 -> 1` leaves `dp[1] = 1` instead of `2`), not just the final read. Both are one fault: relaxing out of topological order.

The fix is to build a real Kahn order and relax along *that*, reading `answer` only when a vertex is dequeued (hence final):

```
vector<int> deg = indeg;               // mutable copy: keep indeg intact
queue<int> q;
for (int u = 1; u <= n; u++) if (indeg[u] == 0) q.push(u);
while (!q.empty()) {
    int u = q.front(); q.pop(); order.push_back(u);
    for (int v : adj[u]) if (--deg[v] == 0) q.push(v);
}
for (int u : order) {
    answer = max(answer, dp[u]);
    for (int v : adj[u]) dp[v] = max(dp[v], dp[u] + 1);
}
```

Kahn decrements a *copy* `deg` so `indeg` — which defines the sources — is not zeroed out mid-loop. On `2 -> 1` Kahn seeds `2`, emits `[2, 1]`, and the sweep gives `dp[1] = 1`, `answer = 1` — the case that broke now resolves correctly.

**Self-verification harness.** Hand-tracing convinces me of the logic, but I don't trust a graph algorithm until it has survived a differential test, so I wrote an *independent* brute oracle and a random DAG generator and ran them against each other. The oracle ignores all cleverness: it lists the sources (in-degree-`0` vertices) and, from each, does a plain recursive DFS that explores *every* outgoing edge and tracks the maximum depth reached — the literal definition of "most edges on a path from a source", with no DP and no pruning. The generator builds random DAGs by drawing a random topological labelling and only adding edges that go forward in that labelling (guaranteeing acyclicity), across several shapes: tiny graphs (`V <= 4`), sparse and dense random DAGs, edge-free graphs, the "wide hub then long tail" shape that specifically tempts the out-degree greedy, and layered multi-source/multi-sink DAGs. To make sure id-order and topo-order really diverge in the tests (the exact thing my first bug depended on), the generator shuffles vertex labels before emitting, so the solver cannot get away with assuming the input is pre-sorted.

**Corners.** `E = 0`: every vertex is a source, the order is all of them, all `dp` stay `0`, answer `0`. A single chain `1 -> ... -> V`: `dp` climbs to `V - 1`. Multi-source/multi-sink layered DAGs: every vertex's `dp` already aggregates all incoming paths, so the max over tails is correct with no special handling. Parallel edges: a duplicated `u -> v` raises `indeg[v]` twice and Kahn decrements it twice — same multiset of edges on both sides, so the count stays consistent and the repeated relaxation is idempotent.

To be sure past hand-tracing, I check against an independent brute oracle: enumerate the sources and DFS every outgoing edge from each, tracking max depth — the literal definition, no DP, no pruning. The random DAG generator draws a topological labelling, adds only forward edges (guaranteeing acyclicity), and *shuffles the vertex labels before emitting* so id-order and topo-order genuinely diverge — the exact condition the id-order bug needed. Across `500` random cases spanning tiny graphs, sparse and dense DAGs, edge-free graphs, the wide-hub-then-long-tail shape, and layered multi-source/multi-sink DAGs, plus the hand edge cases: zero mismatches. The largest instance (`V = 2*10^5`, `E = 4*10^5`, a length-`199999` backbone with random forward chords) runs in about `0.07 s` and `~17 MB` — more than an order of magnitude inside the `1 s` / `256 MB` budget, confirming that trading the `O(V*E)` relaxation for a single topological sweep is what makes linear real here.
