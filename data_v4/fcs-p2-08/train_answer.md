**Problem.** Given a DAG with `V` vertices and `E` directed edges, report the maximum number of edges
on any directed path that starts at a *source* (an in-degree-`0` vertex). An isolated vertex is a
length-`0` path, so the answer is at least `0`. Read `V E` then the `E` edges `u v` from stdin, print
the maximum length.

**Why the tempting greedy is wrong.** "From each source, repeatedly step to the out-neighbour with the
largest out-degree (the most options ahead)" fails because out-degree is a *local breadth* signal
while the longest path is a *global depth* property. Concrete counterexample (`V = 9`): from source
`1`, neighbour `2` has out-degree `3` (`2 -> 3, 4, 5`, all sinks) and neighbour `6` has out-degree `1`
but begins the chain `6 -> 7 -> 8 -> 9`. The greedy walks `1 -> 2` and dead-ends after `2` edges; the
true answer is `4` via `1 -> 6 -> 7 -> 8 -> 9`. A fat fan-out into sinks beats a thin edge into a long
tail on the local signal and loses globally. The only quantity that predicts how far a vertex can
continue is the downstream longest path itself — the thing we are computing — so the heuristic is
discarded.

**Key idea — topological-order DP.** Let `dp[v]` be the longest path (in edges) that *ends* at `v`:

- `dp[v] = max over edges (u -> v) of (dp[u] + 1)`, and `dp[v] = 0` if `v` has no incoming edge.

Crucially, `dp[v] = 0` holds *exactly* when `v` is a source, so tracing any longest-ending path
backwards must terminate at a source. Hence the longest path that starts at a source equals
`max_v dp[v]`, with no per-source enumeration needed. Evaluate the recurrence by relaxing edges in a
**topological order** (built with Kahn's algorithm), so every predecessor is finalized before it
contributes.

**Two pitfalls to get right.**
1. *Relaxation order.* Relax in topological order, not vertex-id order. The input is only guaranteed
   acyclic, not numbered low-to-high, so an edge like `2 -> 1` makes an id-order sweep read a stale
   `dp` and report `0` instead of `1`. Kahn's order fixes this; reading the answer only when a vertex
   is dequeued (finalized) keeps it honest.
2. *Mutable degree copy.* Decrement a copy of the in-degree array inside Kahn so the original
   in-degree information (which defines the sources) is not destroyed mid-loop.

**Edge cases (all handled by Kahn + the single sweep):** `E = 0` -> every vertex a source, answer `0`;
a single chain -> `V - 1`; multi-source/multi-sink layered DAGs -> max over all tails; parallel edges
-> in-degree counts each copy and Kahn decrements each, consistently, answer unaffected.

**Complexity.** `O(V + E)` time and space; the path length is at most `V - 1`, so a 32-bit `int`
suffices. At `V = 2*10^5`, `E = 4*10^5` it runs in well under `0.1 s` — far inside the `1 s` / `256 MB`
budget, whereas an `O(V*E)` relaxation (`~8*10^10` operations) would time out.

**Code.**

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
