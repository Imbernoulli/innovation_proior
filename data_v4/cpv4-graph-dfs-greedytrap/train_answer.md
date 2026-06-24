**Problem.** Given a DAG of `n` papers with integer prestige `p[i]` (may be negative) and `m` directed citation links `u -> v`, pick a start paper and follow links one at a time, stopping whenever you like, to form a directed path. The score is the sum of `p[]` over the path's nodes (each once); you must read at least one paper. Read `n`, `m`, the prestige array, and the `m` links from stdin; print the maximum score.

**Why the obvious greedy is wrong.** "From the current paper always hop to the reachable neighbour with the largest prestige `p[v]`" fails because the path constraint is global: what matters about a neighbour is not its own prestige but `best[neighbour]`, the value of the best path it leads into. On nodes `0,1,2,3` with `p = [1, 10, 1, 100]` and links `0->1, 0->2, 2->3`, greedy leaves `0` for paper `1` (prestige `10 > 1`) and dead-ends at `1 + 10 = 11`; the optimal play `0 -> 2 -> 3` scores `1 + 1 + 100 = 102`. Greedy's best over all starts is `101`, the truth is `102`. Greedy is discarded.

**Key idea — DFS-based DP on the DAG.** Let `best[u]` be the maximum score of a path that *starts* at `u`. You must read `u`, then either stop or move to exactly one out-neighbour and continue optimally:

- `best[u] = p[u] + max(0, max over edges u->v of best[v])`.

The inner `max(0, ...)` is the "stop at `u`" option; it declines any continuation whose `best[v]` is negative. Because the graph is acyclic, `best[u]` depends only on `best[v]` of out-neighbours, which a depth-first post-order traversal finalizes first. The answer is `max over all u of best[u]`, with the outer maximum seeded at `-infinity` (not `0`).

**Correctness.** Any path from `u` either ends at `u` (score `p[u]`) or takes one first edge `u -> v` and is then an optimal path from `v` (score `p[u] + best[v]`); maximizing over "stop" and all first edges gives the recurrence, and acyclicity makes the recursion well-founded. Taking the max of `best[u]` over all `u` covers every possible start.

**Pitfalls to get right.**
1. *The greedy trap.* Compare `best[v]`, never `p[v]`, when choosing the next hop; a modest paper can lead to the richest continuation.
2. *Empty path is illegal.* You must read at least one paper, so the *outer* answer is seeded at `LLONG_MIN`, not `0` — otherwise an all-negative instance wrongly returns `0` instead of its largest (negative) prestige. (A trace of `p=[-5]` returning `0` exposes this.) The *inner* `max(0,...)` is fine: it only declines to extend, after `p[u]` is already paid.
3. *Recursion depth.* A natural recursive DFS overflows the call stack on an induced chain of `2*10^5` nodes (~12.8 MB of frames). Use an explicit-stack post-order DFS.
4. *Overflow.* With `n` up to `2*10^5` and `|p[i]|` up to `10^9`, scores reach `~2*10^14`; use `long long`. An `int` is a silent wrong-answer on long chains.

**Edge cases.** `n = 1` positive -> that value; `n = 1` negative -> that (negative) value; all-negative -> the single largest prestige; `m = 0` -> the largest single prestige; a node worth passing through vs. avoiding is handled automatically by the inner `max(0,...)`. The `done`-guarded pushes also ensure each node is finalized exactly once, even when reachable from several parents (diamonds).

**Complexity.** `O(n + m)` time, `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    vector<vector<int>> adj(n);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    // best[u] = maximum total prestige of a directed path that STARTS at u.
    // You must read u (so p[u] is always counted), then optionally extend to
    // exactly one out-neighbour v (counting all of best[v]) or stop at u.
    //   best[u] = p[u] + max(0, max over edges u->v of best[v])
    // Computed by memoized DFS on the DAG (no cycles, so no in-progress guard needed
    // for correctness, but we keep a visited/state array to memoize).
    vector<long long> best(n);
    vector<char> done(n, 0);

    // Iterative DFS to avoid stack overflow at n = 2*10^5.
    vector<int> stk;
    stk.reserve(n);
    vector<int> it(n, 0); // edge iterator per node

    for (int s = 0; s < n; s++) {
        if (done[s]) continue;
        stk.push_back(s);
        while (!stk.empty()) {
            int u = stk.back();
            if (it[u] < (int)adj[u].size()) {
                int v = adj[u][it[u]++];
                if (!done[v]) stk.push_back(v);
            } else {
                long long ext = 0; // option: stop at u (extend nothing)
                for (int v : adj[u]) ext = max(ext, best[v]);
                best[u] = p[u] + ext;
                done[u] = 1;
                stk.pop_back();
            }
        }
    }

    long long ans = LLONG_MIN;
    for (int u = 0; u < n; u++) ans = max(ans, best[u]);
    cout << ans << "\n";
    return 0;
}
```
