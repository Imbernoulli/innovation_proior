**Problem.** A cave has `n` chambers `0..n-1` (entrance is `0`) and `m` one-way tunnels, each `a -> b` with `a < b` (so the network is a DAG that only goes deeper). Chamber `i` has value `v[i]`, which may be positive, zero, or negative. An explorer enters at `0`, walks along tunnels, collects the value of every chamber it stands in, and may stop at any chamber. `v[0]` is always collected (you cannot refuse to enter), and the explorer may stop at any point. Print the maximum total value collectible — which may be **negative**.

**Why the obvious greedy is wrong.** "Step into the best-looking child while it improves the total, else stop" fails because a negative toll chamber can guard a much larger treasure. On the chain `0 -> 1 -> 2` with `v = [1, -5, 100]`, greedy refuses to step onto the `-5` and stops at `0` with total `1`, but `0 -> 1 -> 2` collects `1 - 5 + 100 = 96`. A one-step-lookahead rule cannot pay a toll for a far-off payoff. Greedy is discarded.

**Key idea — memoized DFS / DAG DP.** Define `best[u]` = the most value collectible on a descent that *starts* at `u`. Standing at `u` collects `v[u]` unconditionally; then either stop (add nothing) or descend into the single best child:

- `best[u] = v[u] + max(0, max over children c of best[c])`

The inner `max(0, ...)` is the *stop* option (descend into no one, add `0`). The `v[u] +` sits *outside* that max: you must stand on `u`, so `v[u]` is never clamped. The answer is `best[0]` with **no** outer clamp (you cannot refuse the entrance).

Because every tunnel goes `a < b`, all children of `u` have larger index, so filling `best` in **decreasing index order** is a valid reverse topological order — one `O(n + m)` pass, no recursion (a depth-`2*10^5` chain would overflow a recursive stack), no separate topological sort.

**Correctness.** By induction on decreasing index: when `u` is processed, every child (index `> u`) already holds its correct `best`. Any descent from `u` either stops at `u` (value `v[u]`, captured by the `0` in the max) or steps to some child `c` and continues optimally (value `v[u] + best[c]`); the recurrence takes the max over both, so `best[u]` is optimal. Hence `best[0]` is the optimum over all descents from the entrance. An independent exhaustive checker (enumerate every walk from `0`, take the max running sum over all prefixes) agrees on small cases.

**Pitfalls.**
1. *Clamp the descent, not the node.* Writing `best[u] = max(0, v[u] + descend)` clamps the mandatory entrance and turns a legitimately negative answer into a phantom `0` (e.g. `n = 1`, `v = [-7]` should give `-7`). It also corrupts internal values, advertising `0` to parents for nodes whose true best is negative. Clamp only the descent (`max(0, descend)`); leave `v[u]` and the final `best[0]` unclamped.
2. *Keep the stop option.* Dropping the `0` from the inner max (`best[u] = v[u] + max over children`) forces the explorer to plunge into every reachable trap; on `v = [5, -1, -100]` it returns `-96` instead of the correct `5`. The `0`-start of the descent is load-bearing.
3. *Overflow.* With `n` up to `2*10^5` and `|v[i]|` up to `10^9`, a path sum reaches `~2*10^14`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 1` with `v[0]` negative -> that negative value (not `0`); all-negative cave -> the least-bad reachable prefix (negative, e.g. `[-1,-2,-3]` chain -> `-1`, stop at entrance); a node whose every descendant is negative -> stop early; `m = 0` -> only the entrance is reachable, answer `v[0]`; negative entrance redeemed by a deep treasure -> the DP pays the toll. All handled by clamping the descent at `0` while leaving `v[u]` and `best[0]` unclamped.

**Complexity.** `O(n + m)` time, `O(n + m)` space (the adjacency lists and `best` array).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;
    vector<vector<int>> adj(n);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        adj[a].push_back(b);   // directed tunnel a -> b, guaranteed a < b (DAG, deeper)
    }

    // best[u] = max value collectible on a walk that STARTS at u, following edges,
    //           and may STOP at any chamber. You always collect v[u] (you are there),
    //           and you MAY descend into the single best child if that helps, else stop.
    //   best[u] = v[u] + max(0, max over children c of best[c])
    // Memoized DFS over the DAG.
    vector<long long> best(n, LLONG_MIN);
    // Iterative post-order via explicit recursion replacement (n up to 2e5, avoid stack overflow).
    // Since edges go a < b, processing nodes in decreasing index order gives a valid reverse
    // topological order: all children of u have index > u, so they are computed first.
    for (int u = n - 1; u >= 0; u--) {
        long long descend = 0;                 // option to STOP at u contributes 0 extra
        for (int c : adj[u]) descend = max(descend, best[c]);
        best[u] = v[u] + descend;              // v[u] is NOT clamped: you must stand on u
    }

    cout << best[0] << "\n";                    // start is chamber 0; answer may be negative
    return 0;
}
```
