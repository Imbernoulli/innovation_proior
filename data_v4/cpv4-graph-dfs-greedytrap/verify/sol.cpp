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
