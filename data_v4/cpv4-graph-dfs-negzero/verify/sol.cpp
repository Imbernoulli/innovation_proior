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
