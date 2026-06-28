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
