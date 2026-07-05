// TIER: trivial
// Index-order greedy = exactly the checker's internal baseline B -> ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<char> in(n + 1, 0);
    vector<int> pick;
    for (int u = 1; u <= n; u++) {
        bool ok = true;
        for (int v : adj[u]) if (in[v]) { ok = false; break; }
        if (ok) { in[u] = 1; pick.push_back(u); }
    }
    printf("%d\n", (int)pick.size());
    for (int u : pick) printf("%d\n", u);
    return 0;
}
