// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    // scan sites in decreasing value; equip one iff no equipped site conflicts.
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });
    vector<char> blocked(n + 1, 0);
    vector<int> sel;
    for (int v : order) {
        if (blocked[v]) continue;
        sel.push_back(v);
        for (int u : adj[v]) blocked[u] = 1;
    }
    printf("%d\n", (int)sel.size());
    for (int v : sel) printf("%d\n", v);
    return 0;
}
