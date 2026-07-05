// TIER: greedy
// Yield-greedy: consider plots in descending yield; add each if it stays runoff-free and within budget.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m; long long W;
    if (scanf("%d %d %lld", &n, &m, &W) != 3) return 0;
    vector<long long> w(n + 1), d(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    for (int i = 1; i <= n; i++) scanf("%lld", &d[i]);
    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (w[a] != w[b]) return w[a] > w[b];
        return a < b;
    });
    vector<char> chosen(n + 1, 0);
    long long used = 0;
    for (int i : order) {
        if (used + d[i] > W) continue;
        bool ok = true;
        for (int j : adj[i]) if (chosen[j]) { ok = false; break; }
        if (!ok) continue;
        chosen[i] = 1;
        used += d[i];
    }
    vector<int> res;
    for (int i = 1; i <= n; i++) if (chosen[i]) res.push_back(i);
    printf("%d\n", (int)res.size());
    for (int i : res) printf("%d\n", i);
    return 0;
}
