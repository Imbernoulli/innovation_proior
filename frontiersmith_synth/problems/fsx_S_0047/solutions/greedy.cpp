// TIER: greedy
// Weight-greedy: sort sites by value descending and add each one whose conflicts are
// all still free. One pass, ignores conflict density -> leaves value on the table.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });
    vector<char> sel(n + 1, 0), blocked(n + 1, 0);
    vector<int> chosen;
    for (int v : order) {
        if (blocked[v]) continue;
        sel[v] = 1;
        chosen.push_back(v);
        for (int u : adj[v]) blocked[u] = 1;
    }
    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], (i + 1 < chosen.size()) ? ' ' : '\n');
    if (chosen.empty()) printf("\n");
    return 0;
}
