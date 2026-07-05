// TIER: greedy
// Weight-greedy: sort sites by data value descending, add each if it conflicts
// with nothing already chosen. Tempted by high-value (but high-conflict) hubs.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
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

    vector<char> chosen(n + 1, 0), blocked(n + 1, 0);
    vector<int> sel;
    for (int v : order) {
        if (blocked[v]) continue;
        chosen[v] = 1;
        sel.push_back(v);
        for (int u : adj[v]) blocked[u] = 1;
    }

    printf("%d\n", (int)sel.size());
    for (size_t i = 0; i < sel.size(); i++)
        printf("%d%c", sel[i], i + 1 == sel.size() ? '\n' : ' ');
    if (sel.empty()) printf("\n");
    return 0;
}
