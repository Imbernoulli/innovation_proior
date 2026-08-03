// TIER: greedy
// Weight-greedy: sort performers by hype descending, feature each one whose addition
// keeps the roster rivalry-free. One pass, maximal by weight.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> order(n);
    iota(order.begin(), order.end(), 1);
    sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });

    vector<char> chosen(n + 1, 0), blocked(n + 1, 0);
    vector<int> sel;
    for (int u : order) {
        if (blocked[u]) continue;
        chosen[u] = 1;
        sel.push_back(u);
        for (int v : adj[u]) blocked[v] = 1;
    }

    printf("%d\n", (int)sel.size());
    for (int u : sel) printf("%d\n", u);
    return 0;
}
