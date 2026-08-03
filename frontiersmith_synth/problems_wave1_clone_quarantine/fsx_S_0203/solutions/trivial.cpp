// TIER: trivial
// Reproduces the checker's index-order greedy reference slate -> scores ~0.1.
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
    vector<char> blocked(n + 1, 0), sel(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        if (blocked[i]) continue;
        sel[i] = 1;
        blocked[i] = 1;
        for (int nb : adj[i]) blocked[nb] = 1;
    }
    vector<int> out;
    for (int i = 1; i <= n; i++) if (sel[i]) out.push_back(i);
    printf("%d\n", (int)out.size());
    for (size_t i = 0; i < out.size(); i++)
        printf("%d%c", out[i], i + 1 == out.size() ? '\n' : ' ');
    if (out.empty()) printf("\n");
    return 0;
}
