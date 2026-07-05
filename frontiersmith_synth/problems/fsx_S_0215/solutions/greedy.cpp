// TIER: greedy
// Weight-greedy: consider individuals in descending surveillance value; add each
// one whose contacts are all still unmonitored.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v); adj[v].push_back(u);
    }
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){ return w[a] > w[b]; });
    vector<char> inSet(n + 1, 0), blocked(n + 1, 0);
    vector<int> pick;
    for (int u : order) {
        if (blocked[u]) continue;
        inSet[u] = 1; pick.push_back(u);
        for (int v : adj[u]) blocked[v] = 1;
    }
    printf("%d\n", (int)pick.size());
    for (int u : pick) printf("%d\n", u);
    return 0;
}
