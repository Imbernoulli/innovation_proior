// TIER: greedy
// One-pass profit-descending greedy independent set: consider routes from most to
// least profitable, take a route if it does not conflict with anything already taken.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<ll> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        g[u].push_back(v); g[v].push_back(u);
    }
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){ return w[a] > w[b]; });

    vector<char> chosen(n + 1, 0), blocked(n + 1, 0);
    vector<int> sel;
    for (int v : order) {
        if (blocked[v]) continue;
        chosen[v] = 1; sel.push_back(v);
        for (int u : g[v]) blocked[u] = 1;
    }
    printf("%d\n", (int)sel.size());
    for (int v : sel) printf("%d\n", v);
    return 0;
}
