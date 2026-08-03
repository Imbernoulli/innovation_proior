// TIER: greedy
// Weight-descending greedy: take stations from highest value to lowest, activating each
// one that does not conflict with the already-activated set. One pass, no restarts.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    vector<long long> w(N + 1);
    for (int j = 1; j <= N; j++) scanf("%lld", &w[j]);
    vector<vector<int>> adj(N + 1);
    for (int i = 0; i < M; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> order(N);
    for (int j = 0; j < N; j++) order[j] = j + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (w[a] != w[b]) return w[a] > w[b];
        return a < b;
    });

    vector<char> sel(N + 1, 0);
    vector<int> chosen;
    for (int j : order) {
        bool ok = true;
        for (int nb : adj[j]) if (sel[nb]) { ok = false; break; }
        if (ok) { sel[j] = 1; chosen.push_back(j); }
    }

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    if (chosen.empty()) printf("\n");
    return 0;
}
