// TIER: greedy
// Weight-descending greedy: consider pitches from highest to lowest value; grant each
// one that conflicts with nothing already granted. One pass, no restarts.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N; long long M;
    if (scanf("%d %lld", &N, &M) != 2) return 0;
    vector<long long> w(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);

    vector<vector<int>> adj(N + 1);
    for (long long i = 0; i < M; i++) {
        int a, b; scanf("%d %d", &a, &b);
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int x, int y){
        if (w[x] != w[y]) return w[x] > w[y];
        return x < y;
    });

    vector<char> blocked(N + 1, 0), sel(N + 1, 0);
    vector<int> chosen;
    for (int v : order) {
        if (blocked[v]) continue;
        sel[v] = 1;
        chosen.push_back(v);
        for (int u : adj[v]) blocked[u] = 1;
    }

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    if (chosen.empty()) printf("\n");
    return 0;
}
