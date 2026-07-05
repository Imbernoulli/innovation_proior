// TIER: greedy
// One-pass constructive: process modules in index order, assign each to the cryostat that
// maximizes earned value against already-placed neighbors, respecting the n/2 capacity.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<vector<array<int,3>>> adj(n + 1); // (nbr, type, w)
    for (int e = 0; e < m; e++) {
        int u, v, t, w; scanf("%d %d %d %d", &u, &v, &t, &w);
        adj[u].push_back({v, t, w});
        adj[v].push_back({u, t, w});
    }
    vector<int> side(n + 1, -1);
    int cnt0 = 0, cnt1 = 0, half = n / 2;
    for (int i = 1; i <= n; i++) {
        long long g0 = 0, g1 = 0;
        for (auto &e : adj[i]) {
            int y = e[0]; if (side[y] < 0) continue;
            int t = e[1], w = e[2];
            // if i -> 0
            bool cut0 = (0 != side[y]); if ((t == 0) ? cut0 : !cut0) g0 += w;
            // if i -> 1
            bool cut1 = (1 != side[y]); if ((t == 0) ? cut1 : !cut1) g1 += w;
        }
        int choose;
        if (cnt0 >= half) choose = 1;
        else if (cnt1 >= half) choose = 0;
        else choose = (g1 > g0) ? 1 : 0;
        side[i] = choose;
        if (choose == 0) cnt0++; else cnt1++;
    }
    for (int i = 1; i <= n; i++) printf("%d%c", side[i], i == n ? '\n' : ' ');
    return 0;
}
