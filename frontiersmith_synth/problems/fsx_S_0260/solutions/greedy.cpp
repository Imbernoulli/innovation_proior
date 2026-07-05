// TIER: greedy
// One-pass greedy: place each wagon in the bowl that maximizes marginal cut
// against already-placed neighbors, capping bowl sizes to keep balance.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, tol;
    scanf("%d %d %d", &n, &m, &tol);
    vector<vector<pair<int,int>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    // max size of either bowl so that final |c0-c1| <= tol
    int maxSide = (n + tol) / 2;

    vector<int> side(n + 1, -1);
    int c0 = 0, c1 = 0;
    for (int u = 1; u <= n; u++) {
        ll w0 = 0, w1 = 0; // weight to already-placed neighbors on each side
        for (auto& e : adj[u]) {
            if (side[e.first] == 0) w0 += e.second;
            else if (side[e.first] == 1) w1 += e.second;
        }
        // placing u in bowl 0 cuts edges to side-1 neighbors (gain w1); vice versa.
        int choose;
        if (c0 >= maxSide) choose = 1;
        else if (c1 >= maxSide) choose = 0;
        else choose = (w1 >= w0) ? 0 : 1;
        side[u] = choose;
        if (choose == 0) c0++; else c1++;
    }
    for (int i = 1; i <= n; i++) printf("%d%c", side[i], i == n ? '\n' : ' ');
    return 0;
}
