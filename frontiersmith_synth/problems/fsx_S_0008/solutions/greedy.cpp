// TIER: greedy
// Greedy constructive: process sensors 1..n in order; assign each to the station
// that maximizes cross-checked weight to already-placed sensors, respecting the
// n/2 capacity of each station.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<vector<pair<int,int>>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> side(n + 1, -1);
    int cnt0 = 0, cnt1 = 0, cap = n / 2;
    for (int i = 1; i <= n; i++) {
        long long w0 = 0, w1 = 0; // weight to already-placed on each station
        for (auto &pr : adj[i]) {
            int j = pr.first, w = pr.second;
            if (side[j] == 0) w0 += w;
            else if (side[j] == 1) w1 += w;
        }
        // placing on station 0 gains w1 (cross links to station 1); on 1 gains w0
        int choose;
        if (cnt0 >= cap) choose = 1;
        else if (cnt1 >= cap) choose = 0;
        else choose = (w1 >= w0) ? 0 : 1;
        side[i] = choose;
        if (choose == 0) cnt0++; else cnt1++;
    }
    for (int i = 1; i <= n; i++) printf("%d%c", side[i], i == n ? '\n' : ' ');
    return 0;
}
