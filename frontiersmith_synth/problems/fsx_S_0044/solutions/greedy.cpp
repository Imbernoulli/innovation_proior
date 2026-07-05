// TIER: greedy
// Weight-greedy: place asteroids in order of decreasing incident weight, each to the
// refinery that maximizes the marginal separation bonus subject to the n/2 capacity.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<vector<pair<int,int>>> adj(n + 1); // (nbr, w)
    vector<ll> wsum(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        wsum[u] += w; wsum[v] += w;
    }

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(),
         [&](int a, int b){ return wsum[a] > wsum[b]; });

    vector<int> side(n + 1, -1);
    int cnt[2] = {0, 0};
    int cap = n / 2;

    for (int idx = 0; idx < n; idx++) {
        int u = order[idx];
        // marginal bonus of placing u on side 0 vs side 1, given already-placed nbrs
        ll gain0 = 0, gain1 = 0; // bonus earned = weight to placed nbrs on OTHER side
        for (auto& e : adj[u]) {
            int v = e.first, w = e.second;
            if (side[v] == -1) continue;
            if (side[v] == 1) gain0 += w; // u on 0 splits edge to a side-1 nbr
            else              gain1 += w; // u on 1 splits edge to a side-0 nbr
        }
        int choose;
        if (cnt[0] >= cap)      choose = 1;
        else if (cnt[1] >= cap) choose = 0;
        else if (gain0 != gain1) choose = (gain0 > gain1) ? 0 : 1;
        else                    choose = (cnt[0] <= cnt[1]) ? 0 : 1;
        side[u] = choose;
        cnt[choose]++;
    }

    for (int i = 1; i <= n; i++) printf("%d\n", side[i]);
    return 0;
}
