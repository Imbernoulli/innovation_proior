// TIER: greedy
// Greedy constructive balanced min k-cut.  Process hives in descending weighted
// degree order; assign each hive to the yard (with remaining capacity) to which it
// already has the most placed drift, so heavy drift tends to stay intra-yard.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long m; int k;
    scanf("%d %lld %d", &n, &m, &k);
    int s = n / k;

    vector<vector<pair<int,long long>>> adj(n + 1);
    vector<long long> wdeg(n + 1, 0);
    for (long long e = 0; e < m; e++) {
        int u, v; long long w;
        scanf("%d %d %lld", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        wdeg[u] += w; wdeg[v] += w;
    }

    // order hives by descending weighted degree (ties by id)
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (wdeg[a] != wdeg[b]) return wdeg[a] > wdeg[b];
        return a < b;
    });

    vector<int> yard(n + 1, -1);
    vector<int> cap(k, s);
    // pull[g] reused per hive
    vector<long long> pull(k, 0);

    for (int idx = 0; idx < n; idx++) {
        int h = order[idx];
        for (int g = 0; g < k; g++) pull[g] = 0;
        for (auto &pr : adj[h]) {
            int nb = pr.first;
            if (yard[nb] >= 0) pull[yard[nb]] += pr.second;
        }
        // choose yard with capacity maximizing pull; tie -> most remaining capacity
        int best = -1;
        long long bestPull = -1;
        int bestCap = -1;
        for (int g = 0; g < k; g++) {
            if (cap[g] <= 0) continue;
            if (pull[g] > bestPull || (pull[g] == bestPull && cap[g] > bestCap)) {
                bestPull = pull[g];
                bestCap = cap[g];
                best = g;
            }
        }
        yard[h] = best;
        cap[best]--;
    }

    for (int i = 1; i <= n; i++)
        printf("%d%c", yard[i], i == n ? '\n' : ' ');
    return 0;
}
