// TIER: greedy
// Sequential greedy max-cut with a balance cap. Process zones 1..n in order; place
// each on the depot that maximizes the incremental separated conflict weight with
// the zones already placed, subject to neither depot exceeding n/2+slack. A decent
// one-pass heuristic, but it commits early and cannot recover the hidden per-family
// sub-bipartition, so it leaves cut weight on the table vs. local search.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, slack;
    if (scanf("%d %d %d", &n, &m, &slack) != 3) return 0;
    vector<vector<pair<int,ll>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    int cap = n / 2 + slack;
    vector<int> lab(n + 1, -1);
    int cnt0 = 0, cnt1 = 0;
    for (int v = 1; v <= n; v++) {
        // gain of placing v on side 0 = weight to already-placed neighbors on side 1
        ll g0 = 0, g1 = 0;
        for (auto& e : adj[v]) {
            int u = e.first; if (lab[u] == -1) continue;
            if (lab[u] == 1) g0 += e.second; else g1 += e.second;
        }
        bool can0 = cnt0 < cap, can1 = cnt1 < cap;
        int put;
        if (can0 && !can1) put = 0;
        else if (can1 && !can0) put = 1;
        else {
            if (g0 > g1) put = 0;
            else if (g1 > g0) put = 1;
            else put = (cnt0 <= cnt1) ? 0 : 1; // tie -> smaller side
        }
        lab[v] = put;
        if (put == 0) cnt0++; else cnt1++;
    }
    for (int i = 1; i <= n; i++) printf("%d%c", lab[i], i == n ? '\n' : ' ');
    return 0;
}
