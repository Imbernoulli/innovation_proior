// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m, K;
    if (scanf("%d %d %d", &n, &m, &K) != 3) return 0;
    vector<int> p(n + 1), s(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d %d", &p[i], &s[i]);
    vector<int> eu(m), ev(m), ew(m), eg(m);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d %d", &eu[i], &ev[i], &ew[i], &eg[i]);
        adj[eu[i]].push_back(i);
        adj[ev[i]].push_back(i);
    }
    vector<int> c(n + 1, 0);
    // single pass: assign each beacon the channel with least added cost
    // (cross-talk vs already-placed neighbours + own retuning stress).
    for (int u = 1; u <= n; u++) {
        ll best = LLONG_MAX; int bc = 1;
        for (int ch = 1; ch <= K; ch++) {
            ll cost = (ll)s[u] * abs(ch - p[u]);
            for (int id : adj[u]) {
                int v = eu[id] == u ? ev[id] : eu[id];
                if (c[v] == 0) continue;
                int diff = abs(ch - c[v]);
                int pen = eg[id] - diff; if (pen < 0) pen = 0;
                cost += (ll)ew[id] * pen;
            }
            if (cost < best) { best = cost; bc = ch; }
        }
        c[u] = bc;
    }
    for (int i = 1; i <= n; i++) printf("%d%c", c[i], i < n ? ' ' : '\n');
    return 0;
}
