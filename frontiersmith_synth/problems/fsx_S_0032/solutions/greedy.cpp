// TIER: greedy
// One-pass flow-aware placement, then a feasibility repair.
//  1) order performers by total incident flow weight (desc);
//  2) place each on the stage that MAXIMIZES cross-plaza flow to already-placed
//     neighbours (i.e. opposite the stage carrying more of its neighbour weight),
//     ignoring balance;
//  3) repair: while |T0-T1| > tau, move the least-costly performer from the
//     heavier stage to the lighter one.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m; ll tau;
    if (scanf("%d %d %lld", &n, &m, &tau) != 3) return 0;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
    vector<vector<pair<int,int>>> adj(n + 1); // (nbr, w)
    vector<ll> incid(n + 1, 0);
    for (int e = 0; e < m; e++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
        incid[u] += w; incid[v] += w;
    }

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(),
         [&](int x, int y){ return incid[x] > incid[y]; });

    vector<int> stage(n + 1, -1);
    ll T0 = 0, T1 = 0;
    for (int idx : order) {
        // weight of placed neighbours on each stage
        ll w0 = 0, w1 = 0;
        for (auto& pr : adj[idx]) {
            int s = stage[pr.first];
            if (s == 0) w0 += pr.second;
            else if (s == 1) w1 += pr.second;
        }
        // place opposite the heavier side -> maximizes crossing flow; tie -> lighter stage
        int put;
        if (w0 > w1) put = 1;
        else if (w1 > w0) put = 0;
        else put = (T0 <= T1) ? 0 : 1;
        stage[idx] = put;
        if (put == 0) T0 += a[idx]; else T1 += a[idx];
    }

    // ---- repair to satisfy |T0-T1| <= tau ----
    auto flowGainIfMoved = [&](int i, int to) -> ll {
        // change in cut if performer i moves to stage `to`
        ll delta = 0;
        for (auto& pr : adj[i]) {
            int s = stage[pr.first];
            if (s < 0) continue;
            bool crossedBefore = (s != stage[i]);
            bool crossedAfter  = (s != to);
            if (crossedAfter && !crossedBefore) delta += pr.second;
            if (!crossedAfter && crossedBefore) delta -= pr.second;
        }
        return delta;
    };
    while (llabs(T0 - T1) > tau) {
        int heavy = (T0 > T1) ? 0 : 1, light = 1 - heavy;
        // pick node on heavy stage whose move to light loses the least flow
        int best = -1; ll bestDelta = LLONG_MIN;
        for (int i = 1; i <= n; i++) {
            if (stage[i] != heavy) continue;
            ll d = flowGainIfMoved(i, light);
            if (d > bestDelta) { bestDelta = d; best = i; }
        }
        if (best == -1) break;
        if (heavy == 0) { T0 -= a[best]; T1 += a[best]; }
        else            { T1 -= a[best]; T0 += a[best]; }
        stage[best] = light;
    }

    for (int i = 1; i <= n; i++) printf("%d\n", stage[i]);
    return 0;
}
