// TIER: strong
// Greedy start + swap-based local search (Kernighan-Lin flavor): repeatedly swap the
// best cross-bowl pair to climb the cut while preserving balance exactly.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, tol;
vector<vector<pair<int,int>>> adj;

int main() {
    scanf("%d %d %d", &n, &m, &tol);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    int maxSide = (n + tol) / 2;

    // ---- greedy warm start ----
    vector<int> side(n + 1, -1);
    int c0 = 0, c1 = 0;
    for (int u = 1; u <= n; u++) {
        ll w0 = 0, w1 = 0;
        for (auto& e : adj[u]) {
            if (side[e.first] == 0) w0 += e.second;
            else if (side[e.first] == 1) w1 += e.second;
        }
        int choose;
        if (c0 >= maxSide) choose = 1;
        else if (c1 >= maxSide) choose = 0;
        else choose = (w1 >= w0) ? 0 : 1;
        side[u] = choose;
        if (choose == 0) c0++; else c1++;
    }

    // gain[u] = cut increase if u flips = (weight to same side) - (weight to opposite side)
    auto computeGain = [&](vector<ll>& gain) {
        for (int u = 1; u <= n; u++) {
            ll same = 0, diff = 0;
            for (auto& e : adj[u]) {
                if (side[e.first] == side[u]) same += e.second;
                else diff += e.second;
            }
            gain[u] = same - diff;
        }
    };
    vector<ll> gain(n + 1, 0);
    computeGain(gain);

    int maxIter = min(4 * n + 50, 6000);
    for (int it = 0; it < maxIter; it++) {
        // best-gain wagon on each side
        int bu = -1, bv = -1;
        ll gu = LLONG_MIN, gv = LLONG_MIN;
        for (int u = 1; u <= n; u++) {
            if (side[u] == 0) { if (gain[u] > gu) { gu = gain[u]; bu = u; } }
            else               { if (gain[u] > gv) { gv = gain[u]; bv = u; } }
        }
        if (bu < 0 || bv < 0) break;
        // edge weight between the chosen pair (subtracted twice in a swap)
        ll wuv = 0;
        for (auto& e : adj[bu]) if (e.first == bv) wuv += e.second;
        ll delta = gu + gv - 2 * wuv;
        if (delta <= 0) break;              // local optimum
        side[bu] = 1; side[bv] = 0;         // swap preserves balance
        computeGain(gain);                  // O(m) recompute
    }

    for (int i = 1; i <= n; i++) printf("%d%c", side[i], i == n ? '\n' : ' ');
    return 0;
}
