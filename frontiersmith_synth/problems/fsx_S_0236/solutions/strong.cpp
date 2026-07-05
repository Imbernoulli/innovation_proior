// TIER: strong
// Greedy constructive start, then deterministic balance-preserving local search:
// repeatedly try swapping a module on side 0 with a module on side 1; apply the swap if it
// increases earned value. Uses a fixed-seed RNG for reproducibility and a bounded budget.
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

    // --- greedy constructive start ---
    for (int i = 1; i <= n; i++) {
        long long g0 = 0, g1 = 0;
        for (auto &e : adj[i]) {
            int y = e[0]; if (side[y] < 0) continue;
            int t = e[1], w = e[2];
            bool cut0 = (0 != side[y]); if ((t == 0) ? cut0 : !cut0) g0 += w;
            bool cut1 = (1 != side[y]); if ((t == 0) ? cut1 : !cut1) g1 += w;
        }
        int choose;
        if (cnt0 >= half) choose = 1;
        else if (cnt1 >= half) choose = 0;
        else choose = (g1 > g0) ? 1 : 0;
        side[i] = choose;
        if (choose == 0) cnt0++; else cnt1++;
    }

    // satisfaction contribution of one wire given endpoint labels
    auto contrib = [](int su, int sv, int t, int w) -> long long {
        bool cut = (su != sv);
        bool sat = (t == 0) ? cut : !cut;
        return sat ? (long long)w : 0LL;
    };

    // buckets of module ids per side
    vector<int> S0, S1;
    for (int i = 1; i <= n; i++) (side[i] == 0 ? S0 : S1).push_back(i);

    mt19937 rng(987654321u);
    long long touchBudget = 200000000LL; // bound edge touches for the time limit
    long long touched = 0;
    int noImprove = 0;
    int cap = max(2000, n * 400);
    while (touched < touchBudget && noImprove < cap && !S0.empty() && !S1.empty()) {
        int a = S0[rng() % S0.size()]; // side 0
        int b = S1[rng() % S1.size()]; // side 1
        // delta of earned value if we swap a (->1) and b (->0)
        long long delta = 0;
        for (auto &e : adj[a]) {
            int y = e[0], t = e[1], w = e[2];
            int sy = (y == b) ? side[b] : side[y];
            int syNew = (y == b) ? 0 : side[y]; // b would move to 0
            long long oldc = contrib(0, sy, t, w);
            long long newc = contrib(1, syNew, t, w);
            delta += newc - oldc;
            touched++;
        }
        for (auto &e : adj[b]) {
            int y = e[0]; if (y == a) continue; // already handled by a's edge to b
            int t = e[1], w = e[2];
            int sy = side[y];
            long long oldc = contrib(1, sy, t, w);
            long long newc = contrib(0, sy, t, w);
            delta += newc - oldc;
            touched++;
        }
        if (delta > 0) {
            side[a] = 1; side[b] = 0;
            // buckets: a moved 0->1, b moved 1->0. Rebuild membership lazily by swapping ids.
            for (int &x : S0) if (x == a) { x = b; break; }
            for (int &x : S1) if (x == b) { x = a; break; }
            noImprove = 0;
        } else {
            noImprove++;
        }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", side[i], i == n ? '\n' : ' ');
    return 0;
}
