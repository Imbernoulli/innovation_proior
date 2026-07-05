// TIER: strong
// Greedy construction, then balance-preserving pair-swap hill climbing:
// repeatedly pick a sensor from each station and swap them if the swap increases
// cross-checked weight. Deterministic (fixed seed), with an effort budget scaled
// by the graph size. Recovers most of the planted balanced bisection.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<vector<pair<int,int>>> adj(n + 1);
    long long totdeg = 0;
    for (int e = 0; e < m; e++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        totdeg += 2;
    }
    vector<int> side(n + 1, -1);
    int cnt0 = 0, cnt1 = 0, cap = n / 2;
    // greedy init (same as greedy.cpp)
    for (int i = 1; i <= n; i++) {
        long long w0 = 0, w1 = 0;
        for (auto &pr : adj[i]) {
            int j = pr.first, w = pr.second;
            if (side[j] == 0) w0 += w;
            else if (side[j] == 1) w1 += w;
        }
        int choose;
        if (cnt0 >= cap) choose = 1;
        else if (cnt1 >= cap) choose = 0;
        else choose = (w1 >= w0) ? 0 : 1;
        side[i] = choose;
        if (choose == 0) cnt0++; else cnt1++;
    }

    // lists of vertices on each station
    vector<int> A, Bv;
    for (int i = 1; i <= n; i++) (side[i] == 0 ? A : Bv).push_back(i);

    // delta of swapping a (station 0) with b (station 1)
    auto delta = [&](int a, int b) -> long long {
        long long d = 0;
        for (auto &pr : adj[a]) { // a moves 0->1
            int x = pr.first, w = pr.second;
            if (x == b) continue;
            d += (side[x] == 0) ? w : -w;
        }
        for (auto &pr : adj[b]) { // b moves 1->0
            int x = pr.first, w = pr.second;
            if (x == a) continue;
            d += (side[x] == 1) ? w : -w;
        }
        return d;
    };

    mt19937 rng(987654321u);
    double avgdeg = (n > 0) ? (double)totdeg / n : 1.0;
    long long budget = (long long)(4.0e7 / (avgdeg + 1.0));
    budget = max<long long>(budget, 50000);
    budget = min<long long>(budget, 2000000);

    if (!A.empty() && !Bv.empty()) {
        for (long long it = 0; it < budget; it++) {
            int ia = rng() % A.size();
            int ib = rng() % Bv.size();
            int a = A[ia], b = Bv[ib];
            long long d = delta(a, b);
            if (d > 0) {
                side[a] = 1; side[b] = 0;
                A[ia] = b; Bv[ib] = a;
            }
        }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", side[i], i == n ? '\n' : ' ');
    return 0;
}
