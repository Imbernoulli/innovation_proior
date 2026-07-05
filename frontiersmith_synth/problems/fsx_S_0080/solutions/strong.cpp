// TIER: strong
// Multi-restart, multi-pass balance-preserving local search for balanced max-cut.
// Restart 0 starts from the reference partition; further restarts start from
// deterministic perturbations. Keep the balanced assignment with the largest cut.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

int n, m;
vector<vector<pair<int,ll>>> adj;
vector<int> eu, ev; vector<ll> ew;
vector<int> x;
vector<ll> df;

ll wij(int i, int j) {
    ll s = 0;
    for (auto& e : adj[i]) if (e.first == j) s += e.second;
    return s;
}
void recomputeDf(int u) {
    ll sw = 0, dw = 0;
    for (auto& e : adj[u]) { if (x[e.first] == x[u]) sw += e.second; else dw += e.second; }
    df[u] = sw - dw;
}
void recomputeAllDf() { for (int u = 1; u <= n; u++) recomputeDf(u); }
ll cutOf() {
    ll F = 0;
    for (int e = 0; e < m; e++) if (x[eu[e]] != x[ev[e]]) F += ew[e];
    return F;
}

// deterministic LCG
static u64 rngState = 0;
u64 nextRand() { rngState = rngState * 6364136223846793005ULL + 1442695040888963407ULL; return rngState >> 33; }

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    adj.assign(n + 1, {});
    eu.resize(m); ev.resize(m); ew.resize(m);
    for (int e = 0; e < m; e++) {
        int u, v; ll w; if (scanf("%d %d %lld", &u, &v, &w) != 3) return 0;
        eu[e] = u; ev[e] = v; ew[e] = w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // reference balanced partition R
    vector<int> refX(n + 1, 0);
    {
        vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i + 1;
        const u64 MUL = 11400714819323198485ULL;
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            u64 ka = (u64)a * MUL, kb = (u64)b * MUL;
            if (ka != kb) return ka < kb; return a < b;
        });
        int half = n / 2;
        for (int r = 0; r < n; r++) refX[idx[r]] = (r < half) ? 0 : 1;
    }

    df.assign(n + 1, 0);
    vector<int> bestX = refX;
    x = refX; ll bestCut = cutOf();

    const int RESTARTS = 5;
    const int MAXPASS = 60;

    for (int rs = 0; rs < RESTARTS; rs++) {
        x = refX;
        if (rs > 0) {
            // perturbation: a batch of random balance-preserving swaps
            rngState = 0x1234ABCD ^ (u64)(rs * 2654435761u);
            vector<int> s0, s1;
            for (int i = 1; i <= n; i++) (x[i] == 0 ? s0 : s1).push_back(i);
            int kswap = 1 + (int)(nextRand() % max((size_t)1, s0.size() / 2 + 1));
            for (int t = 0; t < kswap && !s0.empty() && !s1.empty(); t++) {
                int a = s0[nextRand() % s0.size()];
                int b = s1[nextRand() % s1.size()];
                swap(x[a], x[b]);
            }
        }
        recomputeAllDf();

        bool improved = true; int passes = 0;
        while (improved && passes < MAXPASS) {
            improved = false; passes++;
            for (int i = 1; i <= n; i++) {
                if (x[i] != 0) continue;
                for (int j = 1; j <= n; j++) {
                    if (x[j] != 1) continue;
                    ll delta = df[i] + df[j] + 2 * wij(i, j);
                    if (delta > 0) {
                        x[i] = 1; x[j] = 0;
                        recomputeDf(i); recomputeDf(j);
                        for (auto& e : adj[i]) recomputeDf(e.first);
                        for (auto& e : adj[j]) recomputeDf(e.first);
                        improved = true;
                        break; // i moved to crew 1
                    }
                }
            }
        }
        ll c = cutOf();
        if (c > bestCut) { bestCut = c; bestX = x; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", bestX[i], i == n ? '\n' : ' ');
    return 0;
}
