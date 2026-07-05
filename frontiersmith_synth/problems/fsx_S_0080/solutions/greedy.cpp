// TIER: greedy
// From the reference balanced partition, do ONE first-improvement sweep of
// balance-preserving crew swaps (a node on crew 0 traded with a node on crew 1).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

int n, m;
vector<vector<pair<int,ll>>> adj; // node -> (nbr, weight)
vector<int> x;                    // 1..n assignment
vector<ll> df;                    // deltaFlip[u] = same_w - diff_w

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

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    adj.assign(n + 1, {});
    for (int e = 0; e < m; e++) {
        int u, v; ll w; if (scanf("%d %d %lld", &u, &v, &w) != 3) return 0;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // reference balanced partition R
    x.assign(n + 1, 0);
    {
        vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i + 1;
        const u64 MUL = 11400714819323198485ULL;
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            u64 ka = (u64)a * MUL, kb = (u64)b * MUL;
            if (ka != kb) return ka < kb; return a < b;
        });
        int half = n / 2;
        for (int r = 0; r < n; r++) x[idx[r]] = (r < half) ? 0 : 1;
    }

    df.assign(n + 1, 0);
    for (int u = 1; u <= n; u++) recomputeDf(u);

    // single first-improvement sweep
    for (int i = 1; i <= n; i++) {
        if (x[i] != 0) continue;
        for (int j = 1; j <= n; j++) {
            if (x[j] != 1) continue;
            ll delta = df[i] + df[j] + 2 * wij(i, j);
            if (delta > 0) {
                x[i] = 1; x[j] = 0;               // balance preserved
                // refresh df for affected nodes
                recomputeDf(i); recomputeDf(j);
                for (auto& e : adj[i]) recomputeDf(e.first);
                for (auto& e : adj[j]) recomputeDf(e.first);
                break;                             // i moved to crew 1; next i
            }
        }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", x[i], i == n ? '\n' : ' ');
    return 0;
}
