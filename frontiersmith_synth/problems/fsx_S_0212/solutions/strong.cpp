// TIER: strong
// Pairwise-swap local search with randomized multi-restart. From a balanced start, repeatedly
// apply the best cross-fleet swap (one label-0 cell with one label-1 cell) that increases the
// cut; swaps preserve exact balance so feasibility is maintained. Keep the best labeling over
// all restarts. Delta of a swap uses cached single-flip gains plus an adjacency-weight
// correction for a shared edge.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D;
vector<vector<pair<int,ll>>> adj;
vector<vector<ll>> wmat; // dense weight matrix (n <= 240)

ll cutOf(const vector<int>& x) {
    ll F = 0;
    for (int u = 1; u <= n; u++)
        for (auto& e : adj[u])
            if (e.first > u && x[u] != x[e.first]) F += e.second;
    return F;
}

// gain of flipping node u alone: edges to same-side neighbors become cut (+w),
// edges to opposite-side neighbors stop being cut (-w).
ll flipGain(const vector<int>& x, int u) {
    ll g = 0;
    for (auto& e : adj[u]) g += (x[u] == x[e.first]) ? e.second : -e.second;
    return g;
}

int main() {
    if (scanf("%d %d %d", &n, &m, &D) != 3) return 0;
    adj.assign(n + 1, {});
    wmat.assign(n + 1, vector<ll>(n + 1, 0));
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        wmat[u][v] += w; wmat[v][u] += w;
    }

    mt19937 rng(2024u);
    vector<int> best;
    ll bestF = -1;

    int restarts = 8;
    for (int rs = 0; rs < restarts; rs++) {
        // balanced random start
        vector<int> x(n + 1, 0);
        vector<int> ord(n);
        for (int i = 0; i < n; i++) ord[i] = i + 1;
        shuffle(ord.begin(), ord.end(), rng);
        for (int i = 0; i < n; i++) x[ord[i]] = (i < n / 2) ? 0 : 1;

        int iters = 0, maxIters = 3 * n + 50;
        while (iters++ < maxIters) {
            vector<ll> fg(n + 1);
            for (int u = 1; u <= n; u++) fg[u] = flipGain(x, u);

            ll bestG = 0; int bu = -1, bv = -1;
            for (int u = 1; u <= n; u++) {
                if (x[u] != 0) continue;
                for (int v = 1; v <= n; v++) {
                    if (x[v] != 1) continue;
                    // flipping both u and v: shared edge toggled twice -> net unchanged,
                    // but each flipGain counted it as -w, so add back 2*w(u,v).
                    ll g = fg[u] + fg[v] + 2 * wmat[u][v];
                    if (g > bestG) { bestG = g; bu = u; bv = v; }
                }
            }
            if (bu == -1) break; // no improving swap
            x[bu] = 1; x[bv] = 0;
        }

        ll F = cutOf(x);
        if (F > bestF) { bestF = F; best.assign(x.begin(), x.end()); }
    }

    for (int i = 1; i <= n; i++) printf("%d\n", best[i]);
    return 0;
}
