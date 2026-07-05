// TIER: strong
// Randomized multi-restart Kernighan-Lin swap local search for balanced max-cut.
// From several random balanced starts, repeatedly apply the best cut-improving swap
// of one blue rack with one red rack (balance preserved), until no improving swap.
// Keep the best-scoring balanced partition over all restarts.
//
// For a[x]=0, a[y]=1, with D[u] = external - internal incident weight, swapping x,y
// changes the cut by  gain = -(D[x] + D[y]) + 2*w(x,y)  (positive gain improves).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<vector<pair<int,ll>>> g;

ll cutOf(const vector<int>& a) {
    ll F = 0;
    for (int u = 1; u <= n; u++)
        for (auto& e : g[u])
            if (e.first > u && a[u] != a[e.first]) F += e.second;
    return F;
}

mt19937 rng(1234567u);

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }

    vector<int> best;
    ll bestF = -1;

    int restarts = max(8, min(48, 6000 / max(1, n)));
    vector<ll> wrow(n + 1, 0);        // scratch: neighbor weights of a fixed x

    for (int rs = 0; rs < restarts; rs++) {
        // random balanced start (exactly n/2 ones)
        vector<int> a(n + 1, 0);
        vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i + 1;
        shuffle(idx.begin(), idx.end(), rng);
        for (int i = 0; i < n / 2; i++) a[idx[i]] = 1;

        for (int pass = 0; pass < 300; pass++) {
            // D[u] = external - internal incident weight
            vector<ll> D(n + 1, 0);
            for (int u = 1; u <= n; u++) {
                ll ext = 0, intl = 0;
                for (auto& e : g[u]) {
                    if (a[e.first] != a[u]) ext += e.second;
                    else intl += e.second;
                }
                D[u] = ext - intl;
            }

            vector<int> s0, s1;
            for (int u = 1; u <= n; u++) (a[u] == 0 ? s0 : s1).push_back(u);

            ll bestGain = 0; int bx = -1, by = -1;
            for (int x : s0) {
                for (auto& e : g[x]) wrow[e.first] += e.second;  // set neighbor weights
                for (int y : s1) {
                    ll gain = -(D[x] + D[y]) + 2 * wrow[y];
                    if (gain > bestGain) { bestGain = gain; bx = x; by = y; }
                }
                for (auto& e : g[x]) wrow[e.first] = 0;          // reset scratch
            }
            if (bx == -1) break;         // no improving swap -> local optimum
            swap(a[bx], a[by]);
        }

        ll F = cutOf(a);
        if (F > bestF) { bestF = F; best = a; }
    }

    for (int i = 1; i <= n; i++) printf("%d\n", best[i]);
    return 0;
}
