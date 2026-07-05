// TIER: strong
// Degree-aware greedy + local search with seeded randomized restarts (deterministic).
//   * key ordering value/(1+degree) so high-value / low-conflict sites go first
//   * add-improvement: repeatedly add any currently-free (non-conflicting) site
//   * remove-one/add-many swaps on the best set found
// Keeps the best independent set over all restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<vector<int>> adj;
vector<ll> w;

static inline ll evalSet(const vector<char>& sel) {
    ll s = 0;
    for (int i = 1; i <= n; i++) if (sel[i]) s += w[i];
    return s;
}

// add any free vertex whose neighbours are all unselected
static void addImprove(vector<char>& sel) {
    bool ch = true;
    while (ch) {
        ch = false;
        for (int v = 1; v <= n; v++) {
            if (sel[v]) continue;
            bool ok = true;
            for (int u : adj[v]) if (sel[u]) { ok = false; break; }
            if (ok) { sel[v] = 1; ch = true; }
        }
    }
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    adj.assign(n + 1, {});
    for (int e = 0; e < m; e++) {
        int u, v;
        scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> deg(n + 1);
    for (int i = 1; i <= n; i++) deg[i] = (int)adj[i].size();

    mt19937 rng(987654321u);

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;

    auto construct = [&](vector<double>& key) -> vector<char> {
        sort(order.begin(), order.end(), [&](int a, int b) { return key[a] > key[b]; });
        vector<char> sel(n + 1, 0), blk(n + 1, 0);
        for (int v : order) {
            if (blk[v]) continue;
            sel[v] = 1;
            for (int u : adj[v]) blk[u] = 1;
        }
        return sel;
    };

    vector<char> best;
    ll bestW = -1;
    vector<double> key(n + 1);

    int R = max(8, min(60, 200000 / (n + 1)));
    for (int r = 0; r < R; r++) {
        for (int i = 1; i <= n; i++) {
            double base = (double)w[i] / (1.0 + deg[i]);
            if (r == 0) key[i] = base;                 // pure degree-aware
            else if (r == 1) key[i] = (double)w[i];    // pure weight-greedy
            else {
                double noise = 0.5 + (double)(rng() % 1000) / 1000.0; // [0.5,1.5)
                key[i] = base * noise;
            }
        }
        vector<char> sel = construct(key);
        addImprove(sel);
        ll val = evalSet(sel);
        if (val > bestW) { bestW = val; best = sel; }
    }

    // remove-one / add-many swaps on the incumbent
    bool improved = true;
    int iter = 0;
    while (improved && iter < 8) {
        improved = false;
        iter++;
        for (int v = 1; v <= n; v++) {
            if (!best[v]) continue;
            vector<char> t = best;
            t[v] = 0;
            addImprove(t);
            ll val = evalSet(t);
            if (val > bestW) { bestW = val; best = t; improved = true; }
        }
    }

    vector<int> ch;
    for (int i = 1; i <= n; i++) if (best[i]) ch.push_back(i);
    printf("%d\n", (int)ch.size());
    for (size_t i = 0; i < ch.size(); i++)
        printf("%d%c", ch[i], (i + 1 < ch.size()) ? ' ' : '\n');
    if (ch.empty()) printf("\n");
    return 0;
}
