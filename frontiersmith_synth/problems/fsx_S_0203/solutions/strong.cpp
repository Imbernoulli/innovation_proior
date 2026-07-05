// TIER: strong
// Two greedy seeds (weight-desc and weight-per-degree) each refined by an
// eviction local search (add a lot, drop lighter conflicting chosen lots when
// that raises profit); keep the best resulting slate. Deterministic.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> w;
vector<vector<int>> adj;

// eviction local search from a starting selection
ll refine(vector<char>& sel) {
    bool improved = true;
    int guard = 0;
    while (improved && guard++ < 1000) {
        improved = false;
        for (int u = 1; u <= n; u++) {
            if (sel[u]) continue;
            ll lossW = 0;
            bool ok = true;
            for (int nb : adj[u]) if (sel[nb]) lossW += w[nb];
            (void)ok;
            if ((ll)w[u] > lossW) {
                // add u, evict conflicting chosen lots
                for (int nb : adj[u]) if (sel[nb]) sel[nb] = 0;
                sel[u] = 1;
                improved = true;
            }
        }
    }
    ll tot = 0;
    for (int i = 1; i <= n; i++) if (sel[i]) tot += w[i];
    return tot;
}

vector<char> greedyBy(vector<int> order) {
    vector<char> blocked(n + 1, 0), sel(n + 1, 0);
    for (int u : order) {
        if (blocked[u]) continue;
        sel[u] = 1; blocked[u] = 1;
        for (int nb : adj[u]) blocked[nb] = 1;
    }
    return sel;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> deg(n + 1, 0);
    for (int i = 1; i <= n; i++) deg[i] = (int)adj[i].size();

    // seed A: weight descending
    vector<int> ordA(n);
    for (int i = 0; i < n; i++) ordA[i] = i + 1;
    sort(ordA.begin(), ordA.end(), [&](int a, int b) {
        if (w[a] != w[b]) return w[a] > w[b];
        return a < b;
    });
    // seed B: weight per (degree+1) descending
    vector<int> ordB(n);
    for (int i = 0; i < n; i++) ordB[i] = i + 1;
    sort(ordB.begin(), ordB.end(), [&](int a, int b) {
        double ra = (double)w[a] / (deg[a] + 1.0);
        double rb = (double)w[b] / (deg[b] + 1.0);
        if (ra != rb) return ra > rb;
        return a < b;
    });

    vector<char> selA = greedyBy(ordA);
    vector<char> selB = greedyBy(ordB);
    ll fA = refine(selA);
    ll fB = refine(selB);

    vector<char>& best = (fA >= fB) ? selA : selB;

    vector<int> out;
    for (int i = 1; i <= n; i++) if (best[i]) out.push_back(i);
    printf("%d\n", (int)out.size());
    for (size_t i = 0; i < out.size(); i++)
        printf("%d%c", out[i], i + 1 == out.size() ? '\n' : ' ');
    if (out.empty()) printf("\n");
    return 0;
}
