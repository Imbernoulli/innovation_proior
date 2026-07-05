// TIER: strong
// Randomized multi-restart maximum-weight independent set with local search.
// Each restart: build a maximal independent docket via a randomized value/(1+degree)
// greedy (weights perturbed), then improve it with (1,k)-swap local search:
//   for a free target v whose ONLY scheduled conflict is one target u with w[v] > w[u],
//   swap u out for v; after any swap, re-absorb every now-free target.
// Keep the best-valued docket across all restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<vector<int>> adj;

mt19937 rng(0xA11CE5EDu);

// count scheduled conflicts of v; if exactly one, return it in 'the', else -1 sentinel logic
static inline int schedConflicts(int v, const vector<char>& chosen, int& theOne) {
    int c = 0; theOne = -1;
    for (int u : adj[v]) if (chosen[u]) { c++; theOne = u; if (c > 1) return c; }
    return c;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    adj.assign(n + 1, {});
    for (int e = 0; e < m; e++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v); adj[v].push_back(u);
    }
    vector<int> deg(n + 1);
    for (int i = 1; i <= n; i++) deg[i] = (int)adj[i].size();

    vector<int> bestPick;
    ll bestF = -1;

    int restarts = 40;
    for (int rs = 0; rs < restarts; rs++) {
        // --- randomized value/(1+degree) greedy build ---
        vector<double> keyv(n + 1);
        for (int i = 1; i <= n; i++) {
            double base = (double)w[i] / (1.0 + deg[i]);
            double jitter = (rs == 0) ? 1.0
                          : (0.55 + 0.9 * (double)(rng() % 1000) / 1000.0);
            keyv[i] = base * jitter;
            if (rs == 0) keyv[i] = (double)w[i]; // first restart = plain weight order
        }
        vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i + 1;
        sort(order.begin(), order.end(), [&](int a, int b){ return keyv[a] > keyv[b]; });

        vector<char> blocked(n + 1, 0), chosen(n + 1, 0);
        for (int u : order) {
            if (blocked[u]) continue;
            chosen[u] = 1;
            for (int v : adj[u]) blocked[v] = 1;
        }

        // --- (1,k)-swap local search ---
        bool improved = true; int guard = 0;
        while (improved && guard++ < 200) {
            improved = false;
            for (int v = 1; v <= n; v++) {
                if (chosen[v]) continue;
                int u; int c = schedConflicts(v, chosen, u);
                if (c == 0) {                 // free -> just add (keeps maximality)
                    chosen[v] = 1; improved = true;
                } else if (c == 1 && w[v] > w[u]) {
                    chosen[u] = 0; chosen[v] = 1; improved = true;
                    // re-absorb any target that became free after removing u
                    for (int x : adj[u]) {
                        if (chosen[x]) continue;
                        int uu; int cc = schedConflicts(x, chosen, uu);
                        if (cc == 0) chosen[x] = 1;
                    }
                }
            }
        }

        ll F = 0;
        vector<int> pick;
        for (int i = 1; i <= n; i++) if (chosen[i]) { F += w[i]; pick.push_back(i); }
        if (F > bestF) { bestF = F; bestPick = pick; }
    }

    printf("%d\n", (int)bestPick.size());
    for (int u : bestPick) printf("%d\n", u);
    return 0;
}
