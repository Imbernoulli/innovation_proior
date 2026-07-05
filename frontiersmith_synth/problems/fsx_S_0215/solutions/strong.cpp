// TIER: strong
// Randomized multi-restart greedy local search for max-weight independent set.
// Tries deterministic value-order and value/degree-order, then many perturbed
// orderings; keeps the maximal independent set of highest total weight.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<vector<int>> adj;

// build a maximal independent set greedily following `order`; return total weight,
// and (via out) the chosen membership.
ll buildIS(const vector<int>& order, vector<char>& mem) {
    vector<char> blocked(n + 1, 0);
    fill(mem.begin(), mem.end(), 0);
    ll tot = 0;
    for (int u : order) {
        if (blocked[u]) continue;
        mem[u] = 1; tot += w[u];
        for (int v : adj[u]) blocked[v] = 1;
    }
    return tot;
}

int main() {
    scanf("%d %d", &n, &m);
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    adj.assign(n + 1, {});
    for (int e = 0; e < m; e++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v); adj[v].push_back(u);
    }
    vector<int> deg(n + 1);
    for (int i = 1; i <= n; i++) deg[i] = (int)adj[i].size();

    ll bestW = -1;
    vector<char> bestMem(n + 1, 0), mem(n + 1, 0);

    auto consider = [&](vector<int>& order) {
        ll tw = buildIS(order, mem);
        if (tw > bestW) { bestW = tw; bestMem = mem; }
    };

    vector<int> ids(n);
    for (int i = 0; i < n; i++) ids[i] = i + 1;

    // 1) value order
    { vector<int> o = ids;
      sort(o.begin(), o.end(), [&](int a, int b){ return w[a] > w[b]; });
      consider(o); }
    // 2) value/(degree+1) order -- favors cheap-to-block low-degree vertices
    { vector<int> o = ids;
      sort(o.begin(), o.end(), [&](int a, int b){
          double ka = (double)w[a] / (deg[a] + 1.0);
          double kb = (double)w[b] / (deg[b] + 1.0);
          return ka > kb; });
      consider(o); }

    // 3) many randomized restarts: perturb a value/degree key deterministically
    mt19937 rng(987654321u);
    int restarts = 260;
    for (int r = 0; r < restarts; r++) {
        vector<pair<double,int>> key(n);
        for (int i = 0; i < n; i++) {
            double base = (double)w[ids[i]] / (deg[ids[i]] + 1.0);
            double noise = (double)(rng() % 100000) / 100000.0; // [0,1)
            // blend base ranking with randomness so restarts explore diverse maximal sets
            key[i] = { base * (0.4 + 1.2 * noise), ids[i] };
        }
        sort(key.begin(), key.end(), [](const pair<double,int>& a, const pair<double,int>& b){
            return a.first > b.first; });
        vector<int> o(n);
        for (int i = 0; i < n; i++) o[i] = key[i].second;
        consider(o);
    }

    // local improvement: attempt to add any vertex whose neighbors are all outside the set
    // (buildIS already yields maximal sets, but cross-order merge can leave slack).
    {
        vector<char> nbrIn(n + 1, 0);
        // recompute: for each vertex not in set, check if addable
        bool improved = true;
        int guard = 0;
        while (improved && guard++ < 4) {
            improved = false;
            for (int u = 1; u <= n; u++) {
                if (bestMem[u]) continue;
                bool ok = true;
                for (int v : adj[u]) if (bestMem[v]) { ok = false; break; }
                if (ok) { bestMem[u] = 1; bestW += w[u]; improved = true; }
            }
        }
    }

    vector<int> pick;
    for (int u = 1; u <= n; u++) if (bestMem[u]) pick.push_back(u);
    printf("%d\n", (int)pick.size());
    for (int u : pick) printf("%d\n", u);
    return 0;
}
