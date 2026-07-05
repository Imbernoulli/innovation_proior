// TIER: strong
// Greedy init + iterated local search with weight-aware randomized restarts.
// Each pass: for every gallery, recompute per-channel annoyance over its FULL
// current neighborhood and move it to the least-annoying channel. Repeat until
// convergence. Several perturbed restarts; keep the best (min total annoyance).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, k, m;
vector<int> hu, hv, hw;
vector<vector<pair<int,int>>> adj; // node -> (neighbor, weight)

ll totalAnnoy(const vector<int>& c) {
    ll F = 0;
    for (int i = 0; i < m; i++) if (c[hu[i]] == c[hv[i]]) F += hw[i];
    return F;
}

// one convergence run of local search from assignment c (modifies in place)
void localSearch(vector<int>& c, mt19937& rng) {
    vector<ll> cost(k + 1);
    bool improved = true;
    int guard = 0;
    while (improved && guard < 40) {
        improved = false; guard++;
        for (int g = 1; g <= n; g++) {
            fill(cost.begin(), cost.end(), 0);
            for (auto& e : adj[g]) cost[c[e.first]] += e.second;
            int best = c[g]; ll bestCost = cost[c[g]];
            for (int ch = 1; ch <= k; ch++)
                if (cost[ch] < bestCost) { bestCost = cost[ch]; best = ch; }
            if (best != c[g]) { c[g] = best; improved = true; }
        }
    }
}

int main() {
    if (scanf("%d %d %d", &n, &k, &m) != 3) return 0;
    hu.resize(m); hv.resize(m); hw.resize(m);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &hu[i], &hv[i], &hw[i]);
        adj[hu[i]].push_back({hv[i], hw[i]});
        adj[hv[i]].push_back({hu[i], hw[i]});
    }

    mt19937 rng(12345u); // fixed seed -> deterministic

    // greedy init
    vector<int> c(n + 1, 0);
    {
        vector<ll> cost(k + 1);
        for (int g = 1; g <= n; g++) {
            fill(cost.begin(), cost.end(), 0);
            for (auto& e : adj[g]) if (c[e.first] != 0) cost[c[e.first]] += e.second;
            int best = 1; ll bc = cost[1];
            for (int ch = 2; ch <= k; ch++) if (cost[ch] < bc) { bc = cost[ch]; best = ch; }
            c[g] = best;
        }
    }
    localSearch(c, rng);
    vector<int> bestC = c;
    ll bestF = totalAnnoy(c);

    // weight-aware randomized restarts
    int restarts = 6;
    for (int r = 0; r < restarts; r++) {
        vector<int> cc = bestC;
        // perturb: randomly recolor a fraction of galleries
        int perturb = max(1, n / 5);
        for (int p = 0; p < perturb; p++) {
            int g = 1 + (int)(rng() % n);
            cc[g] = 1 + (int)(rng() % k);
        }
        localSearch(cc, rng);
        ll f = totalAnnoy(cc);
        if (f < bestF) { bestF = f; bestC = cc; }
    }

    for (int g = 1; g <= n; g++) printf("%d%c", bestC[g], g == n ? '\n' : ' ');
    return 0;
}
