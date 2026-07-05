// TIER: strong
// GRASP-style randomized cost-effective set cover with multiple seeded restarts,
// each followed by redundancy pruning; keep the cheapest feasible covering found.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D, r;
vector<int> cost;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;
vector<int> shops;
vector<int> shopId;
vector<vector<int>> coverage; // per node -> shop indices covered

vector<int> coverFrom(int src) {
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[src] = 0; pq.push({0, src});
    vector<int> res;
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (shopId[u] >= 0) res.push_back(shopId[u]);
        if (d >= r) continue;
        for (auto& e : g[u]) {
            ll nd = d + e.w;
            if (nd <= r && nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
        }
    }
    return res;
}

ll totalCost(const vector<int>& sel) { ll c = 0; for (int u : sel) c += cost[u]; return c; }

// remove redundant depots (cheapest-first is greedy; we drop most expensive first
// to shed cost) while keeping full coverage.
vector<int> prune(vector<int> sel) {
    vector<int> cnt(D, 0);
    for (int u : sel) for (int s : coverage[u]) cnt[s]++;
    // try to drop expensive depots first
    sort(sel.begin(), sel.end(), [&](int a, int b){ return cost[a] > cost[b]; });
    vector<char> keep(sel.size(), 1);
    for (size_t i = 0; i < sel.size(); i++) {
        int u = sel[i];
        bool redundant = true;
        for (int s : coverage[u]) if (cnt[s] <= 1) { redundant = false; break; }
        if (redundant) { keep[i] = 0; for (int s : coverage[u]) cnt[s]--; }
    }
    vector<int> res;
    for (size_t i = 0; i < sel.size(); i++) if (keep[i]) res.push_back(sel[i]);
    return res;
}

vector<int> randomizedGreedy(mt19937& rng, double alpha) {
    vector<char> covered(D, 0);
    int remaining = D;
    vector<char> used(n + 1, 0);
    vector<int> chosen;
    while (remaining > 0) {
        double bestRatio = 1e18;
        for (int u = 1; u <= n; u++) {
            if (used[u]) continue;
            int nc = 0; for (int s : coverage[u]) if (!covered[s]) nc++;
            if (nc == 0) continue;
            double ratio = (double)cost[u] / nc;
            if (ratio < bestRatio) bestRatio = ratio;
        }
        if (bestRatio > 1e17) break;
        // restricted candidate list: within alpha of best ratio
        vector<int> rcl;
        for (int u = 1; u <= n; u++) {
            if (used[u]) continue;
            int nc = 0; for (int s : coverage[u]) if (!covered[s]) nc++;
            if (nc == 0) continue;
            double ratio = (double)cost[u] / nc;
            if (ratio <= bestRatio * alpha + 1e-12) rcl.push_back(u);
        }
        int pick = rcl[rng() % rcl.size()];
        used[pick] = 1; chosen.push_back(pick);
        for (int s : coverage[pick]) if (!covered[s]) { covered[s] = 1; remaining--; }
    }
    return chosen;
}

int main() {
    if (scanf("%d %d %d %d", &n, &m, &D, &r) != 4) return 0;
    cost.assign(n + 1, 0);
    for (int u = 1; u <= n; u++) scanf("%d", &cost[u]);
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        g[u].push_back({v, (ll)w}); g[v].push_back({u, (ll)w});
    }
    shops.resize(D);
    shopId.assign(n + 1, -1);
    for (int i = 0; i < D; i++) { scanf("%d", &shops[i]); shopId[shops[i]] = i; }

    coverage.assign(n + 1, {});
    for (int u = 1; u <= n; u++) coverage[u] = coverFrom(u);

    mt19937 rng(987654321u); // fixed seed -> deterministic

    // start from the pure-greedy (alpha=1.0) solution, pruned
    vector<int> best = prune(randomizedGreedy(rng, 1.0));
    ll bestCost = totalCost(best);

    int restarts = 60;
    for (int it = 0; it < restarts; it++) {
        double alpha = 1.0 + 0.6 * ((it % 6) / 5.0); // 1.0 .. 1.6
        vector<int> cand = prune(randomizedGreedy(rng, alpha));
        ll cc = totalCost(cand);
        if (cc < bestCost) { bestCost = cc; best = cand; }
    }

    printf("%d\n", (int)best.size());
    for (int u : best) printf("%d\n", u);
    return 0;
}
