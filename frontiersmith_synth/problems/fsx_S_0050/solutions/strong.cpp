// TIER: strong
// Randomized multi-restart weighted greedy set cover + redundancy pruning.
// Each restart: run greedy with a perturbed cost/coverage score, then prune every
// station that is redundant (its demands are all covered by others), cheapest-covering
// removed first. Keep the cheapest feasible cover found over all restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D, P;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;
vector<int> demIdOf;
vector<int> stNode, stCost, stRad;
vector<vector<int>> cover;

void computeCoverage() {
    cover.assign(P + 1, {});
    for (int i = 1; i <= P; i++) {
        int src = stNode[i]; ll rad = stRad[i];
        vector<ll> dist(n + 1, LLONG_MAX);
        priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
        dist[src] = 0; pq.push({0, src});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            if (d > rad) continue;
            if (demIdOf[u]) cover[i].push_back(demIdOf[u]);
            for (auto& e : g[u]) {
                ll nd = d + e.w;
                if (nd <= rad && nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
            }
        }
    }
}

mt19937 rng(0xC0FFEEu);

// One greedy pass with a multiplicative perturbation on the score; returns chosen set.
vector<int> greedyPass(double jitter) {
    vector<char> covered(D + 1, 0);
    vector<char> chosen(P + 1, 0);
    int remaining = D;
    vector<int> pick;
    while (remaining > 0) {
        int best = -1; double bestScore = 1e18;
        for (int i = 1; i <= P; i++) {
            if (chosen[i]) continue;
            int neu = 0;
            for (int d : cover[i]) if (!covered[d]) neu++;
            if (neu == 0) continue;
            double ratio = (double)stCost[i] / (double)neu;
            if (jitter > 0.0) {
                double f = 1.0 + jitter * ((double)(rng() % 1000) / 1000.0 - 0.5);
                ratio *= f;
            }
            if (ratio < bestScore - 1e-12) { bestScore = ratio; best = i; }
        }
        if (best == -1) break;
        chosen[best] = 1; pick.push_back(best);
        for (int d : cover[best]) if (!covered[d]) { covered[d] = 1; remaining--; }
    }
    return pick;
}

// Remove redundant stations (cheapest-covering-per-demand removed first is not ideal;
// we remove the MOST EXPENSIVE redundant station first to maximize savings).
void prune(vector<int>& pick) {
    vector<int> coverCnt(D + 1, 0);
    vector<char> inSet(P + 1, 0);
    for (int i : pick) { inSet[i] = 1; for (int d : cover[i]) coverCnt[d]++; }
    // most expensive first
    vector<int> order = pick;
    sort(order.begin(), order.end(), [&](int a, int b){ return stCost[a] > stCost[b]; });
    for (int i : order) {
        bool redundant = true;
        for (int d : cover[i]) if (coverCnt[d] < 2) { redundant = false; break; }
        if (redundant) { inSet[i] = 0; for (int d : cover[i]) coverCnt[d]--; }
    }
    vector<int> out;
    for (int i = 1; i <= P; i++) if (inSet[i]) out.push_back(i);
    pick.swap(out);
}

ll totalCost(const vector<int>& pick) {
    ll s = 0; for (int i : pick) s += stCost[i]; return s;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }
    scanf("%d", &D);
    demIdOf.assign(n + 1, 0);
    for (int i = 1; i <= D; i++) { int x; scanf("%d", &x); demIdOf[x] = i; }
    scanf("%d", &P);
    stNode.assign(P + 1, 0); stCost.assign(P + 1, 0); stRad.assign(P + 1, 0);
    for (int i = 1; i <= P; i++) scanf("%d %d %d", &stNode[i], &stCost[i], &stRad[i]);

    computeCoverage();

    vector<int> best;
    ll bestCost = LLONG_MAX;

    int restarts = 40;
    for (int r = 0; r < restarts; r++) {
        double jitter = (r == 0) ? 0.0 : 0.15 + 0.01 * (r % 20);
        vector<int> pick = greedyPass(jitter);
        prune(pick);
        ll c = totalCost(pick);
        if (c < bestCost) { bestCost = c; best = pick; }
    }

    printf("%d\n", (int)best.size());
    for (int i : best) printf("%d\n", i);
    return 0;
}
