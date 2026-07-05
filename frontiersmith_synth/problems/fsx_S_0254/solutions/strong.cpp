// TIER: strong
// Cost-effective greedy (maximize newly-covered-per-cost) with a few randomized
// restarts, followed by a redundancy-pruning pass that drops the most expensive
// sensors whose coverage is fully supplied by others. Beats coverage-only greedy
// because it explicitly minimizes cost, and diverges from it per-instance.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M; ll R;
vector<ll> cost;
struct E { int to; ll w; };
vector<vector<E>> g;
vector<vector<int>> ball;

void buildBalls() {
    ball.assign(N + 1, {});
    vector<ll> dist(N + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    for (int s = 1; s <= N; s++) {
        vector<int> touched;
        dist[s] = 0; pq.push({0, s});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d != dist[u]) continue;
            touched.push_back(u);
            ball[s].push_back(u);
            for (auto& e : g[u]) {
                ll nd = d + e.w;
                if (nd <= R && (dist[e.to] < 0 || nd < dist[e.to])) { dist[e.to] = nd; pq.push({nd, e.to}); }
            }
        }
        for (int u : touched) dist[u] = -1;
    }
}

uint64_t rng = 0x243F6A8885A308D3ULL;
double urand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 11) * (1.0 / 9007199254740992.0); }

// cost-effective greedy; perturb in [1, 1+eps) on the cost denominator per restart.
vector<int> greedyRun(double eps) {
    vector<char> covered(N + 1, 0);
    int remaining = N;
    vector<double> pert(N + 1, 1.0);
    for (int v = 1; v <= N; v++) pert[v] = 1.0 + eps * urand();
    priority_queue<pair<double,int>> pq;
    for (int v = 1; v <= N; v++)
        pq.push({(double)ball[v].size() / (cost[v] * pert[v]), v});
    vector<char> chosen(N + 1, 0);
    vector<int> sol;
    while (remaining > 0 && !pq.empty()) {
        auto [key, v] = pq.top(); pq.pop();
        if (chosen[v]) continue;
        int real = 0;
        for (int u : ball[v]) if (!covered[u]) real++;
        if (real == 0) continue;
        double rkey = (double)real / (cost[v] * pert[v]);
        if (rkey < key - 1e-12) { pq.push({rkey, v}); continue; }
        chosen[v] = 1; sol.push_back(v);
        for (int u : ball[v]) if (!covered[u]) { covered[u] = 1; remaining--; }
    }
    return sol;
}

// drop redundant sensors, most expensive first.
vector<int> prune(vector<int> sol) {
    vector<int> cover(N + 1, 0);
    for (int v : sol) for (int u : ball[v]) cover[u]++;
    sort(sol.begin(), sol.end(), [&](int a, int b){ return cost[a] > cost[b]; });
    vector<char> keep(N + 1, 0);
    for (int v : sol) keep[v] = 1;
    for (int v : sol) {
        bool redundant = true;
        for (int u : ball[v]) if (cover[u] <= 1) { redundant = false; break; }
        if (redundant) { keep[v] = 0; for (int u : ball[v]) cover[u]--; }
    }
    vector<int> out;
    for (int v : sol) if (keep[v]) out.push_back(v);
    return out;
}

ll totalCost(const vector<int>& s) { ll c = 0; for (int v : s) c += cost[v]; return c; }

int main() {
    scanf("%d %d %lld", &N, &M, &R);
    cost.assign(N + 1, 0);
    for (int v = 1; v <= N; v++) scanf("%lld", &cost[v]);
    g.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }
    buildBalls();

    vector<int> best = prune(greedyRun(0.0));
    ll bestC = totalCost(best);
    int restarts = N <= 200 ? 24 : (N <= 2000 ? 12 : 6);
    for (int r = 0; r < restarts; r++) {
        vector<int> cand = prune(greedyRun(0.35));
        ll c = totalCost(cand);
        if (c < bestC) { bestC = c; best = cand; }
    }
    printf("%d\n", (int)best.size());
    for (int v : best) printf("%d\n", v);
    return 0;
}
