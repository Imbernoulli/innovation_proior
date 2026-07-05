// TIER: strong
// Randomized multi-start cost-effectiveness greedy + redundancy pruning.
// Runs several greedy passes with perturbed tie-breaking, prunes redundant stations
// from each, and keeps the cheapest valid plan found. Deterministic (fixed seed).
#include <bits/stdc++.h>
using namespace std;

int N, M, W; long long R;
vector<long long> cost;
vector<vector<uint64_t>> cover;

static inline long long popAnd(const vector<uint64_t>& a, const vector<uint64_t>& b) {
    long long c = 0; for (int k = 0; k < W; k++) c += __builtin_popcountll(a[k] & b[k]); return c;
}

int main() {
    scanf("%d %d %lld", &N, &M, &R);
    cost.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%lld", &cost[i]);
    vector<vector<pair<int,long long>>> adj(N + 1);
    for (int e = 0; e < M; e++) {
        int u, v; long long w; scanf("%d %d %lld", &u, &v, &w);
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
    }
    W = (N + 64) / 64;
    cover.assign(N + 1, vector<uint64_t>(W, 0));
    const long long INF = (long long)4e18;
    for (int j = 1; j <= N; j++) {
        vector<long long> dist(N + 1, INF);
        priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
        dist[j] = 0; pq.push({0, j});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d != dist[u]) continue;
            if (d > R) continue;
            for (auto [v, w] : adj[u]) {
                long long nd = d + w;
                if (nd < dist[v]) { dist[v] = nd; pq.push({nd, v}); }
            }
        }
        for (int i = 1; i <= N; i++)
            if (dist[i] <= R) cover[j][(i - 1) >> 6] |= (1ULL << ((i - 1) & 63));
    }

    vector<uint64_t> full(W, 0);
    for (int i = 0; i < N; i++) full[i >> 6] |= (1ULL << (i & 63));
    auto anyBits = [&](const vector<uint64_t>& b) {
        for (auto x : b) if (x) return true; return false;
    };

    mt19937_64 rng(12345);

    // one greedy pass; alpha in [0,1] adds a random multiplicative jitter to the
    // cost-effectiveness score to diversify the choices across restarts.
    auto greedyPass = [&](double jitter) -> vector<int> {
        vector<uint64_t> unc = full;
        vector<char> used(N + 1, 0);
        vector<int> chosen;
        while (anyBits(unc)) {
            int best = -1; double bestScore = -1;
            for (int j = 1; j <= N; j++) {
                if (used[j]) continue;
                long long nc = popAnd(cover[j], unc);
                if (nc == 0) continue;
                double s = (double)nc / (double)cost[j];
                if (jitter > 0) {
                    double f = 1.0 + jitter * (((double)(rng() % 1000) / 1000.0) - 0.5);
                    s *= f;
                }
                if (s > bestScore || (s == bestScore && best != -1 && cost[j] < cost[best])) {
                    bestScore = s; best = j;
                }
            }
            if (best == -1) break;
            used[best] = 1; chosen.push_back(best);
            for (int k = 0; k < W; k++) unc[k] &= ~cover[best][k];
        }
        return chosen;
    };

    // redundancy pruning: remove the most expensive removable station while the plan
    // still covers everything.
    auto prune = [&](vector<int> plan) -> vector<int> {
        bool changed = true;
        while (changed) {
            changed = false;
            // try removing in order of decreasing cost
            sort(plan.begin(), plan.end(), [&](int a, int b){ return cost[a] > cost[b]; });
            for (size_t idx = 0; idx < plan.size(); idx++) {
                // coverage without plan[idx]
                vector<uint64_t> covd(W, 0);
                for (size_t t = 0; t < plan.size(); t++) {
                    if (t == idx) continue;
                    for (int k = 0; k < W; k++) covd[k] |= cover[plan[t]][k];
                }
                bool ok = true;
                for (int k = 0; k < W; k++) if ((covd[k] & full[k]) != full[k]) { ok = false; break; }
                if (ok) {
                    plan.erase(plan.begin() + idx);
                    changed = true;
                    break;
                }
            }
        }
        return plan;
    };

    auto planCost = [&](const vector<int>& p) {
        long long c = 0; for (int x : p) c += cost[x]; return c;
    };

    // deterministic first pass (no jitter), then randomized restarts.
    vector<int> best = prune(greedyPass(0.0));
    long long bestC = planCost(best);

    int restarts = 40;
    if (N > 120) restarts = 20;
    for (int r = 0; r < restarts; r++) {
        double jit = 0.2 + 0.6 * ((double)(r % 5) / 4.0);
        vector<int> cand = prune(greedyPass(jit));
        long long c = planCost(cand);
        if (c < bestC) { bestC = c; best = cand; }
    }

    printf("%d\n", (int)best.size());
    for (size_t i = 0; i < best.size(); i++)
        printf("%d%c", best[i], i + 1 == best.size() ? '\n' : ' ');
    return 0;
}
