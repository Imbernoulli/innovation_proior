// TIER: greedy
// Cost-effective greedy set cover: repeatedly add the station that serves the most
// still-uncovered zones per unit cost. One pass, no pruning.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M; long long R;
    scanf("%d %d %lld", &N, &M, &R);
    vector<long long> cost(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &cost[i]);
    vector<vector<pair<int,long long>>> adj(N + 1);
    for (int e = 0; e < M; e++) {
        int u, v; long long w; scanf("%d %d %lld", &u, &v, &w);
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
    }
    int W = (N + 64) / 64;
    // coverage bitsets: cover[j] = zones served by a station at j
    vector<vector<uint64_t>> cover(N + 1, vector<uint64_t>(W, 0));
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

    // greedy
    vector<uint64_t> unc(W, 0);
    for (int i = 0; i < N; i++) unc[i >> 6] |= (1ULL << (i & 63));
    auto anyBits = [&](const vector<uint64_t>& b) {
        for (auto x : b) if (x) return true; return false;
    };
    auto popAnd = [&](const vector<uint64_t>& a, const vector<uint64_t>& b) {
        long long c = 0; for (int k = 0; k < W; k++) c += __builtin_popcountll(a[k] & b[k]); return c;
    };
    vector<int> chosen;
    vector<char> used(N + 1, 0);
    while (anyBits(unc)) {
        int best = -1; double bestScore = -1;
        for (int j = 1; j <= N; j++) {
            if (used[j]) continue;
            long long nc = popAnd(cover[j], unc);
            if (nc == 0) continue;
            double s = (double)nc / (double)cost[j];
            if (s > bestScore || (s == bestScore && best != -1 && cost[j] < cost[best])) {
                bestScore = s; best = j;
            }
        }
        if (best == -1) break; // should not happen (self-coverage guarantees progress)
        used[best] = 1; chosen.push_back(best);
        for (int k = 0; k < W; k++) unc[k] &= ~cover[best][k];
    }

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    return 0;
}
