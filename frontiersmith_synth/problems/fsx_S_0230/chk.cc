#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Alpine Lift Network Coverage -- checker / scorer.
// Reads instance from inf, participant plan from ouf. Validates feasibility strictly,
// then scores  ratio = min(1.0, (B/F)/10)  where B = build-everywhere cost.

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    // ---- read instance ----
    int N = inf.readInt();
    int M = inf.readInt();
    long long R = inf.readLong();

    vector<long long> cost(N + 1);
    long long B = 0;
    for (int i = 1; i <= N; i++) {
        cost[i] = inf.readLong();
        B += cost[i];
    }
    // adjacency
    vector<vector<pair<int,long long>>> adj(N + 1);
    for (int e = 0; e < M; e++) {
        int u = inf.readInt();
        int v = inf.readInt();
        long long w = inf.readLong();
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // ---- read participant plan ----
    int K = ouf.readInt(1, N, "K");
    vector<int> chosen(K);
    vector<char> picked(N + 1, 0);
    for (int i = 0; i < K; i++) {
        int s = ouf.readInt(1, N, "station");
        if (picked[s]) quitf(_wa, "station %d listed more than once", s);
        picked[s] = 1;
        chosen[i] = s;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing data after plan");

    // ---- feasibility: every zone served (d(zone, some chosen) <= R) ----
    // multi-source Dijkstra from all chosen stations at once.
    const long long INF = (long long)4e18;
    vector<long long> dist(N + 1, INF);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    for (int s : chosen) { dist[s] = 0; pq.push({0, s}); }
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        if (d > R) continue; // no need to expand beyond radius
        for (auto [v, w] : adj[u]) {
            long long nd = d + w;
            if (nd < dist[v]) { dist[v] = nd; pq.push({nd, v}); }
        }
    }
    for (int i = 1; i <= N; i++) {
        if (dist[i] > R) quitf(_wa, "zone %d is not served (nearest station distance %lld > R=%lld)",
                               i, dist[i] >= INF ? -1 : dist[i], R);
    }

    // ---- objective ----
    long long F = 0;
    for (int s : chosen) F += cost[s];
    if (F < 1) F = 1;
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
