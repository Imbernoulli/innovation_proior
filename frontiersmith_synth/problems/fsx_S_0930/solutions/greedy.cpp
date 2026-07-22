// TIER: greedy
// The obvious first attempt: route every source's flow along a single
// resistance-weighted shortest path to the nearest unmet sink (never splitting
// across parallel pipes), then "fix the leakiest pipes" -- boost the K
// booster-ready pipes with the highest RAW resistance, without ever checking
// how much current actually flows through them.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int V, E, S, T, K;
vector<int> eu, ev;
vector<ll> er, egain;
vector<int> ecand;
vector<vector<pair<int,int>>> adj; // node -> (neighbor, edgeIdx)

// Dijkstra shortest path weighted by pipe resistance r_e (single path, minimize
// total resistance along the route -- ignores the quadratic/splitting benefit).
bool dijkstraPath(int s, int t, vector<int>& outPath) {
    vector<ll> dist(V + 1, LLONG_MAX);
    vector<int> parentEdge(V + 1, -1), parentNode(V + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (u == t) break;
        for (auto &pr : adj[u]) {
            int v = pr.first, ei = pr.second;
            ll nd = d + er[ei];
            if (nd < dist[v]) { dist[v] = nd; parentNode[v] = u; parentEdge[v] = ei; pq.push({nd, v}); }
        }
    }
    if (dist[t] == LLONG_MAX) return false;
    vector<int> path;
    int cur = t;
    while (cur != s) {
        int ei = parentEdge[cur];
        if (ei < 0) return false;
        path.push_back(ei);
        cur = parentNode[cur];
    }
    reverse(path.begin(), path.end());
    outPath = path;
    return true;
}

int main() {
    scanf("%d %d %d %d %d", &V, &E, &S, &T, &K);
    vector<pair<int,ll>> srcs(S), sinks(T);
    for (int i = 0; i < S; i++) scanf("%d %lld", &srcs[i].first, &srcs[i].second);
    for (int i = 0; i < T; i++) scanf("%d %lld", &sinks[i].first, &sinks[i].second);

    eu.assign(E, 0); ev.assign(E, 0); er.assign(E, 0); ecand.assign(E, 0); egain.assign(E, 0);
    adj.assign(V + 1, {});
    for (int e = 0; e < E; e++) {
        scanf("%d %d %lld %d %lld", &eu[e], &ev[e], &er[e], &ecand[e], &egain[e]);
        adj[eu[e]].push_back({ev[e], e});
        adj[ev[e]].push_back({eu[e], e});
    }

    sort(srcs.begin(), srcs.end());
    vector<double> flow(E, 0.0);
    vector<ll> remSrc(S), remSink(T);
    for (int i = 0; i < S; i++) remSrc[i] = srcs[i].second;
    for (int j = 0; j < T; j++) remSink[j] = sinks[j].second;

    int guard = 0;
    for (int i = 0; i < S; i++) {
        while (remSrc[i] > 0) {
            if (++guard > 10000) break;
            int bestJ = -1; ll bestDist = LLONG_MAX;
            for (int j = 0; j < T; j++) {
                if (remSink[j] <= 0) continue;
                vector<int> p;
                if (!dijkstraPath(srcs[i].first, sinks[j].first, p)) continue;
                ll d = 0; for (int ei : p) d += er[ei];
                if (d < bestDist || (d == bestDist && (bestJ == -1 || sinks[j].first < sinks[bestJ].first))) {
                    bestDist = d; bestJ = j;
                }
            }
            if (bestJ < 0) break;
            vector<int> path;
            if (!dijkstraPath(srcs[i].first, sinks[bestJ].first, path)) break;
            ll amt = min(remSrc[i], remSink[bestJ]);
            int cur = srcs[i].first;
            for (int ei : path) {
                if (eu[ei] == cur) { flow[ei] += (double)amt; cur = ev[ei]; }
                else                { flow[ei] -= (double)amt; cur = eu[ei]; }
            }
            remSrc[i] -= amt; remSink[bestJ] -= amt;
        }
    }

    // naive booster choice: rank candidates by raw resistance r_e, ignore current
    vector<int> cands;
    for (int e = 0; e < E; e++) if (ecand[e]) cands.push_back(e);
    sort(cands.begin(), cands.end(), [&](int a, int b) { return er[a] > er[b]; });
    int M = min((int)cands.size(), K);

    printf("%d", M);
    for (int i = 0; i < M; i++) printf(" %d", cands[i] + 1);
    for (int e = 0; e < E; e++) printf(" %.6f", flow[e]);
    printf("\n");
    return 0;
}
