// TIER: trivial
// The reference recipe: route each source's flow to the nearest unmet sink along
// ONE fixed shortest-hop path (fewest pipes), never splitting across parallel
// pipes, never using a booster. Reproduces the checker's own baseline B exactly.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int V, E, S, T, K;
vector<int> eu, ev;
vector<ll> er, egain;
vector<int> ecand;
vector<vector<pair<int,int>>> adj;

bool bfsPath(int s, int t, vector<int>& outPath) {
    vector<int> parentEdge(V + 1, -1), parentNode(V + 1, -1);
    vector<char> vis(V + 1, 0);
    queue<int> q;
    vis[s] = 1; q.push(s);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (u == t) break;
        for (auto &pr : adj[u]) {
            int v = pr.first, ei = pr.second;
            if (!vis[v]) { vis[v] = 1; parentNode[v] = u; parentEdge[v] = ei; q.push(v); }
        }
    }
    if (!vis[t]) return false;
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
    for (int v = 1; v <= V; v++) sort(adj[v].begin(), adj[v].end());

    sort(srcs.begin(), srcs.end());
    vector<double> flow(E, 0.0);
    vector<ll> remSrc(S), remSink(T);
    for (int i = 0; i < S; i++) remSrc[i] = srcs[i].second;
    for (int j = 0; j < T; j++) remSink[j] = sinks[j].second;

    int guard = 0;
    for (int i = 0; i < S; i++) {
        while (remSrc[i] > 0) {
            if (++guard > 10000) break;
            int bestJ = -1, bestDist = INT_MAX;
            for (int j = 0; j < T; j++) {
                if (remSink[j] <= 0) continue;
                vector<int> p;
                if (!bfsPath(srcs[i].first, sinks[j].first, p)) continue;
                int dist = (int)p.size();
                if (dist < bestDist || (dist == bestDist && (bestJ == -1 || sinks[j].first < sinks[bestJ].first))) {
                    bestDist = dist; bestJ = j;
                }
            }
            if (bestJ < 0) break;
            vector<int> path;
            if (!bfsPath(srcs[i].first, sinks[bestJ].first, path)) break;
            ll amt = min(remSrc[i], remSink[bestJ]);
            int cur = srcs[i].first;
            for (int ei : path) {
                if (eu[ei] == cur) { flow[ei] += (double)amt; cur = ev[ei]; }
                else                { flow[ei] -= (double)amt; cur = eu[ei]; }
            }
            remSrc[i] -= amt; remSink[bestJ] -= amt;
        }
    }

    printf("0");
    for (int e = 0; e < E; e++) printf(" %.6f", flow[e]);
    printf("\n");
    return 0;
}
