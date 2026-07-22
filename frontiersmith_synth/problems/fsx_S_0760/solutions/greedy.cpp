// TIER: greedy
// The obvious "textbook" approach: treat every channel's cost as fixed at its FULLY
// REFROZEN (virgin) value f + C^1.5, and run an independent Dijkstra shortest path for
// each convoy, starting the instant it is ready. This never looks at what any other
// convoy is doing, so it can never discover that a channel another convoy just broke is
// temporarily cheap -- it always treats the ice as a static per-edge cost, never a shared
// decaying asset. This is the trap the problem is built around.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    vector<int> eu(M + 1), ev(M + 1), eL(M + 1), eC(M + 1), er(M + 1), ef(M + 1);
    vector<vector<pair<int,int>>> adj(N + 1); // node -> (neighbor, edgeId)
    for (int i = 1; i <= M; i++) {
        scanf("%d %d %d %d %d %d", &eu[i], &ev[i], &eL[i], &eC[i], &er[i], &ef[i]);
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }
    vector<int> co(K + 1), cd(K + 1), cready(K + 1);
    for (int i = 1; i <= K; i++) scanf("%d %d %d", &co[i], &cd[i], &cready[i]);

    vector<long long> vw(M + 1);
    for (int i = 1; i <= M; i++) vw[i] = (long long)ef[i] + (long long)llround(pow((double)eC[i], 1.5));

    vector<long long> dist(N + 1);
    vector<int> parentEdge(N + 1), parentNode(N + 1);
    for (int i = 1; i <= K; i++) {
        fill(dist.begin(), dist.end(), LLONG_MAX);
        priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
        dist[co[i]] = 0;
        pq.push({0, co[i]});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du > dist[u]) continue;
            if (u == cd[i]) break;
            for (auto& pr : adj[u]) {
                int v = pr.first, eid = pr.second;
                long long nd = du + vw[eid];
                if (nd < dist[v]) {
                    dist[v] = nd;
                    parentEdge[v] = eid;
                    parentNode[v] = u;
                    pq.push({nd, v});
                }
            }
        }
        vector<int> path;
        int cur = cd[i];
        while (cur != co[i]) {
            path.push_back(parentEdge[cur]);
            cur = parentNode[cur];
        }
        reverse(path.begin(), path.end());
        printf("%d %d", cready[i], (int)path.size());
        for (int eid : path) printf(" %d", eid);
        printf("\n");
    }
    return 0;
}
