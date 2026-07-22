// TIER: trivial
// BFS fewest-hop path per convoy (ties broken by ascending edge id -- identical
// construction to the checker's own baseline B), starting the instant the convoy is
// ready. Ignores fuel fees, thickness, and all reuse entirely.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    vector<int> eu(M + 1), ev(M + 1), eL(M + 1), eC(M + 1), er(M + 1), ef(M + 1);
    vector<vector<pair<int,int>>> adj(N + 1);
    for (int i = 1; i <= M; i++) {
        scanf("%d %d %d %d %d %d", &eu[i], &ev[i], &eL[i], &eC[i], &er[i], &ef[i]);
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }
    vector<int> co(K + 1), cd(K + 1), cready(K + 1);
    for (int i = 1; i <= K; i++) scanf("%d %d %d", &co[i], &cd[i], &cready[i]);

    vector<int> distHop(N + 1), parentEdge(N + 1), parentNode(N + 1);
    for (int i = 1; i <= K; i++) {
        fill(distHop.begin(), distHop.end(), -1);
        queue<int> q;
        distHop[co[i]] = 0;
        q.push(co[i]);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u == cd[i]) break;
            for (auto& pr : adj[u]) {
                int v = pr.first, eid = pr.second;
                if (distHop[v] == -1) {
                    distHop[v] = distHop[u] + 1;
                    parentEdge[v] = eid;
                    parentNode[v] = u;
                    q.push(v);
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
