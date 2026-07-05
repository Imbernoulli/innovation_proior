// TIER: strong
// Takahashi-Matsuyama shortest-path Steiner heuristic:
//   grow a tree from one hut; repeatedly multi-source-Dijkstra from the current
//   tree and attach the nearest still-disconnected hut via its shortest path.
// Then iteratively prune non-hut leaf trenches to shave dangling cost.
#include <bits/stdc++.h>
using namespace std;

int N, M, K;
vector<int> eu, ev;
vector<long long> ew;
vector<vector<pair<int,int>>> adj;

int main() {
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    eu.assign(M + 1, 0); ev.assign(M + 1, 0); ew.assign(M + 1, 0);
    adj.assign(N + 1, {});
    for (int k = 1; k <= M; k++) {
        scanf("%d %d %lld", &eu[k], &ev[k], &ew[k]);
        adj[eu[k]].push_back({ev[k], k});
        adj[ev[k]].push_back({eu[k], k});
    }
    vector<int> huts(K);
    for (int i = 0; i < K; i++) scanf("%d", &huts[i]);

    const long long INF = (long long)4e18;
    vector<char> inTree(N + 1, 0);
    vector<char> isHut(N + 1, 0);
    for (int h : huts) isHut[h] = 1;
    set<int> chosen;

    inTree[huts[0]] = 1;
    int connectedHuts = 1;

    while (connectedHuts < K) {
        // Multi-source Dijkstra from all current tree nodes.
        vector<long long> dist(N + 1, INF);
        vector<int> parEdge(N + 1, -1);
        priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
        for (int v = 1; v <= N; v++) if (inTree[v]) { dist[v] = 0; pq.push({0, v}); }
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d != dist[u]) continue;
            for (auto [v, idx] : adj[u]) {
                long long nd = d + ew[idx];
                if (nd < dist[v]) { dist[v] = nd; parEdge[v] = idx; pq.push({nd, v}); }
            }
        }
        // Nearest disconnected hut.
        int target = -1; long long best = INF;
        for (int h : huts) if (!inTree[h] && dist[h] < best) { best = dist[h]; target = h; }
        if (target == -1) break; // unreachable (shouldn't happen: backbone connects all)

        // Attach its shortest path.
        int cur = target;
        while (!inTree[cur] && parEdge[cur] != -1) {
            int idx = parEdge[cur];
            chosen.insert(idx);
            int nxt = (eu[idx] == cur) ? ev[idx] : eu[idx];
            inTree[cur] = 1;
            cur = nxt;
        }
        inTree[cur] = 1;
        // recount
        connectedHuts = 0;
        for (int h : huts) if (inTree[h]) connectedHuts++;
    }

    // Iteratively prune non-hut leaves.
    for (;;) {
        vector<int> deg(N + 1, 0);
        vector<vector<int>> incid(N + 1);
        for (int idx : chosen) { deg[eu[idx]]++; deg[ev[idx]]++; incid[eu[idx]].push_back(idx); incid[ev[idx]].push_back(idx); }
        int removed = 0;
        vector<int> toRemove;
        for (int v = 1; v <= N; v++) {
            if (!isHut[v] && deg[v] == 1) {
                // its single incident edge is a danglier
                for (int idx : incid[v]) if (chosen.count(idx)) { toRemove.push_back(idx); break; }
            }
        }
        for (int idx : toRemove) if (chosen.erase(idx)) removed++;
        if (removed == 0) break;
    }

    printf("%d\n", (int)chosen.size());
    bool first = true;
    for (int idx : chosen) { printf("%s%d", first ? "" : " ", idx); first = false; }
    printf("\n");
    return 0;
}
