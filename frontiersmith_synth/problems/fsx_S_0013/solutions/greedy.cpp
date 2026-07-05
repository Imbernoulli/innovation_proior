// TIER: greedy
// Most-used-link greedy: each round, compute one shortest path per demand, accumulate
// priority-weighted "usage" on each edge, and unplug the highest-usage link whose removal
// keeps every demand pair connected. Repeat up to k times. One-pass heuristic (does NOT
// directly evaluate the post-removal objective), so it diverges from best-improvement.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct AdjE { int to; ll w; int idx; };
int n, m, P, k;
vector<vector<AdjE>> g;
vector<int> src, dst, pr;

// Dijkstra with predecessor EDGE recorded (for path reconstruction).
void dijkstra(int s, const vector<char>& removed, vector<ll>& dist, vector<int>& predEdge, vector<int>& predNode) {
    dist.assign(n + 1, LLONG_MAX);
    predEdge.assign(n + 1, -1);
    predNode.assign(n + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) {
                dist[e.to] = nd; predEdge[e.to] = e.idx; predNode[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }
}

bool allConnected(const vector<char>& removed) {
    // group by source; BFS reachability suffices for connectivity
    map<int, vector<int>> bySrc;
    for (int i = 0; i < P; i++) bySrc[src[i]].push_back(i);
    for (auto& kv : bySrc) {
        vector<char> vis(n + 1, 0);
        queue<int> q; q.push(kv.first); vis[kv.first] = 1;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& e : g[u]) if (!removed[e.idx] && !vis[e.to]) { vis[e.to] = 1; q.push(e.to); }
        }
        for (int i : kv.second) if (!vis[dst[i]]) return false;
    }
    return true;
}

int main() {
    if (scanf("%d %d %d %d", &n, &m, &P, &k) != 4) return 0;
    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w, i}); g[v].push_back({u, w, i});
    }
    src.resize(P); dst.resize(P); pr.resize(P);
    for (int i = 0; i < P; i++) scanf("%d %d %d", &src[i], &dst[i], &pr[i]);

    vector<char> removed(m + 1, 0);
    vector<int> chosen;

    for (int step = 0; step < k; step++) {
        // accumulate weighted usage of each edge over one shortest path per demand
        vector<ll> usage(m + 1, 0);
        map<int, vector<int>> bySrc;
        for (int i = 0; i < P; i++) bySrc[src[i]].push_back(i);
        vector<ll> dist; vector<int> predE, predN;
        for (auto& kv : bySrc) {
            dijkstra(kv.first, removed, dist, predE, predN);
            for (int i : kv.second) {
                int cur = dst[i];
                if (dist[cur] == LLONG_MAX) continue;
                while (predE[cur] != -1) {
                    usage[predE[cur]] += pr[i];
                    cur = predN[cur];
                }
            }
        }
        // candidates sorted by usage desc
        vector<int> cand;
        for (int e = 1; e <= m; e++) if (!removed[e] && usage[e] > 0) cand.push_back(e);
        sort(cand.begin(), cand.end(), [&](int a, int b){ return usage[a] > usage[b]; });

        int pick = -1;
        for (int e : cand) {
            removed[e] = 1;
            if (allConnected(removed)) { pick = e; break; }
            removed[e] = 0;
        }
        if (pick == -1) break;
        chosen.push_back(pick);
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
