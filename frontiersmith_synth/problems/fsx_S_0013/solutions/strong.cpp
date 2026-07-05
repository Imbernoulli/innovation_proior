// TIER: strong
// Best-improvement local search. Each of k rounds: gather candidate links (those on some
// current shortest path), tentatively unplug each, recompute the TRUE priority-weighted
// total lag over all pairs, and commit the removal with the largest actual gain that keeps
// every pair connected. Directly optimizes the objective -> beats the usage-count greedy.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct AdjE { int to; ll w; int idx; };
int n, m, P, k;
vector<vector<AdjE>> g;
vector<int> src, dst, pr;

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

// weighted total lag; LLONG_MAX if a pair is disconnected. also returns union of path edges.
ll totalLag(const vector<char>& removed, vector<char>* onPath) {
    if (onPath) fill(onPath->begin(), onPath->end(), 0);
    map<int, vector<int>> bySrc;
    for (int i = 0; i < P; i++) bySrc[src[i]].push_back(i);
    ll F = 0;
    vector<ll> dist; vector<int> predE, predN;
    for (auto& kv : bySrc) {
        dijkstra(kv.first, removed, dist, predE, predN);
        for (int i : kv.second) {
            if (dist[dst[i]] == LLONG_MAX) return LLONG_MAX;
            F += (ll)pr[i] * dist[dst[i]];
            if (onPath) {
                int cur = dst[i];
                while (predE[cur] != -1) { (*onPath)[predE[cur]] = 1; cur = predN[cur]; }
            }
        }
    }
    return F;
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
    vector<char> onPath(m + 1, 0);
    vector<int> chosen;

    for (int step = 0; step < k; step++) {
        ll base = totalLag(removed, &onPath);
        if (base == LLONG_MAX) break; // should not happen
        // candidates = edges currently on some shortest path
        vector<int> cand;
        for (int e = 1; e <= m; e++) if (!removed[e] && onPath[e]) cand.push_back(e);
        // safety cap to bound runtime on the largest tests
        const int CAP = 160;
        if ((int)cand.size() > CAP) cand.resize(CAP);

        int bestE = -1; ll bestF = base;
        for (int e : cand) {
            removed[e] = 1;
            ll f = totalLag(removed, nullptr);
            removed[e] = 0;
            if (f != LLONG_MAX && f > bestF) { bestF = f; bestE = e; }
        }
        if (bestE == -1) break;
        removed[bestE] = 1;
        chosen.push_back(bestE);
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
