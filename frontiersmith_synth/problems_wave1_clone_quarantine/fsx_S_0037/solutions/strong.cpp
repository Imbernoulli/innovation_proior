// TIER: strong
// Best-improvement local search with a knapsack flavor: at each step, among all
// affordable walkways on the current shortest stroll that keep the finale reachable,
// tentatively close each, recompute the TRUE post-closure shortest stroll, and commit
// the closure maximizing delay-gain per barricade cost (ties -> larger absolute gain).
// A short randomized multi-restart (fixed seed -> deterministic) keeps the best set.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t;
ll P;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;
vector<ll> ecost;

ll dijkstraPath(const vector<char>& removed, vector<int>& pathEdges) {
    vector<ll> dist(n + 1, LLONG_MAX);
    vector<int> preEdge(n + 1, -1), preNode(n + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) {
                dist[e.to] = nd; preEdge[e.to] = e.idx; preNode[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }
    pathEdges.clear();
    if (dist[t] == LLONG_MAX) return -1;
    int cur = t;
    while (cur != s) { pathEdges.push_back(preEdge[cur]); cur = preNode[cur]; }
    return dist[t];
}

ll shortest(const vector<char>& removed) {
    vector<int> pe;
    return dijkstraPath(removed, pe);
}

int main() {
    int PP;
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &PP) != 5) return 0;
    P = PP;
    g.assign(n + 1, {});
    ecost.assign(m + 1, 0);
    for (int i = 1; i <= m; i++) {
        int u, v, w, c; scanf("%d %d %d %d", &u, &v, &w, &c);
        ecost[i] = c;
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
    }

    auto runOnce = [&](unsigned seed, vector<int>& outSet) -> ll {
        mt19937 rng(seed);
        vector<char> removed(m + 1, 0);
        ll budget = P;
        vector<int> chosen;
        while (true) {
            vector<int> pe;
            ll cur = dijkstraPath(removed, pe);
            if (cur == -1) break;
            double bestScore = -1e18; int best = -1; ll bestGain = -1;
            // small deterministic perturbation of evaluation order per restart
            shuffle(pe.begin(), pe.end(), rng);
            for (int idx : pe) {
                if (removed[idx]) continue;
                if (ecost[idx] > budget) continue;
                removed[idx] = 1;
                ll nf = shortest(removed);
                removed[idx] = 0;
                if (nf == -1) continue;          // would disconnect
                ll gain = nf - cur;
                if (gain <= 0) continue;
                double score = (double)gain / (double)ecost[idx];
                if (score > bestScore + 1e-9 ||
                    (fabs(score - bestScore) <= 1e-9 && gain > bestGain)) {
                    bestScore = score; best = idx; bestGain = gain;
                }
            }
            if (best == -1) break;
            removed[best] = 1; budget -= ecost[best]; chosen.push_back(best);
        }
        outSet = chosen;
        return shortest(removed);
    };

    ll bestF = -1; vector<int> bestSet;
    for (int r = 0; r < 6; r++) {
        vector<int> setr;
        ll f = runOnce(12345u + 7919u * r, setr);
        if (f > bestF) { bestF = f; bestSet = setr; }
    }

    printf("%d\n", (int)bestSet.size());
    for (int idx : bestSet) printf("%d\n", idx);
    return 0;
}
