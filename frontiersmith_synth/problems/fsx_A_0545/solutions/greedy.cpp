// TIER: greedy
// Short-cycle greedy: enumerate profitable 2-cycles and 3-cycles only. Blind to the
// long multi-hop planted loops that dominate the objective.



#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const int DMAX = 2;   // 2-cycles and 3-cycles only (the obvious short-cycle greedy)

int N, M;
vector<int> eu, ev;
vector<ll> ecap, erev, residual, flowUsed;
vector<vector<int>> adj;     // positive-rev outgoing pipes
vector<char> onpath;
vector<int> path, touched;

bool dfs(int cur, int depth, int s) {
    if (depth > DMAX) return false;
    for (int e : adj[cur]) {
        if (residual[e] <= 0) continue;
        int w = ev[e];
        if (w == s) {
            if (depth >= 1) {
                path.push_back(e);
                ll fmin = LLONG_MAX;
                for (int pe : path) fmin = min(fmin, residual[pe]);
                for (int pe : path) { residual[pe] -= fmin; flowUsed[pe] += fmin; }
                path.pop_back();
                return true;
            }
            continue;
        }
        if (onpath[w]) continue;
        onpath[w] = 1; path.push_back(e); touched.push_back(w);
        if (dfs(w, depth + 1, s)) { path.pop_back(); return true; }
        path.pop_back(); onpath[w] = 0;
    }
    return false;
}

int main() {
    if (scanf("%d %d", &N, &M) != 2) return 0;
    eu.assign(M + 1, 0); ev.assign(M + 1, 0);
    ecap.assign(M + 1, 0); erev.assign(M + 1, 0);
    residual.assign(M + 1, 0); flowUsed.assign(M + 1, 0);
    adj.assign(N + 1, {}); onpath.assign(N + 1, 0);
    for (int e = 1; e <= M; e++) {
        scanf("%d %d %lld %lld", &eu[e], &ev[e], &ecap[e], &erev[e]);
        residual[e] = ecap[e];
        if (erev[e] > 0) adj[eu[e]].push_back(e);
    }
    for (int s = 1; s <= N; s++) {
        if (adj[s].empty()) continue;
        while (true) {
            for (int t : touched) onpath[t] = 0;
            touched.clear(); path.clear();
            onpath[s] = 1;
            bool got = dfs(s, 0, s);
            onpath[s] = 0;
            if (!got) break;
        }
    }
    int K = 0;
    for (int e = 1; e <= M; e++) if (flowUsed[e] > 0) K++;
    printf("%d\n", K);
    for (int e = 1; e <= M; e++) if (flowUsed[e] > 0) printf("%d %lld\n", e, flowUsed[e]);
    return 0;
}
