// TIER: greedy
// Cost-effectiveness greedy set cover: repeatedly build the ladder that covers the most
// still-uncovered spawning grounds per unit installation cost.
#include <bits/stdc++.h>
using namespace std;

int N, M, D;
vector<vector<int>> adj;
vector<int> c, r, dem;
vector<int> demIdx;   // node -> index in dem (or -1)

// for node v, list of spawning-ground indices covered by a ladder at v
vector<vector<int>> ballDem;

void computeBalls() {
    ballDem.assign(N + 1, {});
    vector<int> dist(N + 1, -1);
    vector<int> stamp(N + 1, -1);
    for (int s = 1; s <= N; s++) {
        queue<int> q;
        dist[s] = 0; stamp[s] = s; q.push(s);
        while (!q.empty()) {
            int x = q.front(); q.pop();
            if (demIdx[x] >= 0) ballDem[s].push_back(demIdx[x]);
            if (dist[x] == r[s]) continue;
            for (int y : adj[x]) if (stamp[y] != s) {
                stamp[y] = s; dist[y] = dist[x] + 1; q.push(y);
            }
        }
    }
}

int main() {
    scanf("%d %d", &N, &M);
    adj.assign(N + 1, {});
    for (int i = 0; i < M; i++) { int u, v; scanf("%d %d", &u, &v); adj[u].push_back(v); adj[v].push_back(u); }
    c.assign(N + 1, 0); r.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%d", &c[i]);
    for (int i = 1; i <= N; i++) scanf("%d", &r[i]);
    scanf("%d", &D);
    dem.assign(D, 0); demIdx.assign(N + 1, -1);
    for (int i = 0; i < D; i++) { scanf("%d", &dem[i]); demIdx[dem[i]] = i; }

    computeBalls();

    vector<char> covered(D, 0);
    int remaining = D;
    vector<int> chosen;
    vector<char> built(N + 1, 0);

    while (remaining > 0) {
        int best = -1;
        double bestEff = -1.0;   // newcov / cost, maximize
        for (int v = 1; v <= N; v++) {
            if (built[v]) continue;
            int nc = 0;
            for (int di : ballDem[v]) if (!covered[di]) nc++;
            if (nc == 0) continue;
            double eff = (double)nc / (double)c[v];
            if (eff > bestEff + 1e-12) { bestEff = eff; best = v; }
        }
        if (best < 0) break;   // should not happen: every demand covers itself
        built[best] = 1;
        chosen.push_back(best);
        for (int di : ballDem[best]) if (!covered[di]) { covered[di] = 1; remaining--; }
    }

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    if (chosen.empty()) printf("\n");
    return 0;
}
