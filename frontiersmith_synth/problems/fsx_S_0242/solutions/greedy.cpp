// TIER: greedy
// Cost-ratio greedy for min-cost R-multi-cover.
// Repeatedly build the lamp maximizing (newly-satisfied demand units) / cost.
#include <bits/stdc++.h>
using namespace std;

int N, M, R;
vector<vector<int>> adj;

// Compute R-ball of v (vertices within R hops, inclusive).
vector<int> ball(int v, vector<int>& stamp, vector<int>& dist, int tok) {
    vector<int> res;
    queue<int> q; q.push(v); stamp[v] = tok; dist[v] = 0; res.push_back(v);
    while (!q.empty()) {
        int x = q.front(); q.pop();
        if (dist[x] == R) continue;
        for (int y : adj[x]) if (stamp[y] != tok) {
            stamp[y] = tok; dist[y] = dist[x] + 1; res.push_back(y); q.push(y);
        }
    }
    return res;
}

int main() {
    if (scanf("%d %d %d", &N, &M, &R) != 3) return 0;
    vector<long long> c(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &c[i]);
    vector<int> d(N + 1);
    for (int i = 1; i <= N; i++) scanf("%d", &d[i]);
    adj.assign(N + 1, {});
    for (int e = 0; e < M; e++) { int u, v; scanf("%d %d", &u, &v); adj[u].push_back(v); adj[v].push_back(u); }

    // Precompute each vertex's R-ball once.
    vector<vector<int>> balls(N + 1);
    {
        vector<int> stamp(N + 1, 0), dist(N + 1, 0);
        for (int v = 1; v <= N; v++) balls[v] = ball(v, stamp, dist, v);
    }

    vector<int> rem(N + 1);          // remaining unmet demand per vertex
    for (int i = 1; i <= N; i++) rem[i] = d[i];
    vector<char> built(N + 1, 0);
    long long outstanding = 0;
    for (int i = 1; i <= N; i++) outstanding += rem[i];

    vector<int> chosen;
    while (outstanding > 0) {
        int best = -1;
        double bestScore = -1.0;
        int bestBenefit = 0;
        for (int v = 1; v <= N; v++) {
            if (built[v]) continue;
            int benefit = 0;
            for (int u : balls[v]) if (rem[u] > 0) benefit++;
            if (benefit == 0) continue;
            double sc = (double)benefit / (double)c[v];
            if (sc > bestScore) { bestScore = sc; best = v; bestBenefit = benefit; }
        }
        if (best == -1) break; // should not happen (feasible by construction)
        built[best] = 1;
        chosen.push_back(best);
        for (int u : balls[best]) if (rem[u] > 0) { rem[u]--; outstanding--; }
    }

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    if (chosen.empty()) printf("\n");
    return 0;
}
