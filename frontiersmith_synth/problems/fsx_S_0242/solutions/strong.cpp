// TIER: strong
// Cost-ratio greedy for min-cost R-multi-cover, then redundancy pruning:
// drop the most expensive lamps whose removal still leaves every demand met.
#include <bits/stdc++.h>
using namespace std;

int N, M, R;
vector<vector<int>> adj;

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

    vector<vector<int>> balls(N + 1);
    {
        vector<int> stamp(N + 1, 0), dist(N + 1, 0);
        for (int v = 1; v <= N; v++) balls[v] = ball(v, stamp, dist, v);
    }

    // --- Phase 1: cost-ratio greedy ---
    vector<int> rem(N + 1);
    for (int i = 1; i <= N; i++) rem[i] = d[i];
    vector<char> built(N + 1, 0);
    long long outstanding = 0;
    for (int i = 1; i <= N; i++) outstanding += rem[i];

    while (outstanding > 0) {
        int best = -1;
        double bestScore = -1.0;
        for (int v = 1; v <= N; v++) {
            if (built[v]) continue;
            int benefit = 0;
            for (int u : balls[v]) if (rem[u] > 0) benefit++;
            if (benefit == 0) continue;
            double sc = (double)benefit / (double)c[v];
            if (sc > bestScore) { bestScore = sc; best = v; }
        }
        if (best == -1) break;
        built[best] = 1;
        for (int u : balls[best]) if (rem[u] > 0) { rem[u]--; outstanding--; }
    }

    // --- Phase 2: redundancy pruning ---
    // Coverage counts over the built set.
    vector<int> cov(N + 1, 0);
    vector<int> chosen;
    for (int v = 1; v <= N; v++) if (built[v]) { chosen.push_back(v); for (int u : balls[v]) cov[u]++; }
    // Try removing the most expensive lamps first.
    sort(chosen.begin(), chosen.end(), [&](int a, int b) { return c[a] > c[b]; });
    for (int v : chosen) {
        if (!built[v]) continue;
        bool removable = true;
        for (int u : balls[v]) if (cov[u] - 1 < d[u]) { removable = false; break; }
        if (removable) {
            built[v] = 0;
            for (int u : balls[v]) cov[u]--;
        }
    }

    vector<int> out;
    for (int v = 1; v <= N; v++) if (built[v]) out.push_back(v);
    printf("%d\n", (int)out.size());
    for (size_t i = 0; i < out.size(); i++)
        printf("%d%c", out[i], i + 1 == out.size() ? '\n' : ' ');
    if (out.empty()) printf("\n");
    return 0;
}
