// TIER: greedy
// The obvious first-order approach (the trap): compute the ORIGINAL principal
// eigenvector once, rank deletable edges by x_u * x_v, and delete the top k.
// Deletable = non-tree edges (keeps the graph connected). Because the eigenvector
// is concentrated on the dominant core, the whole budget is poured into that core;
// once it drops below a secondary core the top eigenvalue migrates there and the
// remaining cuts are wasted -- a static ranking cannot see the drift.
#include <bits/stdc++.h>
using namespace std;

int N;
vector<vector<int>> adj;

vector<double> principal(int iters) {
    vector<double> x(N, 1.0), y(N, 0.0);
    for (int it = 0; it < iters; it++) {
        for (int u = 0; u < N; u++) y[u] = 0.0;
        for (int u = 0; u < N; u++) { double xu = x[u]; for (int v : adj[u]) y[v] += xu; }
        double nrm = 0; for (int u = 0; u < N; u++) nrm += y[u]*y[u];
        nrm = sqrt(nrm); if (nrm < 1e-12) break;
        double inv = 1.0/nrm; for (int u = 0; u < N; u++) x[u] = y[u]*inv;
    }
    return x;
}

int main() {
    int n, m; long long k;
    scanf("%d %d %lld", &n, &m, &k);
    N = n; adj.assign(N, {});
    vector<int> eu(m), ev(m);
    vector<vector<pair<int,int>>> tadj(n);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v); u--; v--;
        if (u > v) swap(u, v);
        eu[i] = u; ev[i] = v;
        adj[u].push_back(v); adj[v].push_back(u);
        tadj[u].push_back({v, i}); tadj[v].push_back({u, i});
    }
    vector<char> intree(m, 0), seen(n, 0);
    for (int s = 0; s < n; s++) {
        if (seen[s]) continue;
        seen[s] = 1; queue<int> q; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto &pr : tadj[u])
                if (!seen[pr.first]) { seen[pr.first] = 1; intree[pr.second] = 1; q.push(pr.first); }
        }
    }
    vector<double> x = principal(400);
    vector<int> cand;
    for (int i = 0; i < m; i++) if (!intree[i]) cand.push_back(i);
    sort(cand.begin(), cand.end(), [&](int a, int b) {
        double sa = x[eu[a]] * x[ev[a]];
        double sb = x[eu[b]] * x[ev[b]];
        if (sa != sb) return sa > sb;
        return a < b;
    });
    long long take = min<long long>(k, (long long)cand.size());
    for (long long j = 0; j < take; j++)
        printf("%d %d\n", eu[cand[j]] + 1, ev[cand[j]] + 1);
    return 0;
}
