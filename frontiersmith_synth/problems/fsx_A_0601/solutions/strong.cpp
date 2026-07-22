// TIER: strong
// The insight: the certifying eigenvector DRIFTS as you cut, so follow it. Delete the
// budget in small batches, recomputing the principal eigenvector of the CURRENT residual
// graph before each batch. After the dominant core is suppressed the eigenvector migrates
// to the next core, so the next batch of cuts lands there -- this water-fills suppression
// across all cores and minimizes the maximum residual core spectral radius, instead of
// over-spending on one core the way a static ranking does.
#include <bits/stdc++.h>
using namespace std;

int N;
vector<vector<int>> adj;

void rebuild(int m, const vector<int>& eu, const vector<int>& ev, const vector<char>& del) {
    for (int u = 0; u < N; u++) adj[u].clear();
    for (int i = 0; i < m; i++) if (!del[i]) { adj[eu[i]].push_back(ev[i]); adj[ev[i]].push_back(eu[i]); }
}
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
        tadj[u].push_back({v, i}); tadj[v].push_back({u, i});
    }
    // non-tree = deletable (deleting any subset of them preserves connectivity).
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
    vector<int> deletable;
    for (int i = 0; i < m; i++) if (!intree[i]) deletable.push_back(i);

    vector<char> del(m, 0);
    long long done = 0;
    int batches = 24;
    long long per = max<long long>(1, k / batches);
    while (done < k) {
        rebuild(m, eu, ev, del);
        vector<double> x = principal(130);
        // rank still-deletable edges by current x_u*x_v
        vector<int> cur;
        cur.reserve(deletable.size());
        for (int id : deletable) if (!del[id]) cur.push_back(id);
        long long take = min<long long>(per, k - done);
        if ((long long)cur.size() <= take) {
            for (int id : cur) { del[id] = 1; done++; }
        } else {
            nth_element(cur.begin(), cur.begin() + take, cur.end(), [&](int a, int b) {
                double sa = x[eu[a]] * x[ev[a]];
                double sb = x[eu[b]] * x[ev[b]];
                if (sa != sb) return sa > sb;
                return a < b;
            });
            for (long long j = 0; j < take; j++) { del[cur[j]] = 1; done++; }
        }
    }
    for (int i = 0; i < m; i++) if (del[i]) printf("%d %d\n", eu[i] + 1, ev[i] + 1);
    return 0;
}
