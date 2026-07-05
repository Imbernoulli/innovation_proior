// TIER: strong
// Cost-effectiveness greedy: repeatedly install the regulator minimizing
// cost / (newly covered zones), then prune redundant regulators in reverse order.
#include <bits/stdc++.h>
using namespace std;

int n, m, r;
vector<vector<int>> g;

vector<int> coverOf(int src) {
    vector<int> out;
    vector<int> dist(n + 1, -1);
    queue<int> q;
    dist[src] = 0; q.push(src); out.push_back(src);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (dist[u] == r) continue;
        for (int w : g[u]) if (dist[w] == -1) {
            dist[w] = dist[u] + 1;
            out.push_back(w);
            q.push(w);
        }
    }
    return out;
}

int main() {
    scanf("%d %d %d", &n, &m, &r);
    vector<double> c(n + 1);
    for (int v = 1; v <= n; v++) { int x; scanf("%d", &x); c[v] = x; }
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        g[u].push_back(v); g[v].push_back(u);
    }

    vector<vector<int>> cov(n + 1);
    for (int v = 1; v <= n; v++) cov[v] = coverOf(v);

    vector<char> covered(n + 1, 0);
    vector<char> chosen(n + 1, 0);
    int remaining = n;
    vector<int> Z;

    while (remaining > 0) {
        int best = -1;
        double bestRatio = 1e18;
        for (int v = 1; v <= n; v++) {
            if (chosen[v]) continue;
            int gain = 0;
            for (int u : cov[v]) if (!covered[u]) gain++;
            if (gain <= 0) continue;
            double ratio = c[v] / (double)gain;
            if (ratio < bestRatio) { bestRatio = ratio; best = v; }
        }
        if (best == -1) break; // safety
        chosen[best] = 1;
        Z.push_back(best);
        for (int u : cov[best]) if (!covered[u]) { covered[u] = 1; remaining--; }
    }

    // prune: how many chosen regulators cover each zone
    vector<int> cnt(n + 1, 0);
    for (int v : Z) for (int u : cov[v]) cnt[u]++;

    // try removing the most expensive chosen first
    sort(Z.begin(), Z.end(), [&](int a, int b) { return c[a] > c[b]; });
    vector<int> keep;
    for (int v : Z) {
        bool redundant = true;
        for (int u : cov[v]) if (cnt[u] <= 1) { redundant = false; break; }
        if (redundant) {
            for (int u : cov[v]) cnt[u]--;
        } else {
            keep.push_back(v);
        }
    }

    printf("%d\n", (int)keep.size());
    for (int v : keep) printf("%d\n", v);
    return 0;
}
