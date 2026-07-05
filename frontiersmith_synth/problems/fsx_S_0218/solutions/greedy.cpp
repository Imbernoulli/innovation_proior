// TIER: greedy
// Coverage greedy: repeatedly install the regulator that newly covers the most
// uncovered zones, ignoring cost. Precomputes each candidate's coverage set by
// depth-limited BFS.
#include <bits/stdc++.h>
using namespace std;

int n, m, r;
vector<vector<int>> g;

vector<int> coverOf(int src) {
    // zones within hop distance r of src
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
    vector<int> c(n + 1);
    for (int v = 1; v <= n; v++) scanf("%d", &c[v]);
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
        int best = -1, bestGain = -1;
        for (int v = 1; v <= n; v++) {
            if (chosen[v]) continue;
            int gain = 0;
            for (int u : cov[v]) if (!covered[u]) gain++;
            if (gain > bestGain) { bestGain = gain; best = v; }
        }
        if (best == -1 || bestGain <= 0) break; // safety
        chosen[best] = 1;
        Z.push_back(best);
        for (int u : cov[best]) if (!covered[u]) { covered[u] = 1; remaining--; }
    }

    printf("%d\n", (int)Z.size());
    for (int v : Z) printf("%d\n", v);
    return 0;
}
