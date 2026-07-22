// TIER: strong
// Insight: the right affinity signal is co-residence within an LRU reuse window of the
// ACTUAL trace, not graph adjacency. Build a windowed co-visit weight graph directly
// from the tour (pairs of animals that appear within the last L*pageCap tour steps of
// each other), then greedily merge the highest-weight pairs into shared shelves under
// the capacity cap (bounded union-find). This recovers the hub group as one shelf
// regardless of where the paddock graph structurally embeds each hub.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, pageCap, L, T;
    if (scanf("%d %d %d %d %d", &n, &m, &pageCap, &L, &T) != 5) return 0;
    for (int i = 0; i < m; i++) { int u, v; scanf("%d %d", &u, &v); }
    vector<int> trace(T);
    for (int i = 0; i < T; i++) scanf("%d", &trace[i]);

    int W = max(2, L * pageCap); // reuse-window size

    vector<vector<int>> w(n + 1, vector<int>(n + 1, 0));
    deque<int> win;
    for (int i = 0; i < T; i++) {
        int x = trace[i];
        for (int y : win) {
            if (y != x) { w[x][y]++; w[y][x]++; }
        }
        win.push_back(x);
        if ((int)win.size() > W) win.pop_front();
    }

    vector<array<int,3>> pairs; // {weight, u, v}
    for (int u = 1; u <= n; u++)
        for (int v = u + 1; v <= n; v++)
            if (w[u][v] > 0) pairs.push_back({w[u][v], u, v});
    sort(pairs.begin(), pairs.end(), [](const array<int,3>& a, const array<int,3>& b) {
        return a[0] > b[0];
    });

    vector<int> par(n + 1), sz(n + 1);
    for (int i = 1; i <= n; i++) { par[i] = i; sz[i] = 1; }
    function<int(int)> find_ = [&](int x) {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    };
    for (auto &pr : pairs) {
        int u = pr[1], v = pr[2];
        int ru = find_(u), rv = find_(v);
        if (ru == rv) continue;
        if (sz[ru] + sz[rv] <= pageCap) {
            if (sz[ru] < sz[rv]) swap(ru, rv);
            par[rv] = ru;
            sz[ru] += sz[rv];
        }
    }

    unordered_map<int,int> pageOf;
    int nextPage = 1;
    vector<int> page(n + 1);
    for (int i = 1; i <= n; i++) {
        int r = find_(i);
        auto it = pageOf.find(r);
        int pid;
        if (it == pageOf.end()) { pid = nextPage++; pageOf[r] = pid; }
        else pid = it->second;
        page[i] = pid;
    }
    for (int i = 1; i <= n; i++) printf("%d\n", page[i]);
    return 0;
}
