// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<vector<int>> adj;

// greedily top up a selection to a MAXIMAL independent set (value-first order).
void maximalize(vector<char>& chosen) {
    vector<char> blocked(n + 1, 0);
    for (int v = 1; v <= n; v++)
        if (chosen[v]) for (int u : adj[v]) blocked[u] = 1;
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });
    for (int v : order) {
        if (chosen[v] || blocked[v]) continue;
        chosen[v] = 1;
        for (int u : adj[v]) blocked[u] = 1;
    }
}

ll valueOf(const vector<char>& chosen) {
    ll s = 0;
    for (int v = 1; v <= n; v++) if (chosen[v]) s += w[v];
    return s;
}

// degree-aware greedy (GWMIN): repeatedly pick alive site maximizing value/(deg+1).
vector<char> gwmin(unsigned seed) {
    mt19937 rng(seed);
    vector<char> alive(n + 1, 1);
    vector<int> deg(n + 1, 0);
    for (int i = 1; i <= n; i++) deg[i] = (int)adj[i].size();
    vector<char> chosen(n + 1, 0);
    int remaining = n;
    while (remaining > 0) {
        double bestScore = -1; int bestV = -1;
        for (int v = 1; v <= n; v++) if (alive[v]) {
            double sc = (double)w[v] / (deg[v] + 1) + (rng() % 1000) * 1e-9;
            if (sc > bestScore) { bestScore = sc; bestV = v; }
        }
        chosen[bestV] = 1;
        vector<int> torem;
        alive[bestV] = 0; remaining--; torem.push_back(bestV);
        for (int u : adj[bestV]) if (alive[u]) { alive[u] = 0; remaining--; torem.push_back(u); }
        for (int x : torem)
            for (int u : adj[x]) if (alive[u]) deg[u]--;
    }
    return chosen;
}

// pure weight-greedy candidate (guarantees strong >= greedy).
vector<char> weightGreedy() {
    vector<char> chosen(n + 1, 0);
    maximalize(chosen); // top-up from empty in value order == weight-greedy
    return chosen;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<char> best = weightGreedy();
    ll bestVal = valueOf(best);

    // multi-restart degree-aware greedy, each maximalized; keep the best.
    int restarts = 24;
    for (int s = 0; s < restarts; s++) {
        vector<char> cand = gwmin(12345u + 7919u * (unsigned)s);
        maximalize(cand);
        ll v = valueOf(cand);
        if (v > bestVal) { bestVal = v; best = cand; }
    }

    vector<int> sel;
    for (int v = 1; v <= n; v++) if (best[v]) sel.push_back(v);
    printf("%d\n", (int)sel.size());
    for (int v : sel) printf("%d\n", v);
    return 0;
}
