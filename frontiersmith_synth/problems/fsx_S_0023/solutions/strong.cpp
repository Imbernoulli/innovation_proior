// TIER: strong
// Multi-strategy heuristic for maximum-weight independent set, keep the best roster:
//   (1) weight-greedy (same construction as the greedy tier),
//   (2) GWMIN: dynamic weight/(1+degree) greedy,
//   (3) seeded randomized restarts with local-search fill.
// Deterministic (fixed seed). Guaranteed >= greedy because it includes it as a candidate.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> w;
vector<vector<int>> adj;

// Given a priority ordering, build a maximal independent set greedily.
ll buildFromOrder(const vector<int>& order, vector<char>& outChosen) {
    vector<char> chosen(n + 1, 0), blocked(n + 1, 0);
    ll tot = 0;
    for (int u : order) {
        if (blocked[u]) continue;
        chosen[u] = 1;
        tot += w[u];
        for (int v : adj[u]) blocked[v] = 1;
    }
    outChosen = chosen;
    return tot;
}

// Add any still-addable vertex (highest weight first) -> makes the set maximal.
ll localFill(vector<char>& chosen) {
    vector<char> blocked(n + 1, 0);
    for (int u = 1; u <= n; u++)
        if (chosen[u]) for (int v : adj[u]) blocked[v] = 1;
    vector<int> cand;
    for (int u = 1; u <= n; u++)
        if (!chosen[u] && !blocked[u]) cand.push_back(u);
    sort(cand.begin(), cand.end(), [&](int a, int b) { return w[a] > w[b]; });
    ll add = 0;
    for (int u : cand) {
        if (blocked[u] || chosen[u]) continue;
        chosen[u] = 1;
        add += w[u];
        for (int v : adj[u]) blocked[v] = 1;
    }
    ll tot = 0;
    for (int u = 1; u <= n; u++) if (chosen[u]) tot += w[u];
    return tot;
}

int main() {
    scanf("%d %d", &n, &m);
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    ll bestTot = -1;
    vector<char> bestChosen;

    auto consider = [&](vector<char> ch) {
        ll t = 0;
        for (int u = 1; u <= n; u++) if (ch[u]) t += w[u];
        if (t > bestTot) { bestTot = t; bestChosen = ch; }
    };

    // (1) weight-greedy
    {
        vector<int> order(n);
        iota(order.begin(), order.end(), 1);
        sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });
        vector<char> ch;
        buildFromOrder(order, ch);
        localFill(ch);
        consider(ch);
    }

    // (2) GWMIN: repeatedly pick alive vertex maximizing w/(1+curdeg)
    {
        vector<char> alive(n + 1, 1), chosen(n + 1, 0);
        vector<int> curdeg(n + 1, 0);
        for (int u = 1; u <= n; u++) curdeg[u] = (int)adj[u].size();
        int remaining = n;
        while (remaining > 0) {
            int best = -1; double bestVal = -1;
            for (int u = 1; u <= n; u++) {
                if (!alive[u]) continue;
                double val = (double)w[u] / (1.0 + curdeg[u]);
                if (val > bestVal) { bestVal = val; best = u; }
            }
            if (best == -1) break;
            chosen[best] = 1;
            // remove best and all its (alive) neighbors
            vector<int> toRemove;
            toRemove.push_back(best);
            for (int v : adj[best]) if (alive[v]) toRemove.push_back(v);
            for (int x : toRemove) {
                if (!alive[x]) continue;
                alive[x] = 0; remaining--;
                for (int y : adj[x]) if (alive[y]) curdeg[y]--;
            }
        }
        localFill(chosen);
        consider(chosen);
    }

    // (3) seeded randomized restarts with local-search fill
    {
        std::mt19937 rng(12345u);
        int restarts = (n <= 1500) ? 40 : 15;
        for (int r = 0; r < restarts; r++) {
            // priority = weight scaled by a random jitter -> perturbed weight order
            vector<pair<double,int>> pr(n);
            for (int u = 1; u <= n; u++) {
                double jitter = 0.6 + 0.8 * (double)(rng() % 100000) / 100000.0;
                pr[u - 1] = {-(double)w[u] * jitter, u};
            }
            sort(pr.begin(), pr.end());
            vector<int> order(n);
            for (int i = 0; i < n; i++) order[i] = pr[i].second;
            vector<char> ch;
            buildFromOrder(order, ch);
            localFill(ch);
            consider(ch);
        }
    }

    if (bestChosen.empty()) bestChosen.assign(n + 1, 0);
    vector<int> sel;
    for (int u = 1; u <= n; u++) if (bestChosen[u]) sel.push_back(u);
    printf("%d\n", (int)sel.size());
    for (int u : sel) printf("%d\n", u);
    return 0;
}
