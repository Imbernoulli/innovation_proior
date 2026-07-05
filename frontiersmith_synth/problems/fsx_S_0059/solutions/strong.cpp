// TIER: strong
// Density-ordered greedy (w/(d*(1+deg))) + seeded randomized multi-restart, keep best schedule.
#include <bits/stdc++.h>
using namespace std;

int n, m; long long W;
vector<long long> w, d;
vector<vector<int>> adj;

// Greedy fill given a scoring vector; higher score => considered earlier.
long long fill(const vector<double>& score, vector<char>& outChosen) {
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (score[a] != score[b]) return score[a] > score[b];
        return a < b;
    });
    vector<char> chosen(n + 1, 0);
    long long used = 0, F = 0;
    for (int i : order) {
        if (used + d[i] > W) continue;
        bool ok = true;
        for (int j : adj[i]) if (chosen[j]) { ok = false; break; }
        if (!ok) continue;
        chosen[i] = 1;
        used += d[i];
        F += w[i];
    }
    outChosen = chosen;
    return F;
}

int main() {
    if (scanf("%d %d %lld", &n, &m, &W) != 3) return 0;
    w.assign(n + 1, 0); d.assign(n + 1, 0); adj.assign(n + 1, {});
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    for (int i = 1; i <= n; i++) scanf("%lld", &d[i]);
    for (int e = 0; e < m; e++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> deg(n + 1, 0);
    for (int i = 1; i <= n; i++) deg[i] = (int)adj[i].size();

    // base density score
    vector<double> base(n + 1, 0.0);
    for (int i = 1; i <= n; i++)
        base[i] = (double)w[i] / ((double)d[i] * (1.0 + (double)deg[i]));

    vector<char> bestChosen;
    long long bestF = -1;

    // restart 0: pure density
    {
        vector<char> ch;
        long long f = fill(base, ch);
        if (f > bestF) { bestF = f; bestChosen = ch; }
    }
    // restart 1: pure yield
    {
        vector<double> s(n + 1);
        for (int i = 1; i <= n; i++) s[i] = (double)w[i];
        vector<char> ch;
        long long f = fill(s, ch);
        if (f > bestF) { bestF = f; bestChosen = ch; }
    }

    // randomized restarts: perturb the density ordering
    std::mt19937 rng(987654321u);
    int R = (int)min<long long>(48, 1500000LL / (n + 1) + 6);
    for (int r = 0; r < R; r++) {
        vector<double> s(n + 1);
        double lo = 0.5, hi = 1.5;
        std::uniform_real_distribution<double> U(lo, hi);
        for (int i = 1; i <= n; i++) s[i] = base[i] * U(rng);
        vector<char> ch;
        long long f = fill(s, ch);
        if (f > bestF) { bestF = f; bestChosen = ch; }
    }

    // light local search: try to add any still-addable plot to the best schedule
    {
        vector<char> chosen = bestChosen;
        long long used = 0, F = 0;
        for (int i = 1; i <= n; i++) if (chosen[i]) { used += d[i]; F += w[i]; }
        bool improved = true;
        while (improved) {
            improved = false;
            for (int i = 1; i <= n; i++) {
                if (chosen[i]) continue;
                if (used + d[i] > W) continue;
                bool ok = true;
                for (int j : adj[i]) if (chosen[j]) { ok = false; break; }
                if (!ok) continue;
                chosen[i] = 1; used += d[i]; F += w[i];
                improved = true;
            }
        }
        if (F > bestF) { bestF = F; bestChosen = chosen; }
    }

    vector<int> res;
    for (int i = 1; i <= n; i++) if (bestChosen[i]) res.push_back(i);
    printf("%d\n", (int)res.size());
    for (int i : res) printf("%d\n", i);
    return 0;
}
