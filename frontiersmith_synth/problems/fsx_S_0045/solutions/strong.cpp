// TIER: strong
// Greedy init + node-move local search, wrapped in a seeded multi-restart loop.
// Deterministic (fixed seed). Keeps the least-interference assignment found.
#include <bits/stdc++.h>
using namespace std;

int n, m, C;
vector<vector<pair<int,int>>> adj; // (neighbor, weight)

// Local search: repeatedly move each node to the circuit minimizing its local
// interference until a full pass makes no improvement (or pass cap reached).
void localSearch(vector<int>& col) {
    bool improved = true;
    int passes = 0;
    while (improved && passes < 60) {
        improved = false;
        passes++;
        for (int i = 1; i <= n; i++) {
            static vector<long long> cost;
            cost.assign(C + 1, 0);
            for (auto& e : adj[i]) cost[col[e.first]] += e.second;
            int best = col[i];
            long long bestc = cost[col[i]];
            for (int c = 1; c <= C; c++)
                if (cost[c] < bestc) { bestc = cost[c]; best = c; }
            if (best != col[i]) { col[i] = best; improved = true; }
        }
    }
}

long long objective(const vector<int>& col) {
    long long F = 0;
    for (int i = 1; i <= n; i++)
        for (auto& e : adj[i])
            if (e.first > i && col[e.first] == col[i]) F += e.second;
    return F;
}

int main() {
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // start 1: greedy index-order coloring
    vector<int> best(n + 1, 1);
    {
        vector<int> col(n + 1, 0);
        for (int i = 1; i <= n; i++) {
            vector<long long> cost(C + 1, 0);
            for (auto& e : adj[i])
                if (col[e.first] != 0) cost[col[e.first]] += e.second;
            int b = 1;
            for (int c = 2; c <= C; c++) if (cost[c] < cost[b]) b = c;
            col[i] = b;
        }
        localSearch(col);
        best = col;
    }
    long long bestF = objective(best);

    // multi-restart from seeded random initial colorings
    std::mt19937 rng(987654321u);
    int restarts = 8;
    for (int r = 0; r < restarts; r++) {
        vector<int> col(n + 1);
        for (int i = 1; i <= n; i++) col[i] = (int)(rng() % C) + 1;
        localSearch(col);
        long long F = objective(col);
        if (F < bestF) { bestF = F; best = col; }
    }

    for (int i = 1; i <= n; i++) printf("%d ", best[i]);
    printf("\n");
    return 0;
}
