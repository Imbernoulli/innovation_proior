// TIER: strong
// Multi-restart local search: from greedy and several seeded random initializations,
// repeatedly move each zone to its least-conflict program until no improving single-zone
// move remains; keep the assignment with the smallest total conflict.
#include <bits/stdc++.h>
using namespace std;

int N, M, K;
vector<vector<pair<int,long long>>> adj;

long long totalConflict(const vector<int>& col) {
    long long F = 0;
    for (int u = 1; u <= N; u++)
        for (auto &e : adj[u])
            if (e.first > u && col[e.first] == col[u]) F += e.second;
    return F;
}

// Local search to convergence (single-zone best-move). Returns improved assignment.
void localSearch(vector<int>& col) {
    bool improved = true;
    int passes = 0;
    while (improved && passes < 200) {
        improved = false;
        passes++;
        for (int u = 1; u <= N; u++) {
            static vector<long long> cost;
            cost.assign(K + 1, 0);
            for (auto &e : adj[u]) cost[col[e.first]] += e.second;
            int best = col[u]; long long bc = cost[col[u]];
            for (int c = 1; c <= K; c++)
                if (cost[c] < bc) { bc = cost[c]; best = c; }
            if (best != col[u]) { col[u] = best; improved = true; }
        }
    }
}

int main() {
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    adj.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u, v; long long w;
        if (scanf("%d %d %lld", &u, &v, &w) != 3) break;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    mt19937 rng(12345);
    vector<int> best;
    long long bestF = LLONG_MAX;

    // candidate 1: greedy init
    {
        vector<int> col(N + 1, 0);
        for (int j = 1; j <= N; j++) {
            vector<long long> cost(K + 1, 0);
            for (auto &e : adj[j])
                if (col[e.first] != 0) cost[col[e.first]] += e.second;
            int bcl = 1; long long bc = cost[1];
            for (int c = 2; c <= K; c++)
                if (cost[c] < bc) { bc = cost[c]; bcl = c; }
            col[j] = bcl;
        }
        localSearch(col);
        long long F = totalConflict(col);
        if (F < bestF) { bestF = F; best = col; }
    }

    // candidates 2..: random restarts
    int restarts = 30;
    for (int r = 0; r < restarts; r++) {
        vector<int> col(N + 1, 1);
        for (int j = 1; j <= N; j++) col[j] = (int)(rng() % K) + 1;
        localSearch(col);
        long long F = totalConflict(col);
        if (F < bestF) { bestF = F; best = col; }
    }

    for (int j = 1; j <= N; j++) printf("%d\n", best[j]);
    return 0;
}
