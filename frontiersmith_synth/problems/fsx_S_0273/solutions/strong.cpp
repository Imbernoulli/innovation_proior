// TIER: strong
// Seeded multi-restart local search. From each of several deterministic initializations
// (greedy, all-1, and fixed-seed random) run coordinate-descent sweeps: repeatedly move
// each station to its least-cost channel against ALL neighbors until no improvement.
// Keep the assignment with the smallest total interference across all restarts.
#include <bits/stdc++.h>
using namespace std;

static int N, M, K;
static vector<vector<array<int,3>>> adj; // node -> (nbr, g, w)

static long long stationCost(const vector<int>& col, int j, int c) {
    long long cost = 0;
    for (auto &e : adj[j]) {
        int nb = e[0];
        int d = abs(c - col[nb]);
        int deficit = e[1] - d;
        if (deficit > 0) cost += (long long)e[2] * deficit;
    }
    return cost;
}

static long long totalCost(const vector<int>& col) {
    long long tot = 0;
    for (int j = 1; j <= N; j++)
        for (auto &e : adj[j]) {
            if (e[0] < j) continue; // count each undirected edge once (per directed store)
            int d = abs(col[j] - col[e[0]]);
            int deficit = e[1] - d;
            if (deficit > 0) tot += (long long)e[2] * deficit;
        }
    return tot;
}

static void localSearch(vector<int>& col) {
    bool improved = true;
    int sweeps = 0;
    while (improved && sweeps < 60) {
        improved = false;
        sweeps++;
        for (int j = 1; j <= N; j++) {
            long long best = LLONG_MAX;
            int bestc = col[j];
            for (int c = 1; c <= K; c++) {
                long long cost = stationCost(col, j, c);
                if (cost < best) { best = cost; bestc = c; }
            }
            if (bestc != col[j]) { col[j] = bestc; improved = true; }
        }
    }
}

int main() {
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    adj.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u, v, g, w;
        if (scanf("%d %d %d %d", &u, &v, &g, &w) != 4) break;
        adj[u].push_back({v, g, w});
        adj[v].push_back({u, g, w});
    }

    // Greedy init (index order)
    vector<int> greedy(N + 1, 0);
    for (int j = 1; j <= N; j++) {
        long long best = LLONG_MAX; int bestc = 1;
        for (int c = 1; c <= K; c++) {
            long long cost = 0;
            for (auto &e : adj[j]) {
                if (greedy[e[0]] == 0) continue;
                int d = abs(c - greedy[e[0]]);
                int deficit = e[1] - d;
                if (deficit > 0) cost += (long long)e[2] * deficit;
            }
            if (cost < best) { best = cost; bestc = c; }
        }
        greedy[j] = bestc;
    }

    vector<int> best = greedy;
    localSearch(best);
    long long bestTot = totalCost(best);

    // deterministic multi-restart (fixed seed; no wall-time / no randomness leakage)
    std::mt19937 rng(0xC0FFEEu);
    int restarts = (N <= 400) ? 12 : 6;
    for (int r = 0; r < restarts; r++) {
        vector<int> cur(N + 1);
        if (r == 0) {
            for (int j = 1; j <= N; j++) cur[j] = 1;            // all-1 init
        } else {
            for (int j = 1; j <= N; j++) cur[j] = (int)(rng() % K) + 1; // random init
        }
        localSearch(cur);
        long long t = totalCost(cur);
        if (t < bestTot) { bestTot = t; best = cur; }
    }

    for (int j = 1; j <= N; j++) printf("%d\n", best[j]);
    return 0;
}
