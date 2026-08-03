// TIER: strong
// Best-of several constructive heuristics for max-weight independent set:
//   (1) weight-descending greedy,
//   (2) value/(degree+1)-descending greedy,
//   (3) many seeded value-weighted randomized-order greedies,
// each built into a maximal conflict-free set; keep the highest-value set found.
// Deterministic (fixed RNG seed).
#include <bits/stdc++.h>
using namespace std;

int N, M;
vector<long long> w;
vector<vector<int>> adj;

// build a maximal conflict-free set following the given station order; return total value
long long buildFromOrder(const vector<int>& order, vector<char>& sel, vector<int>& chosen) {
    fill(sel.begin(), sel.end(), 0);
    chosen.clear();
    long long val = 0;
    for (int j : order) {
        bool ok = true;
        for (int nb : adj[j]) if (sel[nb]) { ok = false; break; }
        if (ok) { sel[j] = 1; chosen.push_back(j); val += w[j]; }
    }
    return val;
}

int main() {
    scanf("%d %d", &N, &M);
    w.assign(N + 1, 0);
    for (int j = 1; j <= N; j++) scanf("%lld", &w[j]);
    adj.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> deg(N + 1, 0);
    for (int j = 1; j <= N; j++) deg[j] = (int)adj[j].size();

    vector<char> sel(N + 1, 0);
    vector<int> chosen, bestSet;
    long long bestVal = -1;

    auto consider = [&](const vector<int>& order) {
        long long v = buildFromOrder(order, sel, chosen);
        if (v > bestVal) { bestVal = v; bestSet = chosen; }
    };

    // (1) weight-descending
    {
        vector<int> ord(N);
        for (int j = 0; j < N; j++) ord[j] = j + 1;
        sort(ord.begin(), ord.end(), [&](int a, int b) {
            if (w[a] != w[b]) return w[a] > w[b];
            return a < b;
        });
        consider(ord);
    }
    // (2) value/(degree+1)-descending
    {
        vector<int> ord(N);
        for (int j = 0; j < N; j++) ord[j] = j + 1;
        sort(ord.begin(), ord.end(), [&](int a, int b) {
            double ra = (double)w[a] / (deg[a] + 1.0);
            double rb = (double)w[b] / (deg[b] + 1.0);
            if (ra != rb) return ra > rb;
            return a < b;
        });
        consider(ord);
    }
    // (3) seeded value-weighted randomized restarts
    {
        mt19937_64 rng(987654321ULL);
        int restarts = 300;
        vector<int> ord(N);
        for (int j = 0; j < N; j++) ord[j] = j + 1;
        for (int r = 0; r < restarts; r++) {
            // key = weight * random in (0,1]; sort by key descending -> value-biased order
            vector<pair<double,int>> keyed(N);
            for (int j = 1; j <= N; j++) {
                double u = (double)((rng() >> 11) + 1) / (double)((1ULL << 53));
                double key = (double)w[j] * u / (deg[j] + 1.0);
                keyed[j - 1] = {key, j};
            }
            sort(keyed.begin(), keyed.end(),
                 [](const pair<double,int>& a, const pair<double,int>& b) {
                     return a.first > b.first;
                 });
            for (int j = 0; j < N; j++) ord[j] = keyed[j].second;
            consider(ord);
        }
    }

    sort(bestSet.begin(), bestSet.end());
    printf("%d\n", (int)bestSet.size());
    for (size_t i = 0; i < bestSet.size(); i++)
        printf("%d%c", bestSet[i], i + 1 == bestSet.size() ? '\n' : ' ');
    if (bestSet.empty()) printf("\n");
    return 0;
}
