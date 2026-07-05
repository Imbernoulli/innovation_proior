// TIER: strong
// Greedy construction + balance-preserving swap local search for balanced min k-cut.
// Maintain W[i][g] = drift weight from hive i to hives currently in yard g. Repeatedly
// pick two hives in different yards and swap them iff the swap lowers total cross-yard
// drift (hill climbing with a deterministic move budget). Strictly refines greedy.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long m; int k;
    scanf("%d %lld %d", &n, &m, &k);
    int s = n / k;

    vector<int> eu(m), ev(m);
    vector<long long> ew(m);
    vector<vector<pair<int,long long>>> adj(n + 1);
    vector<long long> wdeg(n + 1, 0);
    unordered_map<long long,long long> edgeW;
    edgeW.reserve(m * 2 + 16);
    long long N1 = n + 1;
    for (long long e = 0; e < m; e++) {
        int u, v; long long w;
        scanf("%d %d %lld", &u, &v, &w);
        eu[e] = u; ev[e] = v; ew[e] = w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        wdeg[u] += w; wdeg[v] += w;
        int a = u, b = v; if (a > b) swap(a, b);
        edgeW[(long long)a * N1 + b] = w;
    }

    // ---- greedy construction (same idea as greedy.cpp) ----------------------
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (wdeg[a] != wdeg[b]) return wdeg[a] > wdeg[b];
        return a < b;
    });
    vector<int> yard(n + 1, -1);
    vector<int> cap(k, s);
    vector<long long> pull(k, 0);
    for (int idx = 0; idx < n; idx++) {
        int h = order[idx];
        for (int g = 0; g < k; g++) pull[g] = 0;
        for (auto &pr : adj[h])
            if (yard[pr.first] >= 0) pull[yard[pr.first]] += pr.second;
        int best = -1; long long bestPull = -1; int bestCap = -1;
        for (int g = 0; g < k; g++) {
            if (cap[g] <= 0) continue;
            if (pull[g] > bestPull || (pull[g] == bestPull && cap[g] > bestCap)) {
                bestPull = pull[g]; bestCap = cap[g]; best = g;
            }
        }
        yard[h] = best; cap[best]--;
    }

    // ---- W table -------------------------------------------------------------
    // W[i*k + g]
    vector<long long> W((long long)(n + 1) * k, 0);
    auto Wat = [&](int i, int g) -> long long& { return W[(long long)i * k + g]; };
    for (long long e = 0; e < m; e++) {
        Wat(eu[e], yard[ev[e]]) += ew[e];
        Wat(ev[e], yard[eu[e]]) += ew[e];
    }

    auto edgeWeight = [&](int a, int b) -> long long {
        if (a > b) swap(a, b);
        auto it = edgeW.find((long long)a * N1 + b);
        return it == edgeW.end() ? 0LL : it->second;
    };

    // group members lists for picking swap partners in different yards
    // (we just pick two random hives and require different yards)
    mt19937_64 rng(0x9E3779B97F4A7C15ULL ^ ((unsigned long long)n << 20) ^ ((unsigned long long)m << 3) ^ (unsigned long long)k);

    long long attempts = (long long)max<long long>(400000, (long long)n * 1500);
    attempts = min<long long>(attempts, 4000000);

    for (long long it = 0; it < attempts; it++) {
        int a = (int)(rng() % n) + 1;
        int b = (int)(rng() % n) + 1;
        int A = yard[a], B = yard[b];
        if (A == B) continue;
        long long eab = edgeWeight(a, b);
        // delta cut for swapping a<->b
        long long delta = Wat(a, A) - Wat(a, B) + Wat(b, B) - Wat(b, A) + 2 * eab;
        if (delta < 0) {
            // apply: move a A->B, then b B->A
            for (auto &pr : adj[a]) {
                int x = pr.first; long long w = pr.second;
                Wat(x, A) -= w; Wat(x, B) += w;
            }
            yard[a] = B;
            for (auto &pr : adj[b]) {
                int x = pr.first; long long w = pr.second;
                Wat(x, B) -= w; Wat(x, A) += w;
            }
            yard[b] = A;
        }
    }

    for (int i = 1; i <= n; i++)
        printf("%d%c", yard[i], i == n ? '\n' : ' ');
    return 0;
}
