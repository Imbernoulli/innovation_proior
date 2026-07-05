// TIER: strong
// Seeded multi-restart pairwise-swap local search for balanced max-bisection.
// Start once from the weight-greedy solution, then from many random balanced
// assignments; hill-climb by best-improving swaps; keep the best balanced cut.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<vector<pair<int,int>>> adj; // (nbr, w)
vector<array<int,3>> edges;

ll cutOf(const vector<int>& side) {
    ll F = 0;
    for (auto& e : edges) if (side[e[0]] != side[e[1]]) F += e[2];
    return F;
}

// contribution of node v to the cut (weight to opposite-side neighbors)
ll contrib(int v, const vector<int>& side) {
    ll c = 0;
    for (auto& e : adj[v]) if (side[v] != side[e.first]) c += e.second;
    return c;
}

// hill-climb by best-improving swaps between the two sides
void climb(vector<int>& side) {
    for (int iter = 0; iter < 200; iter++) {
        int bi = -1, bj = -1; ll best = 0;
        // evaluate swaps; group nodes by side
        vector<int> s0, s1;
        for (int v = 1; v <= n; v++) (side[v] ? s1 : s0).push_back(v);
        for (int a : s0) for (int b : s1) {
            ll old = contrib(a, side) + contrib(b, side);
            swap(side[a], side[b]);
            ll neu = contrib(a, side) + contrib(b, side);
            swap(side[a], side[b]);
            ll delta = neu - old;
            if (delta > best) { best = delta; bi = a; bj = b; }
        }
        if (bi < 0) break;
        swap(side[bi], side[bj]);
    }
}

int main() {
    scanf("%d %d", &n, &m);
    adj.assign(n + 1, {});
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        edges[i] = {u, v, w};
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // --- greedy start ---
    vector<ll> wsum(n + 1, 0);
    for (auto& e : edges) { wsum[e[0]] += e[2]; wsum[e[1]] += e[2]; }
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){ return wsum[a] > wsum[b]; });
    vector<int> greedy(n + 1, -1);
    int cnt[2] = {0,0}, cap = n / 2;
    for (int idx = 0; idx < n; idx++) {
        int u = order[idx]; ll g0 = 0, g1 = 0;
        for (auto& e : adj[u]) {
            int v = e.first, w = e.second;
            if (greedy[v] == -1) continue;
            if (greedy[v] == 1) g0 += w; else g1 += w;
        }
        int ch;
        if (cnt[0] >= cap) ch = 1;
        else if (cnt[1] >= cap) ch = 0;
        else if (g0 != g1) ch = (g0 > g1) ? 0 : 1;
        else ch = (cnt[0] <= cnt[1]) ? 0 : 1;
        greedy[u] = ch; cnt[ch]++;
    }

    vector<int> best = greedy;
    climb(best);
    ll bestCut = cutOf(best);

    // --- multi-restart from random balanced assignments (seeded, deterministic) ---
    mt19937 rng(987654321u);
    int restarts = 40;
    for (int r = 0; r < restarts; r++) {
        vector<int> side(n + 1, 0);
        vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i + 1;
        shuffle(idx.begin(), idx.end(), rng);
        for (int i = 0; i < n / 2; i++) side[idx[i]] = 1;
        climb(side);
        ll c = cutOf(side);
        if (c > bestCut) { bestCut = c; best = side; }
    }

    for (int i = 1; i <= n; i++) printf("%d\n", best[i]);
    return 0;
}
