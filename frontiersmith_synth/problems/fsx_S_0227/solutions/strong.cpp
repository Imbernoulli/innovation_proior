// TIER: strong
// Multi-restart randomized greedy for max-weight independent set on a general conflict
// graph.  Candidate orderings:
//   (0) weight-descending (dominates the plain greedy baseline),
//   (1) degree-aware value/(deg+1) descending,
//   (2..) seeded value-weighted random perturbations of the degree-aware key.
// From each ordering we build a MAXIMAL conflict-free set greedily; we keep the highest
// total value found.  Fully deterministic (fixed RNG seed).  A cheap (1-in/2-out) style
// local extension is applied to the best set: repeatedly try to add any grantable pitch.
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static inline uint64_t xr() {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x; return x;
}
static inline double urand() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

int N; long long M;
vector<long long> w;
vector<vector<int>> adj;

// build a maximal conflict-free set following the given vertex order; return total value
long long buildFrom(const vector<int>& order, vector<char>& sel_out) {
    vector<char> blocked(N + 1, 0);
    vector<char> sel(N + 1, 0);
    long long total = 0;
    for (int v : order) {
        if (blocked[v]) continue;
        sel[v] = 1; total += w[v];
        for (int u : adj[v]) blocked[u] = 1;
    }
    sel_out.swap(sel);
    return total;
}

int main() {
    if (scanf("%d %lld", &N, &M) != 2) return 0;
    w.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);
    adj.assign(N + 1, {});
    for (long long i = 0; i < M; i++) {
        int a, b; scanf("%d %d", &a, &b);
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    vector<int> deg(N + 1, 0);
    for (int v = 1; v <= N; v++) deg[v] = (int)adj[v].size();

    long long bestVal = -1;
    vector<char> bestSel;

    auto consider = [&](vector<int>& order) {
        vector<char> sel;
        long long v = buildFrom(order, sel);
        if (v > bestVal) { bestVal = v; bestSel = sel; }
    };

    // ordering 0: weight-descending (>= plain greedy)
    {
        vector<int> order(N);
        for (int i = 0; i < N; i++) order[i] = i + 1;
        sort(order.begin(), order.end(), [&](int x, int y){
            if (w[x] != w[y]) return w[x] > w[y];
            return x < y;
        });
        consider(order);
    }
    // ordering 1: degree-aware value/(deg+1)
    {
        vector<int> order(N);
        for (int i = 0; i < N; i++) order[i] = i + 1;
        sort(order.begin(), order.end(), [&](int x, int y){
            double kx = (double)w[x] / (deg[x] + 1.0);
            double ky = (double)w[y] / (deg[y] + 1.0);
            if (kx != ky) return kx > ky;
            return x < y;
        });
        consider(order);
    }

    // randomized restarts: perturb the degree-aware key.  Budget scales inversely with
    // size to stay well inside the time limit while keeping the run deterministic.
    long long work = (long long)N + 2 * M;
    int restarts = (int)max<long long>(8, min<long long>(400, 3000000LL / max<long long>(1, work / 4)));
    vector<pair<double,int>> keyed(N);
    for (int r = 0; r < restarts; r++) {
        for (int i = 0; i < N; i++) {
            int v = i + 1;
            double base = (double)w[v] / (deg[v] + 1.0);
            double noise = 0.5 + urand();            // in [0.5, 1.5)
            keyed[i] = {-base * noise, v};           // negative -> ascending sort = key desc
        }
        sort(keyed.begin(), keyed.end());
        vector<int> order(N);
        for (int i = 0; i < N; i++) order[i] = keyed[i].second;
        consider(order);
    }

    // local extension of the best set: ensure it is maximal (add any grantable pitch)
    {
        vector<char> blocked(N + 1, 0);
        for (int v = 1; v <= N; v++) if (bestSel[v])
            for (int u : adj[v]) blocked[u] = 1;
        // vertices ordered by weight desc for the fill-in pass
        vector<int> order(N);
        for (int i = 0; i < N; i++) order[i] = i + 1;
        sort(order.begin(), order.end(), [&](int x, int y){
            if (w[x] != w[y]) return w[x] > w[y];
            return x < y;
        });
        for (int v : order) {
            if (bestSel[v] || blocked[v]) continue;
            bestSel[v] = 1;
            for (int u : adj[v]) blocked[u] = 1;
        }
    }

    vector<int> chosen;
    for (int v = 1; v <= N; v++) if (bestSel[v]) chosen.push_back(v);

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    if (chosen.empty()) printf("\n");
    return 0;
}
