// TIER: strong
// Multi-start randomized weighted greedy + local-search improvement.
//  * seed candidates: weight-descending greedy, weight/(deg+1) greedy, and many
//    randomized weighted greedies (deterministic RNG).
//  * each seed circuit is polished by an improving-swap local search:
//    - maximality fill (add any ride with no lit neighbour),
//    - k->1 swaps: if lighting an unlit ride u and switching off its lit
//      neighbours strictly raises total revenue, do it.
//  * keep the best circuit found across all restarts.
// Deterministic given the input (fixed seed), so scores are reproducible.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<vector<int>> adj;

// local search state
vector<char> sel;
vector<int> selNbrCnt;   // # lit neighbours
vector<ll>  selNbrW;     // sum of weights of lit neighbours

static inline void addSel(int v) {
    sel[v] = 1;
    for (int u : adj[v]) { selNbrCnt[u]++; selNbrW[u] += w[v]; }
}
static inline void remSel(int v) {
    sel[v] = 0;
    for (int u : adj[v]) { selNbrCnt[u]--; selNbrW[u] -= w[v]; }
}

ll buildFrom(const vector<int>& order) {
    // fresh greedy from given order
    fill(sel.begin(), sel.end(), 0);
    fill(selNbrCnt.begin(), selNbrCnt.end(), 0);
    fill(selNbrW.begin(), selNbrW.end(), 0);
    for (int v : order)
        if (selNbrCnt[v] == 0) addSel(v);

    // local search: swaps + maximality fill
    bool improved = true;
    int passes = 0;
    while (improved && passes < 12) {
        improved = false;
        passes++;
        for (int v = 1; v <= n; v++) {
            if (sel[v]) continue;
            // gain of lighting v (removing its lit neighbours)
            if (w[v] - selNbrW[v] > 0) {
                // switch off lit neighbours of v, then light v
                // collect lit neighbours first (adj changes selNbr as we remove)
                // removing neighbours: iterate copy
                for (int u : adj[v]) if (sel[u]) remSel(u);
                addSel(v);
                improved = true;
            }
        }
    }
    ll tot = 0;
    for (int v = 1; v <= n; v++) if (sel[v]) tot += w[v];
    return tot;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) { int x; scanf("%d", &x); w[i] = x; }
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    sel.assign(n + 1, 0);
    selNbrCnt.assign(n + 1, 0);
    selNbrW.assign(n + 1, 0);

    vector<int> deg(n + 1, 0);
    for (int v = 1; v <= n; v++) deg[v] = (int)adj[v].size();

    ll best = -1;
    vector<char> bestSel(n + 1, 0);
    auto record = [&](ll val) {
        if (val > best) { best = val; bestSel.assign(sel.begin(), sel.end()); }
    };

    vector<int> base(n);
    for (int i = 0; i < n; i++) base[i] = i + 1;

    // candidate 1: weight descending
    {
        vector<int> o = base;
        sort(o.begin(), o.end(), [&](int a, int b){
            if (w[a] != w[b]) return w[a] > w[b]; return a < b; });
        record(buildFrom(o));
    }
    // candidate 2: weight / (deg+1) descending
    {
        vector<int> o = base;
        sort(o.begin(), o.end(), [&](int a, int b){
            double ka = (double)w[a] / (deg[a] + 1.0);
            double kb = (double)w[b] / (deg[b] + 1.0);
            if (ka != kb) return ka > kb; return a < b; });
        record(buildFrom(o));
    }

    // randomized restarts: perturb a weighted key, deterministic RNG.
    mt19937 rng(0x9E3779B9u ^ (unsigned)n ^ ((unsigned)m << 1));
    // budget scales down with size to respect the time limit.
    int restarts = 400;
    if (n > 500) restarts = 120;
    if (n > 2000) restarts = 40;
    if (n > 4000) restarts = 20;
    for (int r = 0; r < restarts; r++) {
        vector<pair<double,int>> key(n);
        for (int i = 0; i < n; i++) {
            double noise = (double)(rng() % 1000000) / 1000000.0; // [0,1)
            double base_k = (double)w[i + 1] / (deg[i + 1] + 1.0);
            // blend weight-per-degree with weight and random jitter
            double k = base_k * (0.5 + noise) + (double)w[i + 1] * 0.001 * noise;
            key[i] = { -k, i + 1 };
        }
        sort(key.begin(), key.end());
        vector<int> o(n);
        for (int i = 0; i < n; i++) o[i] = key[i].second;
        record(buildFrom(o));
    }

    vector<int> out;
    for (int i = 1; i <= n; i++) if (bestSel[i]) out.push_back(i);
    printf("%d\n", (int)out.size());
    for (size_t i = 0; i < out.size(); i++)
        printf("%d%c", out[i], i + 1 == out.size() ? '\n' : ' ');
    return 0;
}
