// TIER: strong
// The insight: the abelian property means only the COUNT of firings per pile
// matters, never the order -- so instead of simulating one firing at a time,
// compute each pile's true full-stabilization odometer u[v] directly via
// batched simultaneous-sweep relaxation (fire every currently-unstable pile
// as many times as its OWN current height allows, all at once, repeat until
// stable). This turns "simulate a huge number of individual topples" into a
// handful of vector operations.
//
// With u[v] known, group piles into their connected clusters (components of
// the pile-only subgraph), and treat funding a cluster as a knapsack item:
// cost = odometer firings needed to fully stabilize it, value = the
// criticality-weighted risk that fully stabilizing it removes. Rank clusters
// by value/cost density and greedily fund the best ones first -- this is
// what actually spends a limited firing budget well, unlike chasing whatever
// pile currently LOOKS biggest.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ll n64, m64, K;
    cin >> n64 >> m64 >> K;
    int n = (int)n64, m = (int)m64;
    int sink = n;
    vector<vector<pair<int,ll>>> adj(n + 1);
    vector<ll> deg(n, 0);
    vector<array<ll,2>> edgeEnds(m);
    for (int i = 0; i < m; i++) {
        ll a, b, w;
        cin >> a >> b >> w;
        adj[a].push_back({(int)b, w});
        adj[b].push_back({(int)a, w});
        if (a < n) deg[a] += w;
        if (b < n) deg[b] += w;
        edgeEnds[i] = {a, b};
    }
    vector<ll> h(n), c(n);
    for (int i = 0; i < n; i++) cin >> h[i];
    for (int i = 0; i < n; i++) cin >> c[i];

    // ---- Step A: full-stabilization odometer via batched parallel sweeps ----
    vector<ll> height = h;
    vector<ll> u(n, 0);
    for (int iter = 0; iter < 20000; iter++) {
        vector<ll> times(n, 0);
        bool any = false;
        for (int v = 0; v < n; v++) {
            if (height[v] >= deg[v]) { times[v] = height[v] / deg[v]; any = true; }
        }
        if (!any) break;
        vector<ll> newHeight = height;
        for (int v = 0; v < n; v++) {
            if (times[v] > 0) {
                newHeight[v] -= times[v] * deg[v];
                u[v] += times[v];
                for (auto& pr : adj[v]) {
                    int uu = pr.first; ll w = pr.second;
                    if (uu == sink) continue;
                    newHeight[uu] += w * times[v];
                }
            }
        }
        height = newHeight;
    }

    // ---- Step B: union-find clusters over pile-pile edges only ----
    vector<int> parent(n);
    iota(parent.begin(), parent.end(), 0);
    function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto uni = [&](int a, int b) { a = find(a); b = find(b); if (a != b) parent[a] = b; };
    for (auto& e : edgeEnds) {
        ll a = e[0], b = e[1];
        if (a < n && b < n) uni((int)a, (int)b);
    }

    // ---- Step C: per-cluster (value, cost) ----
    vector<ll> e0(n);
    for (int v = 0; v < n; v++) { ll ev = h[v] - deg[v] + 1; e0[v] = ev > 0 ? ev : 0; }

    map<int, ll> clusterCost, clusterValue;
    map<int, vector<int>> members;
    for (int v = 0; v < n; v++) {
        int r = find(v);
        clusterCost[r] += u[v];
        clusterValue[r] += c[v] * e0[v];
        members[r].push_back(v);
    }
    vector<int> roots;
    for (auto& kv : clusterCost) roots.push_back(kv.first);
    sort(roots.begin(), roots.end(), [&](int a, int b) {
        long double va = (long double)clusterValue[a] * (long double)clusterCost[b];
        long double vb = (long double)clusterValue[b] * (long double)clusterCost[a];
        if (va != vb) return va > vb;
        return clusterCost[a] < clusterCost[b];
    });

    // ---- Step D: greedy knapsack pass over clusters by density ----
    vector<ll> f(n, 0);
    ll budget = K;
    map<int, char> funded;
    for (int r : roots) {
        ll cost = clusterCost[r];
        if (cost <= budget) { budget -= cost; funded[r] = 1; }
        else funded[r] = 0;
    }
    for (int v = 0; v < n; v++) { int r = find(v); if (funded[r]) f[v] = u[v]; }

    // ---- Step E: spend any leftover budget on unfunded PENDANT clusters ----
    for (int r : roots) {
        if (funded[r]) continue;
        if (budget <= 0) break;
        auto& mem = members[r];
        if (mem.size() != 1) continue;
        int v = mem[0];
        if (deg[v] <= 0) continue;
        ll doable = min(u[v], budget / deg[v]);
        if (doable > 0) { f[v] += doable; budget -= doable * deg[v]; }
    }

    for (int i = 0; i < n; i++) cout << f[i] << " \n"[i + 1 == n];
    return 0;
}
