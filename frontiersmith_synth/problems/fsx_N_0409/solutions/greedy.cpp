// TIER: greedy
// Cut-only hill climb. Start from round-robin, then repeatedly try size-
// preserving swaps of two cars on different tracks; accept a swap iff it
// increases the total separated conflict weight (sum of boundary values).
// It IGNORES residue tuning, so it lifts the cut a bit above baseline but
// never earns the (1+lam) multiplier -> lands between trivial and strong.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, k; ll m, lam, L, U;
vector<ll> w;
vector<int> t;
vector<int> assign_;
vector<ll> S, bnd;
vector<vector<pair<int, ll>>> adj;

static inline void moveCar(int x, int to) {
    int from = assign_[x];
    if (from == to) return;
    for (auto& pr : adj[x]) {
        int y = pr.first; ll c = pr.second; int ty = assign_[y];
        if (ty == from)      { bnd[from] += c; bnd[to] += c; }
        else if (ty == to)   { bnd[from] -= c; bnd[to] -= c; }
        else                 { bnd[from] -= c; bnd[to] += c; }
    }
    S[from] -= w[x]; S[to] += w[x];
    assign_[x] = to;
}
// swap tracks of u and v (different tracks); size-preserving
static inline void doSwap(int u, int v) {
    int a = assign_[u], b = assign_[v];
    moveCar(u, b); moveCar(v, a);
}
static inline ll baseCut() { ll s = 0; for (int j = 0; j < k; j++) s += bnd[j]; return s; }

int main() {
    if (scanf("%d %d %lld %lld %lld %lld", &n, &k, &m, &lam, &L, &U) != 6) return 0;
    w.resize(n); assign_.resize(n); S.assign(k, 0); bnd.assign(k, 0); adj.assign(n, {});
    for (int i = 0; i < n; i++) scanf("%lld", &w[i]);
    t.resize(k);
    for (int j = 0; j < k; j++) scanf("%d", &t[j]);
    ll E; scanf("%lld", &E);
    for (ll e = 0; e < E; e++) {
        int u, v; ll c; scanf("%d %d %lld", &u, &v, &c); u--; v--;
        adj[u].push_back({v, c}); adj[v].push_back({u, c});
    }
    for (int i = 0; i < n; i++) assign_[i] = i % k;
    for (int i = 0; i < n; i++) S[assign_[i]] += w[i];
    for (int i = 0; i < n; i++)
        for (auto& pr : adj[i])
            if (i < pr.first && assign_[i] != assign_[pr.first]) {
                bnd[assign_[i]] += pr.second;
                bnd[assign_[pr.first]] += pr.second;
            }

    std::mt19937_64 rng(987654321ULL);
    ll cur = baseCut();
    ll iters = min((ll)500000, (ll)6 * n);
    for (ll it = 0; it < iters; it++) {
        int u = (int)(rng() % n), v = (int)(rng() % n);
        if (assign_[u] == assign_[v]) continue;
        doSwap(u, v);
        ll nc = baseCut();
        if (nc >= cur) { cur = nc; }
        else { doSwap(u, v); }   // revert (swap back restores everything)
    }

    for (int i = 0; i < n; i++) printf("%d%c", assign_[i] + 1, i == n - 1 ? '\n' : ' ');
    return 0;
}
