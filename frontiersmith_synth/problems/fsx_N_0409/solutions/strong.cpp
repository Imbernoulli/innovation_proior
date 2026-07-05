// TIER: strong
// Full-objective local search on the TRUE merit F = sum_j Bnd_j*(1+lam*tuned_j).
// Start from round-robin; run size-preserving swaps (always feasible). Half the
// attempts swap two LIGHT (edge-free) adjuster cars -- these leave the cut
// untouched and only shift track residues, so they cheaply tune high-boundary
// tracks for the (1+lam) multiplier. The other half swaps arbitrary cars,
// improving the cut and occasionally tuning. Accept iff F does not decrease.
// This exploits BOTH mechanisms, beating the cut-only greedy and the baseline.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, k; ll m, lam, L, U;
vector<ll> w;
vector<ll> t;
vector<int> assign_;
vector<ll> S, bnd;
vector<vector<pair<int, ll>>> adj;
vector<int> lightList;

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
static inline void doSwap(int u, int v) {
    int a = assign_[u], b = assign_[v];
    moveCar(u, b); moveCar(v, a);
}
static inline ll objF() {
    ll F = 0;
    for (int j = 0; j < k; j++) {
        int tuned = ((S[j] % m) == t[j]) ? 1 : 0;
        F += bnd[j] * (1 + lam * (ll)tuned);
    }
    return F;
}

int main() {
    if (scanf("%d %d %lld %lld %lld %lld", &n, &k, &m, &lam, &L, &U) != 6) return 0;
    w.resize(n); assign_.resize(n); S.assign(k, 0); bnd.assign(k, 0); adj.assign(n, {});
    for (int i = 0; i < n; i++) scanf("%lld", &w[i]);
    t.resize(k);
    for (int j = 0; j < k; j++) scanf("%lld", &t[j]);
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
    for (int i = 0; i < n; i++) if (adj[i].empty()) lightList.push_back(i);

    std::mt19937_64 rng(123456789ULL);
    ll cur = objF();
    int nl = (int)lightList.size();
    ll iters = min((ll)1500000, (ll)25 * n);
    for (ll it = 0; it < iters; it++) {
        int u, v;
        if (nl >= 2 && (rng() & 1)) {           // tuning move: two light adjusters
            u = lightList[rng() % nl];
            v = lightList[rng() % nl];
        } else {                                 // structural move: any two cars
            u = (int)(rng() % n);
            v = (int)(rng() % n);
        }
        if (assign_[u] == assign_[v]) continue;
        doSwap(u, v);
        ll nf = objF();
        if (nf >= cur) { cur = nf; }
        else { doSwap(u, v); }                   // revert
    }

    for (int i = 0; i < n; i++) printf("%d%c", assign_[i] + 1, i == n - 1 ? '\n' : ' ');
    return 0;
}
