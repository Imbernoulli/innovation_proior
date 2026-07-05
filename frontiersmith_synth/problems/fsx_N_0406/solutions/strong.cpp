// TIER: strong
// Hub-aware local search over GF(q) labels. Several sweeps of node best-response
// where the score of a candidate label INCLUDES the all-or-nothing hub bonuses of
// the node itself and of its hub neighbours; multiple seeded restarts, keep best.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

int n, m, q, H;
vector<int> start_, adjNb;
vector<ll> adjW;
vector<u64> adjM;
vector<char> isHub;
vector<ll> bonusOf;
vector<vector<int>> hubNeigh;   // distinct hub-neighbours per node

static inline bool edgeClear(u64 mask, int a, int b) {
    int r = (a + b) % q;
    return !((mask >> r) & 1ULL);
}

// full objective of labeling x
ll fullObj(const vector<int>& x) {
    ll F = 0;
    // edges appear twice in CSR; count each once via nb>u OR handle by /2? Use raw
    // edge pass instead: iterate CSR and only count when u<nb, plus self handled.
    for (int u = 1; u <= n; u++)
        for (int p = start_[u]; p < start_[u + 1]; p++) {
            int v = adjNb[p];
            if (u < v) { if (edgeClear(adjM[p], x[u], x[v])) F += adjW[p]; }
            else if (u == v) { /* no self loops */ }
        }
    // hubs
    for (int u = 1; u <= n; u++) if (isHub[u]) {
        bool ok = true;
        for (int p = start_[u]; p < start_[u + 1]; p++)
            if (!edgeClear(adjM[p], x[u], x[adjNb[p]])) { ok = false; break; }
        if (ok) F += bonusOf[u];
    }
    return F;
}

// score of assigning label L to node u given current x (relative, for argmax)
ll localScore(int u, int L, const vector<int>& x) {
    ll s = 0;
    for (int p = start_[u]; p < start_[u + 1]; p++)
        if (edgeClear(adjM[p], L, x[adjNb[p]])) s += adjW[p];
    if (isHub[u]) {
        bool ok = true;
        for (int p = start_[u]; p < start_[u + 1]; p++)
            if (!edgeClear(adjM[p], L, x[adjNb[p]])) { ok = false; break; }
        if (ok) s += bonusOf[u];
    }
    for (int v : hubNeigh[u]) {
        bool ok = true;
        for (int p = start_[v]; p < start_[v + 1]; p++) {
            int w = adjNb[p];
            int lw = (w == u) ? L : x[w];
            if (!edgeClear(adjM[p], x[v], lw)) { ok = false; break; }
        }
        if (ok) s += bonusOf[v];
    }
    return s;
}

int main() {
    if (scanf("%d %d %d %d", &n, &m, &q, &H) != 4) return 0;
    vector<int> eu(m), ev(m); vector<ll> ew(m); vector<u64> emask(m);
    vector<int> deg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v, k; ll w;
        scanf("%d %d %lld %d", &u, &v, &w, &k);
        u64 mask = 0;
        for (int j = 0; j < k; j++) { int f; scanf("%d", &f); mask |= (1ULL << f); }
        eu[i] = u; ev[i] = v; ew[i] = w; emask[i] = mask; deg[u]++; deg[v]++;
    }
    isHub.assign(n + 1, 0); bonusOf.assign(n + 1, 0);
    vector<int> hn(H); vector<ll> hg(H);
    for (int i = 0; i < H; i++) { scanf("%d %lld", &hn[i], &hg[i]); isHub[hn[i]] = 1; bonusOf[hn[i]] = hg[i]; }

    start_.assign(n + 2, 0);
    for (int i = 1; i <= n; i++) start_[i + 1] = start_[i] + deg[i];
    int tot = start_[n + 1];
    adjNb.resize(tot); adjW.resize(tot); adjM.resize(tot);
    vector<int> cur(n + 2); for (int i = 1; i <= n; i++) cur[i] = start_[i];
    for (int i = 0; i < m; i++) {
        int u = eu[i], v = ev[i];
        int p = cur[u]++; adjNb[p] = v; adjW[p] = ew[i]; adjM[p] = emask[i];
        int p2 = cur[v]++; adjNb[p2] = u; adjW[p2] = ew[i]; adjM[p2] = emask[i];
    }
    // distinct hub-neighbours per node
    hubNeigh.assign(n + 1, {});
    for (int u = 1; u <= n; u++) {
        for (int p = start_[u]; p < start_[u + 1]; p++) {
            int v = adjNb[p];
            if (v != u && isHub[v]) hubNeigh[u].push_back(v);
        }
        auto& hv = hubNeigh[u];
        sort(hv.begin(), hv.end()); hv.erase(unique(hv.begin(), hv.end()), hv.end());
    }

    mt19937 rng(987654321u);
    auto sweep = [&](vector<int>& x, int passes) {
        for (int pass = 0; pass < passes; pass++)
            for (int u = 1; u <= n; u++) {
                ll best = -1; int bl = x[u];
                for (int L = 0; L < q; L++) {
                    ll sc = localScore(u, L, x);
                    if (sc > best) { best = sc; bl = L; }
                }
                x[u] = bl;
            }
    };

    vector<int> bestX(n + 1, 0);
    ll bestF = -1;
    for (int restart = 0; restart < 3; restart++) {
        vector<int> x(n + 1, 0);
        if (restart == 1) { for (int i = 1; i <= n; i++) x[i] = rng() % q; }
        if (restart == 2) {
            // greedy warm start ignoring hubs (one pass)
            for (int u = 1; u <= n; u++) {
                ll best = -1; int bl = 0;
                for (int L = 0; L < q; L++) {
                    ll s = 0;
                    for (int p = start_[u]; p < start_[u + 1]; p++)
                        if (edgeClear(adjM[p], L, x[adjNb[p]])) s += adjW[p];
                    if (s > best) { best = s; bl = L; }
                }
                x[u] = bl;
            }
        }
        sweep(x, 5);
        ll f = fullObj(x);
        if (f > bestF) { bestF = f; bestX = x; }
    }

    for (int i = 1; i <= n; i++) printf("%d ", bestX[i]);
    printf("\n");
    return 0;
}
