// TIER: strong
// Greedy majority init + exact incremental flip local search to a local optimum,
// with a handful of seeded random restarts; keep the best satisfied-weight plan.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<vector<int>> cl;                 // literals per clause
vector<vector<pair<int,int>>> occ;      // var -> (clause, sign +1/-1)

// state
vector<int> x;                          // 1..n assignment
vector<int> satCount;                   // per clause
ll total;                               // satisfied weight

static inline bool litSat(int lit, const vector<int>& a) {
    int v = abs(lit);
    return lit > 0 ? (a[v] == 1) : (a[v] == 0);
}

void rebuild() {
    satCount.assign(m, 0);
    total = 0;
    for (int c = 0; c < m; c++) {
        int s = 0;
        for (int l : cl[c]) if (litSat(l, x)) s++;
        satCount[c] = s;
        if (s > 0) total += w[c];
    }
}

// delta in satisfied weight if we flip variable v (does not modify state)
ll flipDelta(int v) {
    ll d = 0;
    for (auto& pr : occ[v]) {
        int c = pr.first, sign = pr.second;
        // literal of v in c is satisfied now?
        bool nowSat = sign > 0 ? (x[v] == 1) : (x[v] == 0);
        int nc = satCount[c] + (nowSat ? -1 : +1);
        bool oldClause = satCount[c] > 0;
        bool newClause = nc > 0;
        d += (ll)w[c] * ((newClause ? 1 : 0) - (oldClause ? 1 : 0));
    }
    return d;
}

void applyFlip(int v) {
    for (auto& pr : occ[v]) {
        int c = pr.first, sign = pr.second;
        bool nowSat = sign > 0 ? (x[v] == 1) : (x[v] == 0);
        bool oldClause = satCount[c] > 0;
        satCount[c] += (nowSat ? -1 : +1);
        bool newClause = satCount[c] > 0;
        if (newClause && !oldClause) total += w[c];
        else if (!newClause && oldClause) total -= w[c];
    }
    x[v] ^= 1;
}

void localSearch() {
    rebuild();
    bool improved = true;
    int passes = 0;
    while (improved && passes < 40) {
        improved = false;
        passes++;
        for (int v = 1; v <= n; v++) {
            if (flipDelta(v) > 0) { applyFlip(v); improved = true; }
        }
    }
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.resize(m);
    cl.resize(m);
    occ.assign(n + 1, {});
    vector<ll> posW(n + 1, 0), negW(n + 1, 0);
    for (int c = 0; c < m; c++) {
        int L;
        scanf("%lld %d", &w[c], &L);
        cl[c].resize(L);
        for (int j = 0; j < L; j++) {
            int t; scanf("%d", &t);
            cl[c][j] = t;
            int v = abs(t);
            occ[v].push_back({c, t > 0 ? 1 : -1});
            if (t > 0) posW[v] += w[c]; else negW[v] += w[c];
        }
    }

    x.assign(n + 1, 0);
    // init 0: greedy majority
    for (int i = 1; i <= n; i++) x[i] = (posW[i] > negW[i]) ? 1 : 0;
    localSearch();
    vector<int> best = x;
    ll bestW = total;

    // seeded random restarts
    std::mt19937 rng(987654321u);
    int restarts = (n <= 400 ? 6 : 3);
    for (int r = 0; r < restarts; r++) {
        for (int i = 1; i <= n; i++) x[i] = rng() & 1;
        localSearch();
        if (total > bestW) { bestW = total; best = x; }
    }

    for (int i = 1; i <= n; i++)
        printf("%d%c", best[i], i == n ? '\n' : ' ');
    return 0;
}
