// TIER: strong
// Coordinate-ascent local search with seeded random restarts.
// Start from a per-act greedy, then repeatedly flip any act whose move strictly
// increases satisfied weight (best-improvement sweeps) until a local optimum.
// Several deterministic restarts from perturbed/random assignments keep the best
// placement found. Beats the one-pass greedy by escaping its single-sweep myopia.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> cw;
vector<vector<int>> clit;
vector<vector<int>> occ;
vector<int> litOfVarInClause; // unused

vector<int> a;
vector<int> satc;
ll curTotal;

// precomputed sign of var v inside each of its clauses, aligned with occ[v]
vector<vector<int>> occSign; // +1 or -1

void buildSat() {
    for (int c = 0; c < m; c++) {
        int s = 0;
        for (int L : clit[c]) {
            int v = abs(L);
            bool truth = (L > 0) ? (a[v] == 1) : (a[v] == 0);
            if (truth) s++;
        }
        satc[c] = s;
    }
    curTotal = 0;
    for (int c = 0; c < m; c++) if (satc[c] > 0) curTotal += cw[c];
}

ll gainFlip(int v) {
    ll delta = 0;
    int cur = a[v];
    const auto& cs = occ[v];
    const auto& sg = occSign[v];
    for (size_t t = 0; t < cs.size(); t++) {
        int c = cs[t]; int sign = sg[t];
        bool truthNow = (sign > 0) ? (cur == 1) : (cur == 0);
        if (truthNow) { if (satc[c] == 1) delta -= cw[c]; }
        else          { if (satc[c] == 0) delta += cw[c]; }
    }
    return delta;
}

void applyFlip(int v) {
    int cur = a[v];
    const auto& cs = occ[v];
    const auto& sg = occSign[v];
    for (size_t t = 0; t < cs.size(); t++) {
        int c = cs[t]; int sign = sg[t];
        bool truthNow = (sign > 0) ? (cur == 1) : (cur == 0);
        if (truthNow) { satc[c] -= 1; if (satc[c] == 0) curTotal -= cw[c]; }
        else          { if (satc[c] == 0) curTotal += cw[c]; satc[c] += 1; }
    }
    a[v] = 1 - cur;
}

void localSearch(int maxPasses) {
    for (int pass = 0; pass < maxPasses; pass++) {
        bool moved = false;
        for (int v = 1; v <= n; v++) {
            if (gainFlip(v) > 0) { applyFlip(v); moved = true; }
        }
        if (!moved) break;
    }
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    cw.resize(m); clit.resize(m); occ.assign(n + 1, {}); occSign.assign(n + 1, {});
    for (int c = 0; c < m; c++) {
        int k, w; scanf("%d %d", &k, &w); cw[c] = w; clit[c].resize(k);
        for (int t = 0; t < k; t++) {
            int L; scanf("%d", &L); clit[c][t] = L;
            occ[abs(L)].push_back(c);
            occSign[abs(L)].push_back(L > 0 ? 1 : -1);
        }
    }
    a.assign(n + 1, 0);
    satc.assign(m, 0);

    // ---- greedy init: appeal-ordered one pass, then local search ----
    vector<ll> appeal(n + 1, 0);
    for (int c = 0; c < m; c++) for (int L : clit[c]) if (L > 0) appeal[L] += cw[c];
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int x, int y){ return appeal[x] > appeal[y]; });

    buildSat();
    for (int v : order) if (gainFlip(v) > 0) applyFlip(v);
    localSearch(60);

    // record best
    vector<int> best = a;
    ll bestTotal = curTotal;

    // ---- deterministic random restarts ----
    std::mt19937 rng(1234567u);
    int restarts = 6;
    // scale down restarts for very large instances to respect the time limit
    long long lits = 0; for (int c = 0; c < m; c++) lits += (ll)clit[c].size();
    if ((ll)n * (restarts) > 2000000LL && lits > 200000) restarts = 3;

    for (int r = 0; r < restarts; r++) {
        if (r == 0) {
            // biased random: acts lean Main (matches the positive-unit pressure)
            for (int v = 1; v <= n; v++) a[v] = (rng() % 4 == 0) ? 0 : 1;
        } else {
            // perturb the current best: random flips of ~10% of acts
            a = best;
            int flips = max(1, n / 10);
            for (int f = 0; f < flips; f++) { int v = 1 + rng() % n; a[v] ^= 1; }
        }
        buildSat();
        localSearch(60);
        if (curTotal > bestTotal) { bestTotal = curTotal; best = a; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i == n ? '\n' : ' ');
    return 0;
}
