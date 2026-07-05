// TIER: strong
// Multi-restart hill-climbing local search on weighted Max-SAT.
// Incremental flip gains; start from the greedy weighted-majority assignment
// plus several seeded random restarts; keep the best assignment found.
#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<int> cw;
vector<vector<int>> lit;        // clause -> literals
vector<vector<pair<int,int>>> inc; // var -> list of (clauseIdx, +1 if literal positive else 0)

static long long total_sat(const vector<int>& x) {
    long long F = 0;
    for (int j = 0; j < m; j++) {
        bool sat = false;
        for (int L : lit[j]) {
            int v = abs(L);
            bool lt = (L > 0) ? (x[v] == 1) : (x[v] == 0);
            if (lt) { sat = true; break; }
        }
        if (sat) F += cw[j];
    }
    return F;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    cw.resize(m);
    lit.assign(m, {});
    inc.assign(n + 1, {});
    for (int j = 0; j < m; j++) {
        int w, k; scanf("%d %d", &w, &k);
        cw[j] = w;
        lit[j].resize(k);
        for (int i = 0; i < k; i++) {
            int L; scanf("%d", &L);
            lit[j][i] = L;
            inc[abs(L)].push_back({j, L > 0 ? 1 : 0});
        }
    }

    // number of true literals per clause for a given assignment
    vector<int> satCnt(m, 0);
    auto recompute = [&](const vector<int>& x) {
        for (int j = 0; j < m; j++) {
            int c = 0;
            for (int L : lit[j]) {
                int v = abs(L);
                bool lt = (L > 0) ? (x[v] == 1) : (x[v] == 0);
                if (lt) c++;
            }
            satCnt[j] = c;
        }
    };

    // gain of flipping variable v given current x and satCnt
    auto flipGain = [&](const vector<int>& x, int v) -> long long {
        long long g = 0;
        for (auto& pr : inc[v]) {
            int j = pr.first, posLit = pr.second;
            // current truth of this literal:
            bool lt = (posLit == 1) ? (x[v] == 1) : (x[v] == 0);
            int s = satCnt[j];
            int sNew = s + (lt ? -1 : +1);
            bool before = (s > 0), after = (sNew > 0);
            if (after && !before) g += cw[j];
            else if (before && !after) g -= cw[j];
        }
        return g;
    };

    auto applyFlip = [&](vector<int>& x, int v) {
        for (auto& pr : inc[v]) {
            int j = pr.first, posLit = pr.second;
            bool lt = (posLit == 1) ? (x[v] == 1) : (x[v] == 0);
            satCnt[j] += (lt ? -1 : +1);
        }
        x[v] ^= 1;
    };

    // greedy weighted-majority start
    vector<long long> posW(n + 1, 0), negW(n + 1, 0);
    for (int j = 0; j < m; j++)
        for (int L : lit[j]) {
            if (L > 0) posW[abs(L)] += cw[j]; else negW[abs(L)] += cw[j];
        }
    vector<int> greedyStart(n + 1, 0);
    for (int v = 1; v <= n; v++) greedyStart[v] = (posW[v] >= negW[v]) ? 1 : 0;

    unsigned long long rng = 0x9e3779b97f4a7c15ULL;
    auto nextRand = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };

    vector<int> best;
    long long bestF = -1;

    int restarts = 10;
    for (int r = 0; r < restarts; r++) {
        vector<int> x(n + 1, 0);
        if (r == 0) x = greedyStart;
        else for (int v = 1; v <= n; v++) x[v] = (int)(nextRand() & 1ULL);
        recompute(x);

        // hill climb: repeatedly flip the variable with max positive gain
        int maxPasses = 200;
        for (int pass = 0; pass < maxPasses; pass++) {
            long long bg = 0; int bv = -1;
            for (int v = 1; v <= n; v++) {
                long long g = flipGain(x, v);
                if (g > bg) { bg = g; bv = v; }
            }
            if (bv == -1) break; // local optimum
            applyFlip(x, bv);
        }
        long long F = total_sat(x);
        if (F > bestF) { bestF = F; best = x; }
    }

    for (int v = 1; v <= n; v++)
        printf("%d%c", best[v], v == n ? '\n' : ' ');
    if (n == 0) printf("\n");
    return 0;
}
