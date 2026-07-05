// TIER: strong
// Multi-restart DYNAMIC dispatch list scheduling. Each restart recomputes a
// per-mission key every step (remaining work / remaining legs / next duration)
// and dispatches the best ready leg into its earliest feasible slot. Tries the
// classic rules (MWKR, LWKR, SPT, LPT, MOPNR) plus many seeded perturbations of
// most-work-remaining, and keeps the schedule with the smallest makespan.
// Deterministic (fixed RNG seed).
#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<int> o;
vector<vector<int>> M, D;

// mode: 0 MWKR, 1 LWKR, 2 SPT, 3 LPT, 4 MOPNR, 5 perturbed-MWKR (uses noise)
long long runDyn(int mode, const vector<double>& noise, vector<vector<long long>>& out) {
    vector<int> nx(n, 0);
    vector<long long> jobReady(n, 0), machFree(m, 0), remWork(n, 0);
    vector<int> remOps(n, 0);
    for (int j = 0; j < n; j++) {
        remOps[j] = o[j];
        for (int k = 0; k < o[j]; k++) remWork[j] += D[j][k];
    }
    vector<vector<long long>> st(n);
    for (int j = 0; j < n; j++) st[j].assign(o[j], 0);
    int total = 0;
    for (int j = 0; j < n; j++) total += o[j];

    for (int c = 0; c < total; c++) {
        int bj = -1; double bk = -1e300;
        for (int j = 0; j < n; j++) {
            if (nx[j] >= o[j]) continue;
            double nd = D[j][nx[j]];
            double key;
            switch (mode) {
                case 0: key = (double)remWork[j]; break;              // most work remaining
                case 1: key = -(double)remWork[j]; break;             // least work remaining
                case 2: key = -nd; break;                             // shortest processing time
                case 3: key =  nd; break;                             // longest processing time
                case 4: key = (double)remOps[j]; break;               // most ops remaining
                default: key = (double)remWork[j] * noise[j]; break;  // perturbed MWKR
            }
            if (key > bk || (key == bk && (bj < 0 || j < bj))) { bk = key; bj = j; }
        }
        int j = bj, k = nx[j], mch = M[j][k];
        long long s = max(jobReady[j], machFree[mch]);
        st[j][k] = s;
        jobReady[j] = s + D[j][k];
        machFree[mch] = s + D[j][k];
        remWork[j] -= D[j][k];
        remOps[j]--;
        nx[j]++;
    }
    long long mk = 0;
    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            mk = max(mk, st[j][k] + D[j][k]);
    out = st;
    return mk;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    o.resize(n); M.resize(n); D.resize(n);
    for (int j = 0; j < n; j++) {
        scanf("%d", &o[j]);
        M[j].resize(o[j]); D[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) scanf("%d %d", &M[j][k], &D[j][k]);
    }

    vector<vector<long long>> best, cur;
    long long bestmk = LLONG_MAX;
    vector<double> noise(n, 1.0);
    auto consider = [&](int mode) {
        long long mk = runDyn(mode, noise, cur);
        if (mk < bestmk) { bestmk = mk; best = cur; }
    };

    for (int mode = 0; mode <= 4; mode++) consider(mode);  // MWKR, LWKR, SPT, LPT, MOPNR

    mt19937 rng(20260701u);
    for (int it = 0; it < 400; it++) {
        for (int j = 0; j < n; j++) noise[j] = 0.5 + (double)(rng() % 1000) / 1000.0;  // in [0.5,1.5)
        consider(5);  // perturbed most-work-remaining
    }

    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("%lld%c", best[j][k], k + 1 < o[j] ? ' ' : '\n');
    return 0;
}
