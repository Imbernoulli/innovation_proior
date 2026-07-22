#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Quay Cranes on One Rail (monotone matching + congested-stretch staggering).
// testId 1..10 is a difficulty/structure ladder.
//   UNIFORM tests (idx 1..5): positions are spread with EVERY gap >= S (no congestion is
//   possible), and hoisting work is heavily skewed.  Here the load-balancing greedy cleanly
//   beats the even-count baseline (balancing work vs count matters a lot), while the gap-aware
//   strong solution adds only the power-matching edge -> greedy > trivial holds.
//   TRAP tests (idx 6..10): jobs form tight clusters whose INTERNAL spacing is < S while the
//   gaps BETWEEN clusters are >> S.  Total work is skewed across clusters, so a cut that
//   balances *work* must slice through a cluster (creating a congested stretch that serializes
//   cranes and inflates the makespan), whereas cutting at the wide inter-cluster gaps stays
//   parallel.  This is where the obvious greedy lands far from strong.
// Standby powers p_c are heterogeneous so which cranes are deployed on which zone matters.

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ks[10] = {2, 3, 3, 4, 4, 5, 5, 6, 7, 8};
    int Ms[10] = {8, 40, 120, 300, 500, 900, 1500, 2500, 3800, 5000};
    int K = Ks[idx - 1];
    int M = Ms[idx - 1];

    bool trap = (idx >= 6);          // 5 uniform (1..5), 5 traps (6..10)

    int S = 50;                      // safety distance (small: hoisting work dominates travel)
    int alpha = 1;                   // travel cost per unit span
    int gamma = 1 + (idx % 3);       // idle weight 1..3

    // ---- standby powers p_c (heterogeneous) ----
    vector<int> p(K);
    for (int c = 0; c < K; c++) p[c] = rnd.next(1, 5);
    if (idx == 3 || idx == 5 || idx == 8) p[rnd.next(0, K - 1)] = 5;   // dominant crane

    vector<long long> pos;
    vector<int> work;

    if (trap) {
        int C = K + rnd.next(1, 2);              // more clusters than cranes
        vector<int> csz(C, 0);
        int rem = M;
        for (int c = 0; c < C; c++) {
            int lo = 1, hi = max(1, rem - (C - 1 - c));
            int share = (c == C - 1) ? rem : rnd.next(lo, hi);
            share = min(share, rem - (C - 1 - c));
            share = max(1, share);
            csz[c] = share; rem -= share;
        }
        vector<int> heavy(C, 0);
        heavy[rnd.next(0, C - 1)] = 1;
        if (C >= 4) heavy[rnd.next(0, C - 1)] = 1;

        long long cur = rnd.next(0, 300);
        for (int c = 0; c < C; c++) {
            long long q = cur;
            for (int t = 0; t < csz[c]; t++) {
                q += rnd.next(1, max(1, S / 4));       // internal gap < S -> congested
                pos.push_back(q);
                int w = heavy[c] ? rnd.next(600, 1000) : rnd.next(1, 120);
                work.push_back(w);
            }
            cur = q + rnd.next(2 * S, 6 * S);          // wide inter-cluster gap
        }
    } else {
        // Uniform CLEAN-SPREAD: every consecutive gap >= S, so no accidental congestion.
        // Work is large relative to the per-job travel (gap ~ [S,3S)), so balancing WORK
        // (greedy) genuinely beats balancing count/span (the even-count baseline).
        long long q = rnd.next(0, 200);
        for (int i = 0; i < M; i++) {
            q += rnd.next(S, 3 * S);                   // gap in [S, 3S) -> always clean
            pos.push_back(q);
            // heavily skewed work: ~12% very heavy, rest moderate -> count-balance is poor
            if (rnd.next(0, 99) < 12) work.push_back(rnd.next(1500, 3000));
            else                      work.push_back(rnd.next(80, 300));
        }
    }

    int N = (int)pos.size();
    // rescale positions into [0,200000] preserving order + distinctness if needed
    long long mx = 0; for (auto v : pos) mx = max(mx, v);
    if (mx > 200000) {
        vector<pair<long long,int>> pr(N);
        for (int i = 0; i < N; i++) pr[i] = {pos[i], i};
        sort(pr.begin(), pr.end());
        for (int r = 0; r < N; r++)
            pos[pr[r].second] = (long long)((double)r / max(1, N - 1) * 200000.0);
        vector<pair<long long,int>> pr2(N);
        for (int i = 0; i < N; i++) pr2[i] = {pos[i], i};
        sort(pr2.begin(), pr2.end());
        long long last = -1;
        for (int r = 0; r < N; r++) {
            long long v = pos[pr2[r].second];
            if (v <= last) v = last + 1;
            pos[pr2[r].second] = v; last = v;
        }
    }

    // shuffle input order so the monotone rule is non-trivial
    vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i;
    shuffle(perm.begin(), perm.end());

    printf("%d %d %d %d %d\n", K, N, S, alpha, gamma);
    for (int c = 0; c < K; c++) printf("%d%c", p[c], c + 1 < K ? ' ' : '\n');
    for (int k = 0; k < N; k++) {
        int i = perm[k];
        printf("%lld %d\n", pos[i], work[i]);
    }
    return 0;
}
