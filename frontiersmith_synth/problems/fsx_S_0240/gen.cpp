#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: warehouse offline replay ----
    // testId 1 tiny (example scale); grows to a longer, denser trace by testId 10.
    int T = 6 + 4 * testId;                 // 10, 14, ..., 46 steps
    int M = 3 + (testId % 4);               // 3..6 units/step throughput
    // task load: keep well under T*M so a feasible pack always exists.
    int N = (int)llround(0.60 * (double)T * (double)M);
    if (N < 5) N = 5;

    int G = 200 + 40 * (testId % 5);        // 200..360 grid (on-demand) price

    // ---- per-step cost trace (spot availability fluctuates strongly) ----
    // Each step is either CHEAP (low setup, cheap full spot) or EXPENSIVE (high setup,
    // no spot -> everything pays grid). Cheap/expensive steps are scattered in TIME, so
    // packing tasks at the earliest steps (the baseline) lands on many expensive steps,
    // while a cost-aware schedule can route work to the cheap steps.
    vector<int> f(T + 1), s(T + 1), cap(T + 1);
    // ~65% of steps cheap, enough cheap throughput to hold most tasks.
    for (int t = 1; t <= T; t++) {
        double roll = rnd.next(0.0, 1.0);
        if (roll < 0.65) {                              // CHEAP step
            f[t] = rnd.next(5, 25);
            s[t] = rnd.next(1, 6);
            cap[t] = M;
        } else {                                        // EXPENSIVE step
            f[t] = rnd.next(300, 800);
            s[t] = rnd.next(G / 2, G);
            cap[t] = (rnd.next(0, 3) == 0) ? rnd.next(1, M) : 0;
        }
    }

    // ---- feasible task windows: build a valid home assignment first ----
    // multiset of step slots, each step usable up to M times.
    vector<int> slots;
    slots.reserve(T * M);
    for (int t = 1; t <= T; t++)
        for (int c = 0; c < M; c++) slots.push_back(t);
    shuffle(slots.begin(), slots.end());
    // (N <= T*M by construction)

    vector<int> r(N), dl(N);
    for (int i = 0; i < N; i++) {
        int home = slots[i];
        // wide, heterogeneous windows: some tasks nearly free to roam (reach cheap steps),
        // some tightly pinned (must take whatever step is near their home).
        double roll = rnd.next(0.0, 1.0);
        int lo, hi;
        if (roll < 0.55) {                  // roomy window
            lo = rnd.next(1, home);
            hi = rnd.next(home, T);
        } else {                            // tight window around home
            int back = rnd.next(0, 2);
            int fwd  = rnd.next(0, 2);
            lo = max(1, home - back);
            hi = min(T, home + fwd);
        }
        r[i] = lo;
        dl[i] = hi;
    }

    // shuffle task order so input order != any natural schedule order
    vector<int> perm(N);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end());

    printf("%d %d %d\n", T, N, M);
    printf("%d\n", G);
    for (int t = 1; t <= T; t++)
        printf("%d %d %d\n", f[t], s[t], cap[t]);
    for (int i = 0; i < N; i++)
        printf("%d %d\n", r[perm[i]], dl[perm[i]]);
    return 0;
}
