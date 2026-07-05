#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // size ladder: (N tracers, M clusters)
    int Ns[] = {0, 2, 4, 6, 8, 12, 16, 24, 32, 40, 60};
    int Ms[] = {0, 3, 10, 30, 80, 200, 400, 800, 1200, 2500, 5000};
    int N = Ns[testId];
    int M = Ms[testId];
    if (N < 1) N = 1;
    if (M < 1) M = 1;

    int reg = testId % 4; // 4 structural regimes

    // fill factor: fraction of total demand the total budget can cover.
    // smaller => tighter => choices matter more.
    double fill;
    if (reg == 0) fill = 0.40;
    else if (reg == 1) fill = 0.35;
    else if (reg == 2) fill = 0.40;
    else fill = 0.25;

    // cost / value ranges
    int TMAX = 9;              // per-pair contact-hour cost
    // cluster base severity (skewed for reg 1,3)
    vector<long long> w(M);
    for (int i = 0; i < M; i++) {
        if (reg == 0) {
            w[i] = rnd.next(1, 50);                 // uniform-ish severity
        } else if (reg == 1) {
            // heavy-tailed severity: a few clusters worth a lot
            if (rnd.next(0, 9) == 0) w[i] = rnd.next(120, 400);
            else w[i] = rnd.next(1, 30);
        } else if (reg == 2) {
            w[i] = rnd.next(5, 60);                 // moderate; value tied to cost below
        } else {
            if (rnd.next(0, 7) == 0) w[i] = rnd.next(200, 600);
            else w[i] = rnd.next(1, 20);
        }
    }

    // per-pair cost t[i][j] and value v[i][j]
    // store row-by-row to keep memory small
    // capacities computed from expected demand
    // expected cost of serving a cluster ~ (TMAX+1)/2
    double avgT = (TMAX + 1) / 2.0;
    // we plan to be able to serve about (fill*M) clusters total across all tracers
    double servable = fill * (double)M;
    long long totalBudget = (long long)llround(servable * avgT);
    if (totalBudget < N) totalBudget = N; // at least 1 per tracer
    long long Cbase = totalBudget / N;
    if (Cbase < TMAX) Cbase = TMAX; // each tracer can afford at least one cluster

    // print header
    printf("%d %d\n", N, M);
    for (int j = 0; j < N; j++) {
        // vary capacities a bit around Cbase for asymmetry
        long long c;
        long long spread = max<long long>(1, Cbase / 3);
        c = Cbase + rnd.next(-(int)min<long long>(spread, 1000000000LL),
                              (int)min<long long>(spread, 1000000000LL));
        if (c < TMAX) c = TMAX;
        if (c > 1000000000LL) c = 1000000000LL;
        printf("%lld%c", c, j + 1 < N ? ' ' : '\n');
    }

    // emit clusters
    string line;
    line.reserve((size_t)N * 8);
    for (int i = 0; i < M; i++) {
        line.clear();
        for (int j = 0; j < N; j++) {
            int t = rnd.next(1, TMAX);
            long long v;
            if (reg == 2) {
                // value correlated with cost: expensive pairs are worth more,
                // so a naive first-fit (cost-blind, value-blind) wastes budget.
                v = w[i] + (long long)t * rnd.next(2, 6) + rnd.next(0, 5);
            } else if (reg == 0) {
                // pairs largely independent of cluster severity
                v = rnd.next(1, 50);
            } else {
                // value tracks cluster severity with tracer-specific noise
                long long noise = rnd.next(0, (int)max<long long>(1, w[i] / 4));
                v = max<long long>(1, w[i] - noise + rnd.next(0, 4));
            }
            if (v < 1) v = 1;
            if (v > 1000000) v = 1000000;
            line += to_string(t);
            line += ' ';
            line += to_string(v);
            line += (j + 1 < N ? ' ' : '\n');
        }
        fputs(line.c_str(), stdout);
    }
    return 0;
}
