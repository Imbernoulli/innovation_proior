// testlib generator: vineyard irrigation (budget-constrained max-weight independent set)
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int T = atoi(argv[1]);

    // ---- testId 1: the exact statement example (tiny sanity) ----
    if (T == 1) {
        printf("6 6 8\n");
        printf("10 8 6 6 5 4\n");
        printf("4 3 2 2 2 1\n");
        printf("1 2\n2 3\n3 4\n4 5\n5 6\n1 4\n");
        return 0;
    }

    // ---- difficulty / structure ladder for T = 2..10 ----
    int n = min(2600, 60 + (T - 1) * 280);

    // moderately-spread yields (skew cycles so the distribution varies across tests)
    double skew = 1.2 + (T % 3) * 0.7;              // 1.2, 1.9, 2.6, ...
    vector<int> w(n + 1), d(n + 1);
    long long sumd = 0, maxd = 0;
    for (int i = 1; i <= n; i++) {
        double u = rnd.next(0.0, 1.0);
        w[i] = 200 + (int)(pow(u, skew) * 800.0);   // in [200,1000]
        // water demand correlated with yield (heavy plots cost more water)
        d[i] = 1 + w[i] / 20 + rnd.next(0, 12);     // roughly [11,62]
        sumd += d[i];
        maxd = max<long long>(maxd, d[i]);
    }

    // conflict graph: planted cliques + random edges; density varies with T
    set<pair<int,int>> es;
    auto add = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        es.insert({a, b});
    };
    long long target = min<long long>(24000, (long long)n * (7 + (T % 4) * 3));
    long long maxEdges = (long long)n * (n - 1) / 2;
    target = min(target, (long long)(maxEdges * 0.55));

    // half the edges from planted cliques (forces at-most-one selection per clique)
    long long attempts = 0, attemptCap = 20LL * target + 100000;
    while ((long long)es.size() < target / 2 && attempts++ < attemptCap) {
        int k = rnd.next(3, 6);
        vector<int> vs;
        for (int j = 0; j < k; j++) vs.push_back(rnd.next(1, n));
        for (int a = 0; a < k; a++)
            for (int b = a + 1; b < k; b++) add(vs[a], vs[b]);
    }
    // remaining edges random
    attempts = 0;
    while ((long long)es.size() < target && attempts++ < attemptCap) {
        add(rnd.next(1, n), rnd.next(1, n));
    }

    // reservoir budget sized to admit only a small handful of plots, so reaching the
    // 10x-baseline cap requires genuine optimization. K = target plot count (varies with T).
    double avgd = (double)sumd / (double)n;
    double K = 6.0 + (T - 2) * 0.7;                 // T2:6.0 ... T10:11.6
    long long W = max<long long>(maxd, (long long)llround(K * avgd));

    int m = (int)es.size();
    printf("%d %d %lld\n", n, m, W);
    for (int i = 1; i <= n; i++) printf("%d%c", w[i], i == n ? '\n' : ' ');
    for (int i = 1; i <= n; i++) printf("%d%c", d[i], i == n ? '\n' : ' ');
    for (auto& e : es) printf("%d %d\n", e.first, e.second);
    return 0;
}
