#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // testId is a difficulty/structure ladder.
    int n = (t <= 1) ? 3 : t * 120;      // up to 1200
    int m = (t <= 1) ? 6 : t * 500;      // up to 5000
    if (n < 2) n = 2;
    if (m < 1) m = 1;

    bool dense = (t % 3 == 0);           // cluster requirements on few stations -> conflict
    bool skew  = (t % 2 == 0);           // heavy-tailed revenue

    int pool = dense ? max(2, n / 4) : n;
    int kmax = min(4, pool);
    int forcedZero = max(1, m / 50);     // guarantee B > 0 (all-0 opportunities)

    printf("%d %d\n", n, m);
    for (int j = 0; j < m; j++) {
        int k = rnd.next(1, kmax);
        // sample k distinct stations from [1, pool]
        set<int> st;
        while ((int)st.size() < k) st.insert(rnd.next(1, pool));

        long long w;
        if (skew) {
            int r = rnd.next(1, 100);
            w = (r <= 5) ? rnd.next(500, 1000) : rnd.next(1, 50);
        } else {
            w = rnd.next(1, 1000);
        }

        bool forceZero = (j < forcedZero);
        printf("%d", k);
        for (int s : st) {
            int o = forceZero ? 0 : rnd.next(0, 1);
            printf(" %d %d", s, o);
        }
        printf(" %lld\n", w);
    }
    return 0;
}
