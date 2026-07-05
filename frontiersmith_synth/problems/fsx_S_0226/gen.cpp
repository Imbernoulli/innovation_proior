#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // testId is a difficulty/structure ladder.
    int n = (t <= 1) ? 5 : t * 400;         // up to 4000
    int m = (t <= 1) ? 6 : t * 5000;        // up to 50000
    int D = (t <= 1) ? 3 : min(12, 2 + t);  // domain grows; weaker uniform baseline as D grows
    if (n < 2) n = 2;
    if (m < 1) m = 1;
    if (D < 2) D = 2;

    bool dense = (t % 3 == 0);              // cluster cues on few markers -> conflict
    bool skew  = (t % 2 == 0);             // heavy-tailed mapping value

    int pool = dense ? max(2, n / 5) : n;  // shared markers create contention
    int kmax = min(5, pool);

    // A skewed color distribution on some tests: some colors more common than others,
    // which makes the best-uniform baseline stronger and forces genuine per-marker search.
    bool colorSkew = (t % 4 == 0);

    printf("%d %d %d\n", n, m, D);
    for (int j = 0; j < m; j++) {
        int k = rnd.next(1, kmax);
        // sample k distinct markers from [1, pool]
        set<int> mk;
        while ((int)mk.size() < k) mk.insert(rnd.next(1, pool));

        long long w;
        if (skew) {
            int r = rnd.next(1, 100);
            w = (r <= 5) ? rnd.next(500, 1000) : rnd.next(1, 60);
        } else {
            w = rnd.next(1, 1000);
        }

        printf("%d", k);
        for (int v : mk) {
            int c;
            if (colorSkew) {
                // bias toward lower color codes
                int r = rnd.next(0, D * D - 1);
                c = (int)(sqrt((double)r));
                if (c >= D) c = D - 1;
            } else {
                c = rnd.next(0, D - 1);
            }
            printf(" %d %d", v, c);
        }
        printf(" %lld\n", w);
    }
    return 0;
}
