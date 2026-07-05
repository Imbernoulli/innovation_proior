#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Wildlife-corridor JSSP under total weighted completion time.
// Difficulty ladder: testId 1 = tiny example scale; grows to a "large", dense,
// skewed instance by testId 10.  Corridors = jobs, crews = machines, segments = ops.
//
// Each corridor visits every crew once (a square job-shop) so contention is high and
// serial-vs-parallel improvement stays in a discriminating range.  Higher testId adds
// a bottleneck crew (long segments) and skewed priority weights (a few urgent corridors).
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int n, m;
    if (t <= 1) {                 // tiny sanity instance (example scale)
        n = 2; m = 2;
    } else {
        m = min(10, 2 + t);       // 4 .. 10 crews
        n = min(60, 4 + 5 * t);   // 14 .. 54 corridors
    }
    int dmax = 99;

    // Bottleneck crew (only for larger tests): its segments run long.
    int bottleneck = (t >= 5) ? rnd.next(0, m - 1) : -1;

    printf("%d %d\n", n, m);
    for (int j = 0; j < n; j++) {
        // random crew visiting order (deterministic testlib shuffle)
        vector<int> crew(m);
        for (int k = 0; k < m; k++) crew[k] = k;
        shuffle(crew.begin(), crew.end());

        // priority weight: skewed at high testId (a few very urgent corridors)
        int w;
        if (t >= 5 && rnd.next(0, 4) == 0)
            w = rnd.next(70, 100);
        else if (t >= 5)
            w = rnd.next(1, 20);
        else
            w = rnd.next(1, 100);

        printf("%d %d", m, w);
        for (int k = 0; k < m; k++) {
            int c = crew[k];
            int d;
            if (c == bottleneck)
                d = rnd.next(dmax / 2, dmax);          // long bottleneck segment
            else if (t >= 6 && rnd.next(0, 5) == 0)
                d = rnd.next(dmax / 2, dmax);          // occasional long segment
            else
                d = rnd.next(1, max(2, dmax / 3));      // mostly short segments
            printf(" %d %d", c, d);
        }
        printf("\n");
    }
    return 0;
}
