#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Flexible job-shop ("reservoir dam network") generator.
// Difficulty ladder: testId 1 tiny (example scale) -> testId 10 larger/denser
// with a heavier bottleneck-duration skew.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int n, m;
    if (t <= 1) {                       // tiny sanity instance
        n = 3; m = 3;
    } else {
        n = min(40, 3 + 2 * t);         // 7 .. 40 orders
        m = min(8, 3 + t / 2);          // 4 .. 8 dams
    }
    if (m < 3) m = 3;

    printf("%d %d\n", n, m);
    for (int j = 0; j < n; j++) {
        int o = rnd.next(2, min(6, m)); // passes per order
        printf("%d", o);
        for (int k = 0; k < o; k++) {
            int cmax = min(3, m);
            int c = rnd.next(1, cmax);  // eligible dams for this pass
            vector<int> dams(m);
            for (int d = 0; d < m; d++) dams[d] = d;
            shuffle(dams.begin(), dams.end());   // testlib deterministic shuffle
            printf(" %d", c);
            for (int e = 0; e < c; e++) {
                int dur;
                // higher testId -> occasional long (bottleneck) durations
                if (t >= 5 && rnd.next(0, 3) == 0)
                    dur = rnd.next(50, 99);
                else
                    dur = rnd.next(1, 33);
                printf(" %d %d", dams[e], dur);
            }
        }
        printf("\n");
    }
    return 0;
}
