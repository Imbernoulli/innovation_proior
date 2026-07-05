#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Flexible job-shop ("festival stage layout") instance generator.
// Difficulty ladder: testId 1 tiny (example scale) -> testId 10 large & adversarial.
//  - low testId: few stages/crews, DENSE eligibility (lots of parallel choices) -> easy.
//  - high testId: more stages/crews, SPARSE eligibility + a "hot" bottleneck crew that
//    appears in many phases + occasional long durations -> contention, harder.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int n, m;
    if (t <= 1) { n = 3; m = 2; }
    else { n = 4 + 3 * t; m = 3 + t; }
    if (n > 40) n = 40;
    if (m > 15) m = 15;

    printf("%d %d\n", n, m);
    for (int j = 0; j < n; j++) {
        int o;
        if (t <= 1) o = 2;
        else o = rnd.next(max(2, m / 2), m);
        printf("%d\n", o);
        for (int k = 0; k < o; k++) {
            // eligibility set size: dense early, sparse (harder) late
            int e;
            if (t <= 1) e = rnd.next(1, m);
            else if (t <= 5) e = rnd.next(max(1, m / 2), m);
            else e = rnd.next(1, max(2, m / 3));
            if (e > m) e = m;

            // pick e distinct crews; at high testId bias crew 0 in as a bottleneck
            vector<int> perm(m);
            for (int c = 0; c < m; c++) perm[c] = c;
            shuffle(perm.begin(), perm.end());
            vector<int> crews(perm.begin(), perm.begin() + e);
            if (t >= 6 && rnd.next(0, 1) == 0) {
                // force the hot crew 0 into the eligibility set
                bool has0 = false;
                for (int c : crews) if (c == 0) has0 = true;
                if (!has0) crews[0] = 0;
            }

            printf("%d", e);
            for (int idx = 0; idx < e; idx++) {
                int c = crews[idx];
                int d;
                if (t >= 6 && rnd.next(0, 3) == 0)
                    d = rnd.next(60, 99);        // occasional long (bottleneck) phase
                else
                    d = rnd.next(1, 40);
                printf(" %d %d", c, d);
            }
            printf("\n");
        }
    }
    return 0;
}
