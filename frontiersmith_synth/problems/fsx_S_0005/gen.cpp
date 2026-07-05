#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Difficulty ladder: testId 1 tiny (example scale) -> testId 10 larger/denser.
// A "small"-scale square-ish job-shop instance: n missions, m teams, each mission
// visits every team once in a random order with a random leg duration.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int n, m, dmax;
    if (t <= 1) {           // tiny sanity instance
        n = 3; m = 3; dmax = 20;
    } else {
        n = 3 + 2 * t;      // 7 .. 23 missions
        m = 2 + t;          // 4 .. 12 teams
        dmax = 99;
    }
    if (n > 40) n = 40;
    if (m > 20) m = 20;

    printf("%d %d\n", n, m);
    for (int j = 0; j < n; j++) {
        vector<int> team(m);
        for (int k = 0; k < m; k++) team[k] = k;
        shuffle(team.begin(), team.end());   // testlib deterministic shuffle
        // higher testId -> more duration skew (a few long legs among short ones)
        printf("%d", m);
        for (int k = 0; k < m; k++) {
            int d;
            if (t >= 6 && rnd.next(0, 4) == 0)
                d = rnd.next(dmax / 2, dmax);    // occasional long leg (bottleneck)
            else
                d = rnd.next(1, max(2, dmax / 3));
            printf(" %d %d", team[k], d);
        }
        printf("\n");
    }
    return 0;
}
