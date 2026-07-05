#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Difficulty ladder for a "medium"-scale job-shop weighted-completion instance.
// testId 1 : tiny (example scale). Growing to n=24 rides / m=12 crews by testId 10,
// with duration and weight skew (a few bottleneck tasks + a few blockbuster rides)
// introduced at higher testIds so simple index-order serial scheduling is bad.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int n, m, dmax, wmax;
    if (t <= 1) {                 // tiny sanity instance
        n = 3; m = 3; dmax = 20; wmax = 5;
    } else {
        n = 4 + 2 * t;            // 8 .. 24 rides
        m = 2 + t;                // 4 .. 12 crews
        dmax = 99; wmax = 20;
    }
    if (n > 30) n = 30;
    if (m > 15) m = 15;

    printf("%d %d\n", n, m);
    for (int j = 0; j < n; j++) {
        // number of tasks: roughly square (visit most crews), at least 1.
        int o;
        if (t <= 1) o = m;
        else o = rnd.next(max(1, m - 2), m);

        // choose an ordered set of distinct crews for the tasks.
        vector<int> crew(m);
        for (int k = 0; k < m; k++) crew[k] = k;
        shuffle(crew.begin(), crew.end());   // deterministic testlib shuffle
        crew.resize(o);

        // weight: mostly modest, but at high testId a few blockbuster rides.
        int w;
        if (t >= 6 && rnd.next(0, 5) == 0)
            w = rnd.next(wmax / 2, wmax);        // blockbuster
        else
            w = rnd.next(1, max(1, wmax / 4));

        printf("%d %d", w, o);
        for (int k = 0; k < o; k++) {
            int d;
            if (t >= 6 && rnd.next(0, 4) == 0)
                d = rnd.next(dmax / 2, dmax);    // occasional long bottleneck task
            else
                d = rnd.next(1, max(2, dmax / 3));
            printf(" %d %d", crew[k], d);
        }
        printf("\n");
    }
    return 0;
}
