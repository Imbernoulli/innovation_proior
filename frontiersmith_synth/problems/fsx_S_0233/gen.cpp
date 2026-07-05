#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int n, m, relMax;
    if (t == 1) {
        n = 2; m = 2; relMax = 0;
    } else {
        m = min(2 + t, 9);       // 4 .. 9
        n = min(4 + 2 * t, 22);  // 8 .. 22
        relMax = min(40, 4 * t); // 8 .. 40
    }

    // one resource acts as a heavy bottleneck at higher difficulty
    int bott = rnd.next(0, m - 1);

    printf("%d %d\n", n, m);
    for (int j = 0; j < n; j++) {
        int r = (relMax == 0) ? 0 : rnd.next(0, relMax);
        int o;
        if (t == 1) o = 2;
        else o = rnd.next(max(1, m / 2), m); // fairly long commissioning chains

        printf("%d %d", r, o);

        // distinct resources for this telescope's steps (a random permutation prefix)
        vector<int> res(m);
        for (int i = 0; i < m; i++) res[i] = i;
        for (int i = m - 1; i > 0; i--) {
            int k = rnd.next(0, i);
            swap(res[i], res[k]);
        }

        for (int k = 0; k < o; k++) {
            int mac = res[k];
            int dur;
            if (t >= 6 && rnd.next(0, 3) == 0) dur = rnd.next(60, 99); // heavy step
            else dur = rnd.next(1, 50);
            if (mac == bott) dur = min(99, dur + rnd.next(10, 40));    // bottleneck load
            printf(" %d %d", mac, dur);
        }
        printf("\n");
    }
    return 0;
}
