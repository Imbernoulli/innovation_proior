#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Volcano Watch replay instances.  testId 1..10 is a difficulty/structure ladder:
// tiny (T=12,N=4) growing to large (T=5000,N=2000).  Structure varies so the family stays
// interesting: solar (spot) density alternates sparse/dense (contention for cheap steps),
// value profiles alternate uniform / bimodal (a few very valuable tasks vs many cheap ones),
// and window tightness varies.  sum(p_j) is kept above T so the single processor and the
// scarce solar steps genuinely compete -> not all tasks can be cleared cheaply.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ts[10] = {12, 40, 120, 300, 700, 1200, 2000, 3000, 4200, 5000};
    int Ns[10] = {4,  12, 40,  120, 300, 600,  1000, 1400, 1800, 2000};
    int T = Ts[idx - 1];
    int N = Ns[idx - 1];

    // solar availability density: alternate sparse (~30%) and denser (~50%)
    int solarPct = (idx % 2 == 0) ? 50 : 30;
    // value profile: every third test is bimodal (few high-value tasks, many low)
    bool bimodal = (idx % 3 == 0);

    // ---- per-step solar/diesel costs and availability ----
    vector<int> a(T + 1);
    vector<int> cs(T + 1), cd(T + 1);
    for (int t = 1; t <= T; t++) {
        a[t]  = (rnd.next(0, 99) < solarPct) ? 1 : 0;
        cs[t] = rnd.next(1, 3);            // solar is cheap
        cd[t] = rnd.next(10, 20);          // diesel is expensive (>= cs always)
    }

    printf("%d %d\n", T, N);
    for (int t = 1; t <= T; t++)
        printf("%d %d %d\n", a[t], cs[t], cd[t]);

    // ---- tasks ----
    for (int j = 1; j <= N; j++) {
        int p = rnd.next(1, 6);                       // small backlogs -> many tasks contend
        // window length: at least p, with modest slack so scheduling is not trivial
        int slack = rnd.next(0, 2 * p + 5);
        int winLen = min(T, p + slack);
        int r = rnd.next(1, T - winLen + 1);
        int d = r + winLen - 1;

        long long vval;
        if (bimodal) {
            if (rnd.next(0, 99) < 25) vval = (long long)p * rnd.next(12, 20);  // high value
            else                      vval = (long long)p * rnd.next(2, 4);    // low value
        } else {
            vval = (long long)p * rnd.next(4, 14);
        }
        if (vval < 1) vval = 1;

        printf("%d %d %d %lld\n", r, d, p, vval);
    }
    return 0;
}
