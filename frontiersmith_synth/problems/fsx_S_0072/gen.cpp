#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// online-to-offline-replay family, harbour container port theme, variant #5.
// Static full-trace spot/shore/pause crane-power scheduling with a shared spot
// capacity and per-run gantry fixed charge -> fixed-charge scheduling.
//
// testId is a structure/difficulty ladder:
//   testId 1  -> tiny (example scale)
//   testId 10 -> large horizon + many ships, scarce spot, heavy-tailed cheap spot.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int T = 6 + 18 * (testId - 1);   // 6, 24, ..., 168
    int J = 3 + 5 * (testId - 1);    // 3, 8, ..., 48
    if (T < 2) T = 2;
    if (J < 1) J = 1;

    int G = rnd.next(4, 8) + 3 * (testId % 4);   // 4..19 gantry re-mobilisation charge

    // spot scarcity tightens as the ladder grows.
    int capHi = max(1, (J / 2) - (testId % 3));

    // heavy-tailed cheap-spot fraction varies per test.
    double cheapFrac = 0.15 + 0.03 * (testId % 4);

    // hourly power trace
    vector<int> sp(T), od(T), C(T);
    for (int t = 0; t < T; t++) {
        int o = rnd.next(8, 40);
        int s;
        if (rnd.next(0.0, 1.0) < cheapFrac)
            s = rnd.next(1, min(o, 3));           // occasional very cheap spot hour
        else
            s = rnd.next(1, o);                    // sp <= od always
        int c = rnd.next(0, capHi);
        sp[t] = s; od[t] = o; C[t] = c;
    }

    // ships: berth windows and crane-hour demands
    vector<array<int,3>> ships; // a, b, w
    for (int j = 0; j < J; j++) {
        int a = rnd.next(0, T - 2);
        int maxlen = T - a;                        // >= 2
        // bias toward moderate windows with slack so pause/spot chasing matters
        int len = rnd.next(2, maxlen);
        int b = a + len;
        int w = rnd.next(1, len);
        ships.push_back({a, b, w});
    }

    printf("%d %d %d\n", T, J, G);
    for (int t = 0; t < T; t++)
        printf("%d %d %d\n", sp[t], od[t], C[t]);
    for (auto& s : ships)
        printf("%d %d %d\n", s[0], s[1], s[2]);
    return 0;
}
