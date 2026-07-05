#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Difficulty ladder for a "large"-scale prize-collecting precedence-constrained
// pickup-delivery tour (pandemic sample-relay courier).
//
// testId 1 : tiny (example scale, n=3).
// growing to n=250 relays (500 sites) by testId 10.
// At higher testIds we inject "remote" relays: both sites clustered near a far corner
// with a LOW skip penalty, so a good solver is rewarded for CHOOSING to skip them,
// while the mandatory/near relays keep the ordering (TSP) part hard.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    const int C = 1000;
    int n;
    if (t <= 1) n = 3;
    else n = 25 * t;            // 50 .. 250
    if (n > 250) n = 250;

    int dx = rnd.next(0, C), dy = rnd.next(0, C);
    printf("%d\n", n);
    printf("%d %d\n", dx, dy);

    // four far corners for remote clusters
    int corners[4][2] = {{0,0},{0,C},{C,0},{C,C}};

    for (int i = 1; i <= n; i++) {
        int px, py, qx, qy, P;
        bool remote = (t >= 5 && rnd.next(0, 3) == 0);
        if (remote) {
            int c = rnd.next(0, 3);
            int bx = corners[c][0], by = corners[c][1];
            auto clamp = [&](int v){ return max(0, min(C, v)); };
            px = clamp(bx + rnd.next(-120, 120));
            py = clamp(by + rnd.next(-120, 120));
            qx = clamp(bx + rnd.next(-120, 120));
            qy = clamp(by + rnd.next(-120, 120));
            P  = rnd.next(50, 500);           // cheap to skip => choosing matters
        } else {
            px = rnd.next(0, C); py = rnd.next(0, C);
            qx = rnd.next(0, C); qy = rnd.next(0, C);
            P  = rnd.next(400, 3000);         // usually worth serving
        }
        printf("%d %d %d %d %d\n", px, py, qx, qy, P);
    }
    return 0;
}
