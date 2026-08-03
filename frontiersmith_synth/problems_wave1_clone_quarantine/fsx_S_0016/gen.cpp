#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int tid = atoi(argv[1]);

    int sizes[10] = {2, 5, 8, 12, 18, 25, 35, 45, 58, 70};
    int idx = tid - 1;
    if (idx < 0) idx = 0;
    if (idx > 9) idx = 9;
    int P = sizes[idx];

    // Capacity: tight cage on even tests, roomy cage on odd tests.
    int Q = (tid % 2 == 0) ? rnd.next(5, 10) : rnd.next(20, 40);

    int x0 = rnd.next(0, 1000), y0 = rnd.next(0, 1000);

    // Geometry mode: clustered rooftop districts on every third test.
    bool clustered = (tid % 3 == 0);

    // Penalty scale (large relative to flight so serving is genuinely profitable).
    int wlo = 300, whi = 1800;
    if (tid % 4 == 1) { wlo = 900; whi = 3000; }   // high penalties -> serve a lot
    if (tid % 4 == 2) { wlo = 150; whi = 700;  }   // low penalties  -> serve selectively

    // Freshness sensitivity.
    int flo = 1, fhi = 8;
    if (tid % 5 == 0) { flo = 5; fhi = 15; }        // wilting-sensitive shift

    printf("%d %d\n", P, Q);
    printf("%d %d\n", x0, y0);

    int nc = clustered ? rnd.next(2, 4) : 0;
    vector<pair<int,int>> centers;
    for (int c = 0; c < nc; c++)
        centers.push_back({rnd.next(0, 1000), rnd.next(0, 1000)});

    for (int i = 0; i < P; i++) {
        int px, py, dx, dy;
        if (clustered) {
            auto c1 = centers[rnd.next(0, nc - 1)];
            auto c2 = centers[rnd.next(0, nc - 1)];
            px = min(1000, max(0, c1.first  + rnd.next(-120, 120)));
            py = min(1000, max(0, c1.second + rnd.next(-120, 120)));
            dx = min(1000, max(0, c2.first  + rnd.next(-120, 120)));
            dy = min(1000, max(0, c2.second + rnd.next(-120, 120)));
        } else {
            px = rnd.next(0, 1000); py = rnd.next(0, 1000);
            dx = rnd.next(0, 1000); dy = rnd.next(0, 1000);
        }
        int q = rnd.next(1, min(5, Q));
        int w = rnd.next(wlo, whi);
        int f = rnd.next(flo, fhi);
        printf("%d %d %d %d %d %d %d\n", px, py, dx, dy, q, w, f);
    }
    return 0;
}
