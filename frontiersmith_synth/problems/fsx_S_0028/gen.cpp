#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // difficulty / size ladder
    int Ps[11] = {0, 4, 15, 40, 70, 110, 160, 220, 290, 350, 400};
    int N = Ps[t];
    const int C = 10000;

    // capacity alternates tight / loose across the ladder
    int Q = (t % 2 == 0) ? 8 : 12;

    // hub near the centre of the plain
    int x0 = C / 2, y0 = C / 2;

    printf("%d %d\n", N, Q);
    printf("%d %d\n", x0, y0);

    // structural mode: 0 uniform, 1 clustered caches+stations, 2 clustered
    // caches with uniform stations (favours batching very differently)
    int mode = t % 3;

    int nc = 6;
    vector<pair<int,int>> cent;
    for (int i = 0; i < nc; i++)
        cent.push_back({rnd.next(0, C), rnd.next(0, C)});

    for (int j = 0; j < N; j++) {
        int px, py, dx, dy;
        if (mode == 0) {
            px = rnd.next(0, C); py = rnd.next(0, C);
            dx = rnd.next(0, C); dy = rnd.next(0, C);
        } else if (mode == 1) {
            auto c = cent[rnd.next(0, nc - 1)];
            px = min(C, max(0, c.first  + (int)rnd.next(-800, 800)));
            py = min(C, max(0, c.second + (int)rnd.next(-800, 800)));
            auto d = cent[rnd.next(0, nc - 1)];
            dx = min(C, max(0, d.first  + (int)rnd.next(-800, 800)));
            dy = min(C, max(0, d.second + (int)rnd.next(-800, 800)));
        } else {
            auto c = cent[rnd.next(0, nc - 1)];
            px = min(C, max(0, c.first  + (int)rnd.next(-500, 500)));
            py = min(C, max(0, c.second + (int)rnd.next(-500, 500)));
            dx = rnd.next(0, C); dy = rnd.next(0, C);
        }
        int q = rnd.next(1, min(5, Q));
        printf("%d %d %d %d %d\n", px, py, dx, dy, q);
    }
    return 0;
}
