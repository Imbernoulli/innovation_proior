#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10) - 1;
    int Ns[] = {4, 10, 20, 40, 60, 90, 130, 170, 210, 250};
    int N = Ns[idx];
    int SX = 5000, SY = 5000;
    printf("%d %d %d\n", N, SX, SY);

    // Odd test ids: clustered geography (ordering matters more).
    // Even test ids: uniform scatter.
    bool clustered = (testId % 2 == 1);
    int nc = 3 + (testId % 4);          // number of clusters
    vector<int> cx(nc), cy(nc);
    for (int c = 0; c < nc; c++) { cx[c] = rnd.next(1200, 8800); cy[c] = rnd.next(1200, 8800); }

    auto clampc = [](int v) { return min(10000, max(0, v)); };

    for (int i = 0; i < N; i++) {
        int px, py, dx, dy;
        if (clustered) {
            int c1 = rnd.next(0, nc - 1), c2 = rnd.next(0, nc - 1);
            px = clampc(cx[c1] + rnd.next(-900, 900));
            py = clampc(cy[c1] + rnd.next(-900, 900));
            dx = clampc(cx[c2] + rnd.next(-900, 900));
            dy = clampc(cy[c2] + rnd.next(-900, 900));
        } else {
            px = rnd.next(0, 10000); py = rnd.next(0, 10000);
            dx = rnd.next(0, 10000); dy = rnd.next(0, 10000);
        }
        // Penalty magnitude varies by test to keep the serve/skip trade-off live.
        // Penalties sit above typical driving costs so serving is usually (not always)
        // worthwhile, and clever routing beats crude chaining.
        int wlo = 6000 + 1000 * (testId % 3);
        int whi = 26000 + 3000 * (testId % 4);
        int w = rnd.next(wlo, whi);
        printf("%d %d %d %d %d\n", px, py, dx, dy, w);
    }
    return 0;
}
