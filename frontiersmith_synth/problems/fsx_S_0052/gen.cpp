#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Selective pickup-and-delivery single-vehicle routing, skinned as night-shift shunting in a
// railway freight yard (routing-tour family, variant #4).
//
// Distinct from the additive skip-penalty formulation: here there is NO penalty term. Instead
// the participant must complete AT LEAST M of the N precedence-constrained pickup/delivery
// orders (a scheduling quota) while minimizing the closed depot-to-depot Manhattan tour length,
// under a drawbar (capacity) limit Q that allows batching several cuts at once.
//
// Structure ladder: testId 1 is tiny (few orders, ~uniform); the instance grows quadratically
// in the order count and gains more spatial "hub" clustering (rewarding batching and clever
// subset choice) up to a large, tightly-clustered yard by testId 10.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int N = min(400, 4 * testId * testId);   // 4, 16, 36, 64, ..., 400
    if (N < 3) N = 3;
    int Q = 4 + (testId % 3);                // 4,5,6 cycling -> batching room varies
    if (Q < 2) Q = 2;
    if (Q > 8) Q = 8;
    int M = (N + 1) / 2;                      // must complete at least half the orders

    int X0 = 500, Y0 = 500;                   // hump depot near the middle of the yard

    printf("%d %d %d\n", N, Q, M);
    printf("%d %d\n", X0, Y0);

    // Spatial "hub" structure: K track-bundle centres; each order's pickup and delivery are
    // drawn near (possibly different) hubs. More hubs + tighter jitter as testId grows.
    int K = max(1, testId);
    vector<int> cx(K), cy(K);
    for (int k = 0; k < K; k++) {
        cx[k] = (int)rnd.next(60, 940);
        cy[k] = (int)rnd.next(60, 940);
    }
    int jit = max(20, 140 - 10 * testId);    // 130,120,...,40 then floor 20

    auto clampc = [](int v) { return max(0, min(1000, v)); };

    for (int i = 0; i < N; i++) {
        int kp = (int)rnd.next(0, K - 1);
        int kd = (int)rnd.next(0, K - 1);
        int ax = clampc(cx[kp] + (int)rnd.next(-jit, jit));
        int ay = clampc(cy[kp] + (int)rnd.next(-jit, jit));
        int bx = clampc(cx[kd] + (int)rnd.next(-jit, jit));
        int by = clampc(cy[kd] + (int)rnd.next(-jit, jit));
        int q  = (int)rnd.next(1, Q);
        printf("%d %d %d %d %d\n", ax, ay, bx, by, q);
    }
    return 0;
}
