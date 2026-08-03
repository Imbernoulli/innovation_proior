#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // difficulty ladder: tiny at testId 1, growing grid otherwise.
    int R, C;
    if (testId == 1) { R = 2; C = 3; }
    else { R = 4 + testId; C = 4 + testId; } // up to 14x14 = 196 nodes at testId 10

    int n = R * C;
    auto id = [&](int r, int c) { return r * C + c + 1; };

    struct E { int u, v; long long w, c; bool hw; };
    vector<E> es;
    // grid edges. The "plowed highway" is row 0 (horizontal) + last column (vertical):
    // an L-shaped weight-1 corridor from s=(0,0) to t=(R-1,C-1). All other roads are
    // slower (weight 3..9), so the highway is the unique fastest route; closing highway
    // roads forces expensive detours.
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            if (c + 1 < C) {
                bool hw = (r == 0);
                long long w = hw ? 1 : rnd.next(3, 9);
                long long cost = rnd.next(1, 4);
                es.push_back({id(r, c), id(r, c + 1), w, cost, hw});
            }
            if (r + 1 < R) {
                bool hw = (c == C - 1);
                long long w = hw ? 1 : rnd.next(3, 9);
                long long cost = rnd.next(1, 4);
                es.push_back({id(r, c), id(r + 1, c), w, cost, hw});
            }
        }
    }
    int m = (int)es.size();

    // Budget: sum of the q cheapest highway closure efforts, so at least q cheap cuts are
    // affordable (feasible) while the budget is a real binding constraint (< total).
    vector<long long> hwcost;
    for (auto& e : es) if (e.hw) hwcost.push_back(e.c);
    sort(hwcost.begin(), hwcost.end());
    int q = max(2, (R + C - 2) / 4);
    q = min(q, (int)hwcost.size());
    long long B = 0;
    for (int i = 0; i < q; i++) B += hwcost[i];
    if (B < 1) B = 1;

    int s = id(0, 0), t = id(R - 1, C - 1);
    printf("%d %d %d %d %lld\n", n, m, s, t, B);
    for (auto& e : es) printf("%d %d %lld %lld\n", e.u, e.v, e.w, e.c);
    return 0;
}
