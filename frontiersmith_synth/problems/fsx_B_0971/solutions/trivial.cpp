// TIER: trivial
// Leaves every rotor at its textbook default phase 0. This reproduces the
// checker's own baseline construction exactly (uniform-rotor rotor-router
// aggregation from a single source), which is known to round into a disk --
// scores ratio == 0.1 on every case by construction.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int L, N, sx, sy;
    scanf("%d %d", &L, &N);
    scanf("%d %d", &sx, &sy);
    for (int i = 0; i < N; i++) { int x, y; scanf("%d %d", &x, &y); }
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) printf("0%c", x + 1 < L ? ' ' : '\n');
    }
    return 0;
}
