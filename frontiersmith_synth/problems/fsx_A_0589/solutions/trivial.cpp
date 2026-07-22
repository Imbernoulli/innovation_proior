// TIER: trivial
// Monochrome facade: every band period 1, color 0. This is exactly the checker's
// baseline construction -> ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int B, h, P, Q;
    if (scanf("%d %d %d %d", &B, &h, &P, &Q) != 4) return 0;
    for (int i = 0; i < P * P; i++) { int x; if (scanf("%d", &x) != 1) return 0; }
    for (int b = 0; b < B; b++) {
        printf("1\n");
        for (int r = 0; r < h; r++) printf("0\n");
    }
    return 0;
}
