// TIER: invalid
// Deliberately infeasible: declares period 0 (out of [1,Q]) -> checker WA -> 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int B, h, P, Q;
    if (scanf("%d %d %d %d", &B, &h, &P, &Q) != 4) return 0;
    for (int i = 0; i < P * P; i++) { int x; if (scanf("%d", &x) != 1) return 0; }
    for (int b = 0; b < B; b++) {
        printf("0\n");           // invalid period
        for (int r = 0; r < h; r++) printf("0\n");
    }
    return 0;
}
