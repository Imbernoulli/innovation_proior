// TIER: invalid
// Deliberately infeasible: places a module far outside the parcel -> checker rejects -> 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int H, W, T;
    if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    for (int t = 1; t <= T; t++) {
        int k, c; scanf("%d %d", &k, &c);
        for (int i = 0; i < k; i++) { int r, cc; scanf("%d %d", &r, &cc); }
    }
    printf("1\n1 0 %d 0\n", H + 50); // out-of-bounds anchor
    return 0;
}
