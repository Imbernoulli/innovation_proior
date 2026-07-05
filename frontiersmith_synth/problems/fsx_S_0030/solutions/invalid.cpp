// TIER: invalid
// Deliberately infeasible: assigns herd 1 to route R+1, which is out of the valid range
// 0..R, so the checker must reject the whole output (score 0).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, R;
    scanf("%d %d", &H, &R);
    for (long long j = 0; j < R; j++) { long long c; scanf("%lld", &c); }
    for (int i = 1; i <= H; i++)
        for (int j = 1; j <= R; j++) { int v, d; scanf("%d %d", &v, &d); }
    printf("%d\n", R + 1);            // out-of-range on purpose
    for (int i = 2; i <= H; i++) printf("0\n");
    return 0;
}
