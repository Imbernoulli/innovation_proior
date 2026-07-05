// TIER: invalid
// Deliberately INFEASIBLE: output a spanning PATH 1-2-...-N (no closing tunnel). It is
// connected and every module is within its cap, but every tunnel is a bridge, so the
// network is NOT 2-edge-connected (not survivable) and must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N; if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) { long long x, y, c; scanf("%lld %lld %lld", &x, &y, &c); }
    printf("%d\n", N - 1);
    for (int i = 1; i < N; i++) printf("%d %d\n", i, i + 1);
    return 0;
}
