// TIER: trivial
// Do-nothing circuit: serve nobody, drive nowhere. Scores exactly the baseline (0.1).
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N; long long SX, SY;
    if (scanf("%d %lld %lld", &N, &SX, &SY) != 3) return 0;
    for (int i = 0; i < N; i++) {
        long long a, b, c, d, w;
        scanf("%lld %lld %lld %lld %lld", &a, &b, &c, &d, &w);
    }
    printf("0\n\n");
    return 0;
}
