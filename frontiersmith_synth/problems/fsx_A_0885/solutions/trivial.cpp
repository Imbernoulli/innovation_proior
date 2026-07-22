// TIER: trivial
// Single whole-town territory: id=1 everywhere. Reproduces the checker's own
// baseline construction exactly -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, K;
    scanf("%d %d %d", &R, &C, &K);
    long long total = (long long)R * C * 2;
    for (long long i = 0; i < total; i++) { long long tmp; scanf("%lld", &tmp); }
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) printf("1%c", c + 1 == C ? '\n' : ' ');
    }
    return 0;
}
