// TIER: trivial
// Do-nothing baseline: emit the given (input) order 1..N. This is exactly the
// reference order the checker measures, so it scores the calibration point ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, C, E;
    if (scanf("%d %d %d %d", &N, &M, &C, &E) != 4) return 0;
    for (int i = 1; i <= N; i++) printf("%d%c", i, i == N ? '\n' : ' ');
    return 0;
}
