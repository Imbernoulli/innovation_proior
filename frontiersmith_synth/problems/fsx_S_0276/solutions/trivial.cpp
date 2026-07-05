// TIER: trivial
// Do-nothing baseline: pause every step, clear no task, pay all penalties.  This is exactly
// the checker's internal baseline B, so it scores the calibration point ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, N;
    if (scanf("%d %d", &T, &N) != 2) return 0;
    for (int t = 0; t < T; t++) { int x, y, z; scanf("%d %d %d", &x, &y, &z); }
    for (int j = 0; j < N; j++) { int a, b, c; long long d; scanf("%d %d %d %lld", &a, &b, &c, &d); }
    for (int t = 0; t < T; t++) printf("0 0\n");
    return 0;
}
