// TIER: trivial
// Input-order chain 1-2-...-N.  Exactly the checker's baseline construction, so it scores
// the calibration point ratio ~= 0.1.  Always feasible because every cap_i >= 2.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) {
        int x, y, c, w;
        scanf("%d %d %d %d", &x, &y, &c, &w);
    }
    printf("%d\n", N - 1);
    for (int i = 1; i < N; i++) printf("%d %d\n", i, i + 1);
    return 0;
}
