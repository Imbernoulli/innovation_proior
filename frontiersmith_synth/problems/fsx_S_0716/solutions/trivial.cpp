// TIER: trivial
// Input-order pass-through: expose fields in exactly the order they were listed.  This is
// exactly the checker's baseline construction, so it scores the calibration point
// ratio ~= 0.1.  Always feasible (it is a permutation of 1..N by construction).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    double D, alpha, Q0;
    scanf("%lf %lf %lf", &D, &alpha, &Q0);
    for (int i = 0; i < N; i++) {
        int x, y, w;
        scanf("%d %d %d", &x, &y, &w);
    }
    for (int i = 1; i <= N; i++) printf("%d ", i);
    printf("\n");
    return 0;
}
