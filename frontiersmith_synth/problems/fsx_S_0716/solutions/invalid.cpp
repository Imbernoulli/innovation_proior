// TIER: invalid
// Deliberately infeasible for every N >= 2: prints field 1 twice and never prints field 2,
// so the checker sees a repeated value in the exposure order.  Must score 0.
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
    printf("1 ");
    for (int i = 3; i <= N; i++) printf("%d ", i);
    printf("1\n");
    return 0;
}
