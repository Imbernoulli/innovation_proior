// TIER: trivial
// Serve every order, one cut at a time, in input order -- exactly reproduces the checker
// baseline B, so it scores the calibration ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, Q, M;
    scanf("%d %d %d", &N, &Q, &M);
    int X0, Y0; scanf("%d %d", &X0, &Y0);
    for (int i = 0; i < N; i++) {
        int ax, ay, bx, by, q;
        scanf("%d %d %d %d %d", &ax, &ay, &bx, &by, &q);
    }
    printf("%d\n", 2 * N);
    for (int i = 1; i <= N; i++) {
        printf("0 %d\n", i);
        printf("1 %d\n", i);
    }
    return 0;
}
