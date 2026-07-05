// TIER: invalid
// Deliberately infeasible: delivers order 1 without ever picking it up (precedence
// violation), so the checker must reject it with score 0.
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
    printf("1\n");
    printf("1 1\n");   // deliver order 1 with no matching pickup
    return 0;
}
