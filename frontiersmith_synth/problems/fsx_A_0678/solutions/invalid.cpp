// TIER: invalid
// Deliberately infeasible: prints an out-of-range phase (>= 3600) for the
// first element so the checker's bounded read must reject it -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N, M, D1000;
    scanf("%d %d %d", &N, &M, &D1000);
    for (int i = 0; i < N; i++){ int x; scanf("%d", &x); }
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); }
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); }
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); }
    int K; scanf("%d", &K);
    for (int i = 0; i < K; i++){ int x; scanf("%d", &x); }
    int thresh; scanf("%d", &thresh);

    printf("9999");
    for (int i = 1; i < N; i++) printf(" %d", 0);
    printf("\n");
    return 0;
}
