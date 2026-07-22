// TIER: invalid
// Deliberately infeasible: prints a coordinate far outside [0,L] so the
// checker's bounded reads reject it -- must score 0.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N, K, M, L, r, R;
    if (scanf("%d %d %d %d %d %d", &N, &K, &M, &L, &r, &R) != 6) return 0;
    printf("%d %d\n", L + 999999, 0);
    for (int k = 1; k < K; k++) printf("0 0\n");
    return 0;
}
