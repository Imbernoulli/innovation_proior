// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Deliberately infeasible: for every cabinet, claims a single probe with a
// stride far larger than D_c (out of range, and does not sum to D_c). The
// checker's bounded read must reject this immediately -> score 0.
int main() {
    int K; ll M;
    scanf("%d %lld", &K, &M);
    for (int c = 0; c < K; c++) {
        int D; ll W;
        scanf("%d %lld", &D, &W);
        for (int d = 0; d <= D; d++) { ll t; scanf("%lld", &t); }
        printf("1 %d\n", D * 2 + 7);
    }
    return 0;
}
