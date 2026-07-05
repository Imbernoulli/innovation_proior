// TIER: invalid
// Deliberately infeasible: emits an out-of-range pool index (P+1) for school 1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, P;
    scanf("%d %d", &N, &P);
    vector<long long> C(P);
    for (int j = 0; j < P; j++) scanf("%lld", &C[j]);
    long long vv, ww;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++) scanf("%lld %lld", &vv, &ww);

    // school 1 guided into a non-existent pool -> feasibility violation -> score 0
    printf("%d", P + 1);
    for (int i = 1; i < N; i++) printf(" 0");
    printf("\n");
    return 0;
}
