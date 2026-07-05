// TIER: invalid
// Deliberately infeasible: assigns target 1 to a non-existent telescope T+1
// (out of the allowed range 0..T), so the checker must reject it -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T;
    scanf("%d %d", &N, &T);
    vector<long long> C(T);
    for (int j = 0; j < T; j++) scanf("%lld", &C[j]);
    long long tmp;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 2 * T; j++) scanf("%lld", &tmp);

    printf("%d\n", T + 1); // out-of-range telescope index
    for (int i = 1; i < N; i++) printf("0\n");
    return 0;
}
