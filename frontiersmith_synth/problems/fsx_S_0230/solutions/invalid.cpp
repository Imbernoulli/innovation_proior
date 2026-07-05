// TIER: invalid
// Deliberately infeasible: names a station index out of range (N+5), which the
// checker must reject -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M; long long R;
    scanf("%d %d %lld", &N, &M, &R);
    vector<long long> cost(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &cost[i]);
    printf("1\n%d\n", N + 5); // out-of-range station index
    return 0;
}
