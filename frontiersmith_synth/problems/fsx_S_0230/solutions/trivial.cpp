// TIER: trivial
// Build-everywhere baseline: install a station in every zone. Always feasible,
// scores exactly the calibration ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M; long long R;
    scanf("%d %d %lld", &N, &M, &R);
    vector<long long> cost(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &cost[i]);
    // (edges irrelevant for the trivial plan)
    printf("%d\n", N);
    for (int i = 1; i <= N; i++)
        printf("%d%c", i, i == N ? '\n' : ' ');
    return 0;
}
