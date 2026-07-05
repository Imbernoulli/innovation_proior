// TIER: invalid
// Deliberately infeasible: prints an out-of-range mode (3) for every cell.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T; long long warm;
    scanf("%d %d %lld", &N, &T, &warm);
    vector<long long> od(T + 1), sp(T + 1);
    vector<int> C(T + 1);
    for (int t = 1; t <= T; t++) scanf("%lld", &od[t]);
    for (int t = 1; t <= T; t++) scanf("%lld", &sp[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &C[t]);
    vector<int> R(N + 1), d(N + 1);
    for (int j = 1; j <= N; j++) scanf("%d %d", &R[j], &d[j]);

    for (int j = 1; j <= N; j++)
        for (int t = 1; t <= T; t++)
            printf("%d%c", 3, t == T ? '\n' : ' ');
    return 0;
}
