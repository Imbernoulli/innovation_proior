// TIER: invalid
// Deliberately infeasible: assign EVERY rack to unit 1, which overloads its capacity
// (total load of all racks on one unit far exceeds C_1).  Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<long long> v(N);
    for (int i = 0; i < N; i++) scanf("%lld", &v[i]);
    vector<long long> C(M);
    for (int j = 0; j < M; j++) scanf("%lld", &C[j]);
    vector<vector<long long>> d(N, vector<long long>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) scanf("%lld", &d[i][j]);

    for (int i = 0; i < N; i++) printf("1%c", i + 1 < N ? ' ' : '\n');
    return 0;
}
