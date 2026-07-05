// TIER: invalid
// Deliberately infeasible: build no lamps. Every vertex has demand >= 1, so all
// demands are unmet -> checker must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, R;
    if (scanf("%d %d %d", &N, &M, &R) != 3) return 0;
    vector<int> c(N + 1);
    for (int i = 1; i <= N; i++) scanf("%d", &c[i]);
    vector<int> d(N + 1);
    for (int i = 1; i <= N; i++) scanf("%d", &d[i]);
    for (int e = 0; e < M; e++) { int u, v; scanf("%d %d", &u, &v); }
    printf("0\n\n");
    return 0;
}
