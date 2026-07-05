// TIER: trivial
// Build a tower on EVERY cell. Always feasible; reproduces the checker baseline B exactly,
// so it scores ratio = 0.1 by construction.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, R;
    scanf("%d %d %d", &N, &M, &R);
    vector<int> c(N + 1);
    for (int v = 1; v <= N; v++) scanf("%d", &c[v]);
    // edges are irrelevant to this baseline
    for (int i = 0; i < M; i++) { int a, b; scanf("%d %d", &a, &b); }
    printf("%d\n", N);
    for (int v = 1; v <= N; v++) printf("%d%c", v, v == N ? '\n' : ' ');
    return 0;
}
