// TIER: trivial
// Build a lamp on every intersection -> reproduces the checker baseline B -> ratio 0.1.
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
    printf("%d\n", N);
    for (int i = 1; i <= N; i++) printf("%d%c", i, i == N ? '\n' : ' ');
    return 0;
}
