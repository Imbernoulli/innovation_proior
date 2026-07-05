// TIER: trivial
// Dig the whole backbone (trenches 1..N-1). Exactly the checker baseline -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    // We do not even need the edge data; just skip it.
    for (int k = 0; k < M; k++) { int a, b, c; scanf("%d %d %d", &a, &b, &c); (void)a;(void)b;(void)c; }
    for (int i = 0; i < K; i++) { int h; scanf("%d", &h); (void)h; }
    printf("%d\n", N - 1);
    for (int k = 1; k <= N - 1; k++) printf("%d%c", k, k == N - 1 ? '\n' : ' ');
    return 0;
}
