// TIER: trivial
// Grant only the single highest-value pitch -- exactly the checker's baseline B.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N; long long M;
    if (scanf("%d %lld", &N, &M) != 2) return 0;
    int best = 1; long long bestw = -1;
    for (int i = 1; i <= N; i++) {
        long long w; scanf("%lld", &w);
        if (w > bestw) { bestw = w; best = i; }
    }
    // consume edges (not needed)
    for (long long i = 0; i < M; i++) { int a, b; scanf("%d %d", &a, &b); }
    printf("1\n%d\n", best);
    return 0;
}
