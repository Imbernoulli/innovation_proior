// TIER: trivial
// Station-chain baseline: link the K ground stations in input order 1-2-...-K.
// Always feasible (every cap>=2); its length equals the checker baseline B, so ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K;
    if (scanf("%d %d", &N, &K) != 2) return 0;
    for (int i = 0; i < N; i++) {
        int x, y, c;
        scanf("%d %d %d", &x, &y, &c);
    }
    printf("%d\n", K - 1);
    for (int i = 1; i < K; i++)
        printf("%d %d\n", i, i + 1);
    return 0;
}
