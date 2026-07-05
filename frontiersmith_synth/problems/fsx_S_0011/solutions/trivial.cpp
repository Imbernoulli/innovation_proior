// TIER: trivial
// Activate only the single highest-value station -- exactly the checker's baseline B.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    vector<long long> w(N + 1);
    int best = 1;
    for (int j = 1; j <= N; j++) {
        scanf("%lld", &w[j]);
        if (w[j] > w[best]) best = j;
    }
    // consume edges (not needed)
    for (int i = 0; i < M; i++) {
        int u, v; scanf("%d %d", &u, &v);
        (void)u; (void)v;
    }
    printf("1\n%d\n", best);
    return 0;
}
