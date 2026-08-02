// TIER: invalid
// Deliberately infeasible: "accept" every single bid regardless of shared
// items. Every hub/path/cycle gadget has bids that share a capacity-1 item,
// so this always oversubscribes at least one item and must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int M, N;
    scanf("%d %d", &M, &N);
    for (int j = 0; j < M; j++) { long long x; scanf("%lld", &x); }
    for (int i = 0; i < N; i++) {
        int k; long long p;
        scanf("%d %lld", &k, &p);
        for (int t = 0; t < k; t++) { int x; scanf("%d", &x); }
    }
    printf("%d\n", N);
    for (int i = 1; i <= N; i++) printf("%d%c", i, i == N ? '\n' : ' ');
    if (N == 0) printf("\n");
    return 0;
}
