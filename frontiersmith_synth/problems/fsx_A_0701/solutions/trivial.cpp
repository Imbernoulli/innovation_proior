// TIER: trivial
// As few classes as the n<=1000 cap allows (k=1 whenever N<=1000), no base edges at all.
// With E=0 no real edge can ever be reached (a matching slot would require two distinct
// vertices sharing a (class,sheet) pair, which feasibility forbids), so F=8=B always.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    for (int i = 0; i < M; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
    }
    int k = max(1, (N + 999) / 1000); // keep n=ceil(N/k) <= 1000
    int n = (N + k - 1) / k;
    printf("%d %d\n", k, n);
    for (int t = 0; t < N; t++) {
        printf("%d %d\n", t / n, t % n);
    }
    printf("%d\n", 0);
    return 0;
}
