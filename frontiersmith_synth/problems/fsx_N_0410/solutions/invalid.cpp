// TIER: invalid
// Deliberately infeasible: deploy ALL clusters at the LARGEST aperture, where the swept
// disks overlap massively -> the checker's collision test fires -> score 0. This exercises
// the geometric feasibility validation.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, NB; long long D;
    if (scanf("%d %d %lld %d", &N, &K, &D, &NB) != 4) return 0;
    vector<long long> R(K);
    for (int t = 0; t < K; t++) scanf("%lld", &R[t]);
    for (int i = 0; i < N; i++) {
        long long x, y, w, c, a; int b;
        scanf("%lld %lld %lld %lld %lld %d", &x, &y, &w, &c, &a, &b);
    }
    printf("%d %d\n", K, N);                 // largest aperture, deploy everything
    for (int i = 0; i < N; i++) printf("%d%c", i + 1, i + 1 < N ? ' ' : '\n');
    return 0;
}
