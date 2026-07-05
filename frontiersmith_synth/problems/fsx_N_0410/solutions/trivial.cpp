// TIER: trivial
// Deploy exactly the single best cluster+aperture (the checker's baseline construction).
// F = B, so ratio = 0.1 on every test.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, NB; long long D;
    if (scanf("%d %d %lld %d", &N, &K, &D, &NB) != 4) return 0;
    vector<long long> R(K);
    for (int t = 0; t < K; t++) scanf("%lld", &R[t]);
    vector<long long> W(N), C(N), A(N);
    for (int i = 0; i < N; i++) {
        long long x, y; int b;
        scanf("%lld %lld %lld %lld %lld %d", &x, &y, &W[i], &C[i], &A[i], &b);
    }
    long long best = LLONG_MIN; int bi = 0, bt = 0;
    for (int i = 0; i < N; i++)
        for (int t = 0; t < K; t++) {
            long long r = R[t];
            long long v = W[i] * r - C[i] * r * r - A[i];
            if (v > best) { best = v; bi = i; bt = t; }
        }
    printf("%d 1\n%d\n", bt + 1, bi + 1);
    return 0;
}
