// TIER: invalid
// Deliberately infeasible: reports an out-of-range channel index on every
// link (C+5, well outside [-1,C]), which the checker's bounded readInt must
// reject with WA (score 0).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, C, K;
    scanf("%d %d %d", &N, &C, &K);
    long long alpha; double theta; long long N0;
    scanf("%lld %lf %lld", &alpha, &theta, &N0);
    for (int j = 0; j < K; j++) { long long p; scanf("%lld", &p); }
    for (int i = 0; i < N; i++) {
        long long a, b, c, d, e;
        scanf("%lld %lld %lld %lld %lld", &a, &b, &c, &d, &e);
        printf("%d %d\n", C + 5, 1);
    }
    return 0;
}
