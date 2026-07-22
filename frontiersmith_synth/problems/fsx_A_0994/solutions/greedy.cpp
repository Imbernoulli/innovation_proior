// TIER: greedy
// The "obvious" approach: quality strictly decays with radius, so just take
// the M highest-quality sites -- the geometric sweet spot right around the
// tower. This is a one-pass sort/select with no notion of interference at
// all; it walks straight into the dense hot zone where mutual shading and
// mutual blocking are worst.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int P, M, K;
    if (scanf("%d %d %d", &P, &M, &K) != 3) return 0;
    vector<ll> X(P + 1), Y(P + 1), Q(P + 1);
    for (int i = 1; i <= P; i++) scanf("%lld %lld %lld", &X[i], &Y[i], &Q[i]);
    for (int k = 0; k < K; k++) { ll dx, dy, e; scanf("%lld %lld %lld", &dx, &dy, &e); }

    vector<int> ord(P);
    iota(ord.begin(), ord.end(), 1);
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (Q[a] != Q[b]) return Q[a] > Q[b];
        return a < b;
    });

    int m = min(M, P);
    printf("%d\n", m);
    for (int i = 0; i < m; i++) printf("%d ", ord[i]);
    printf("\n");
    return 0;
}
