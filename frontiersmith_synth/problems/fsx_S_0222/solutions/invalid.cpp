// TIER: invalid
// Deliberately infeasible: routes every zone through a non-existent gate index P+1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, P;
    if (scanf("%d %d", &N, &P) != 2) return 0;
    vector<long long> C(P), K(P);
    for (int j = 0; j < P; j++) scanf("%lld %lld", &C[j], &K[j]);
    long long val, vol;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++)
            scanf("%lld %lld", &val, &vol);
    for (int i = 0; i < N; i++) printf("%d%c", P + 1, i + 1 == N ? '\n' : ' ');
    return 0;
}
