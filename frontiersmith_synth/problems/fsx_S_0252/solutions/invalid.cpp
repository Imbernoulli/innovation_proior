// TIER: invalid
// Deliberately infeasible: every step demands N*Q total production, which exceeds the plant
// throughput cap Q (since N >= 1 and we send Q to each room) -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, T; ll Q, S, P;
    scanf("%d %d %lld %lld %lld", &N, &T, &Q, &S, &P);
    vector<int> solar(T); for (auto& z : solar) scanf("%d", &z);
    vector<ll> pr(T); for (auto& z : pr) scanf("%lld", &z);
    vector<ll> Cap(N), I0(N);
    vector<vector<ll>> L(N, vector<ll>(T));
    for (int i = 0; i < N; i++) {
        scanf("%lld %lld", &Cap[i], &I0[i]);
        for (int t = 0; t < T; t++) { ll tmp; scanf("%lld", &tmp); L[i][t] = tmp; }
    }
    for (int t = 0; t < T; t++) {
        printf("2");
        for (int i = 0; i < N; i++) printf(" %lld", Q);   // sum = N*Q > Q
        printf("\n");
    }
    return 0;
}
