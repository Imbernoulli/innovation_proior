// TIER: trivial
// All-grid, refill exactly the load each step = the scoring baseline -> ratio ~= 0.1.
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
        for (int t = 0; t < T; t++) scanf("%lld", &L[i][t]);
    }
    for (int t = 0; t < T; t++) {
        printf("2");
        for (int i = 0; i < N; i++) printf(" %lld", L[i][t]);
        printf("\n");
    }
    return 0;
}
