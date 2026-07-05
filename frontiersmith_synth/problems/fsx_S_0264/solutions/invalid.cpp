// TIER: invalid
// Deliberately infeasible: water each bed at its release step (valid assignment) but source
// ZERO water at every step. Since total demand is positive, some step has avail < demand,
// so the checker rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int T, N; ll R, G;
    scanf("%d %d %lld", &T, &N, &R);
    scanf("%lld", &G);
    vector<ll> f(T + 1), s(T + 1), cap(T + 1);
    for (int t = 1; t <= T; t++) scanf("%lld %lld %lld", &f[t], &s[t], &cap[t]);
    vector<int> r(N), dl(N); vector<ll> d(N);
    for (int i = 0; i < N; i++) scanf("%d %d %lld", &r[i], &dl[i], &d[i]);
    for (int i = 0; i < N; i++) printf("%d\n", r[i]);
    for (int t = 1; t <= T; t++) printf("0 0\n");
    return 0;
}
