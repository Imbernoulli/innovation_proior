// TIER: invalid
// Deliberately infeasible: every piece is printed at x = W+1000, which is
// out of the checker's bounded read range [0,W] for x -- must score 0
// immediately regardless of the rest of the instance.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int N; ll W, r;
    if (scanf("%d %lld %lld", &N, &W, &r) != 3) return 0;
    vector<ll> w(N), h(N), req(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld %lld", &w[i], &h[i], &req[i]);
    int M; scanf("%d", &M);
    for (int k = 0; k < M; k++) { int a,b,c,d; scanf("%d %d %d %d", &a,&b,&c,&d); }
    for (int i = 0; i < N; i++) printf("%lld 0 0\n", W + 1000);
    return 0;
}
