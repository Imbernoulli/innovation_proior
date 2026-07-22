// TIER: trivial
// Do-nothing reference: give every bar the cheapest/most flexible catalog entry
// (index 0). This reproduces the checker's own baseline B exactly.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, M, K;
    scanf("%d %d %d", &N, &M, &K);
    for (int i = 0; i < N; i++) { ll x, y; int s; scanf("%lld %lld %d", &x, &y, &s); }
    for (int i = 0; i < M; i++) { int a, b; scanf("%d %d", &a, &b); }
    for (int k = 0; k < K; k++) { ll a, c; scanf("%lld %lld", &a, &c); }
    ll E; scanf("%lld", &E);
    ll Wd, Wc; scanf("%lld %lld", &Wd, &Wc);
    int L; scanf("%d", &L);
    for (int j = 0; j < L; j++) { int nd; ll fx, fy, tx, ty; scanf("%d %lld %lld %lld %lld", &nd, &fx, &fy, &tx, &ty); }

    for (int i = 0; i < M; i++) printf("0\n");
    return 0;
}
