// TIER: invalid
// Deliberately infeasible: claims to install every depot (routinely exceeding
// the budget BUD), then routes farm 1's supply along a path that does not use
// a real input edge. Must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int S, J, T; ll BUD;
    scanf("%d %d %d %lld", &S, &J, &T, &BUD);
    ll supply1 = 0;
    for (int i = 1; i <= S; i++){ ll s; scanf("%lld", &s); if (i == 1) supply1 = s; }
    for (int j = 1; j <= J; j++){ ll a, b; scanf("%lld %lld", &a, &b); }
    for (int k = 1; k <= T; k++){
        ll c; int m; scanf("%lld %d", &c, &m);
        for (int t = 0; t < m; t++){ int th, pr; scanf("%d %d", &th, &pr); }
    }
    int depot0 = S, mkt0 = S + J;
    int E; scanf("%d", &E);
    for (int e = 0; e < E; e++){ int u, v, r; scanf("%d %d %d", &u, &v, &r); }

    // Claim every depot installed -- this blows the budget almost always.
    printf("%d\n", J);
    for (int j = 1; j <= J; j++) printf("%d ", depot0 + j);
    printf("\n");
    // One flow block using a fabricated edge that (almost certainly) is not
    // in the input: farm 1 straight to the last market node id, 1 hop.
    int fakeMkt = mkt0 + T;
    printf("1\n");
    printf("1 %d %lld 2\n", fakeMkt, max(1LL, supply1));
    printf("1 %d\n", fakeMkt);
    return 0;
}
