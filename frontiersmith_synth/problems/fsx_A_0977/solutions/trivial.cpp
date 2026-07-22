// TIER: trivial
// The literal naive idea: make each cable's rest length exactly equal to the
// distance its endpoints have in the target shape. It "looks" done (zero
// tension error at target) but every cable ends up carrying ZERO force there,
// so the target is not a stable prestressed equilibrium (the checker's own
// baseline construction).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int n, s, c;
    scanf("%d %d %d", &n, &s, &c);
    vector<ll> target(n);
    for (int i = 0; i < n; i++) scanf("%lld", &target[i]);
    for (int i = 0; i < s; i++) { ll u,v,d; scanf("%lld %lld %lld", &u,&v,&d); }
    for (int i = 0; i < c; i++) {
        ll u, v, k;
        scanf("%lld %lld %lld", &u, &v, &k);
        double gap = fabs((double)target[v] - (double)target[u]);
        printf("%.6f\n", gap);
    }
    return 0;
}
