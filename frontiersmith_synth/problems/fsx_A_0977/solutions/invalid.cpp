// TIER: invalid
// Deliberately infeasible: prints a negative rest length for one cable (rest
// lengths must be >= 0), which the checker must reject outright.
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
        if (i == 0) printf("%.6f\n", -5.0);
        else printf("%.6f\n", 0.5 * gap);
    }
    return 0;
}
