// TIER: trivial
// Baseline reproduction: commit every crease in plain ascending index order
// 1,2,...,n-1 (always exposed, since crease i's left neighbor i-1 was just
// committed), always folding Mountain -- never even reads req_i or reasons
// about the parity target T. This is EXACTLY the checker's internal baseline
// construction -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ll n, T;
    if (scanf("%lld %lld", &n, &T) != 2) return 0;
    ll km1 = n - 1;
    char buf[4];
    for (ll i = 1; i <= km1; i++) scanf("%s", buf); // req_i, intentionally unused
    vector<ll> w(km1 + 1);
    for (ll i = 1; i <= km1; i++) scanf("%lld", &w[i]);

    printf("%lld\n", km1);
    for (ll i = 1; i <= km1; i++) printf("%lld M\n", i);
    return 0;
}
