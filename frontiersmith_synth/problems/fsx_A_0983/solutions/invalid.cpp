// TIER: invalid
// Deliberately infeasible: commits an interior crease (not a boundary, no
// neighbor committed) as the very FIRST move -- fails the checker's
// exposure check immediately -> no Ratio -> scores 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ll n, T;
    if (scanf("%lld %lld", &n, &T) != 2) return 0;
    ll km1 = n - 1;
    // consume the rest of the input (not needed for the attack, but keeps us
    // well-formed if anything downstream ever reads stdin further)
    char buf[4];
    for (ll i = 1; i <= km1; i++) scanf("%s", buf);
    ll tmp;
    for (ll i = 1; i <= km1; i++) scanf("%lld", &tmp);

    ll mid = km1 / 2;
    if (mid < 2) mid = 2;
    if (mid > km1 - 1) mid = km1 - 1;

    printf("1\n%lld M\n", mid);
    return 0;
}
