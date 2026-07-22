// TIER: trivial
// Uniform grid cap_i = floor(B/L): exactly the checker's baseline -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    ll L, M, S, B;
    if (scanf("%lld %lld %lld %lld", &L, &M, &S, &B) != 4) return 0;
    ll U = B / L;
    for (ll i = 0; i < L; i++) printf("%lld%c", U, i + 1 < L ? ' ' : '\n');
    return 0;
}
