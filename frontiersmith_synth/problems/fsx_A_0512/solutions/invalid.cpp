// TIER: invalid
// Deliberately infeasible: assign the full budget B to every line so the capacity
// sum vastly exceeds B (L >= 2 always). Must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    ll L, M, S, B;
    if (scanf("%lld %lld %lld %lld", &L, &M, &S, &B) != 4) return 0;
    for (ll i = 0; i < L; i++) printf("%lld%c", B, i + 1 < L ? ' ' : '\n');
    return 0;
}
