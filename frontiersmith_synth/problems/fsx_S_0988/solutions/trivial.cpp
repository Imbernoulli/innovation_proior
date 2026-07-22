// TIER: trivial
// Scale-aware capped rule: unit price weight, quality weight from the coarse
// Pmax/Q ratio only, a hard cap on credited quality at the one-third point --
// never looks at any individual guild's cost table or the true value table V.
// Exactly the checker's internal reference rule, so this always scores
// ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int Q, n; ll Pmax;
    if (scanf("%d %d %lld", &Q, &n, &Pmax) != 3) return 0;
    for (int q = 0; q <= Q; q++) { ll x; scanf("%lld", &x); }
    for (int j = 0; j < n; j++)
        for (int q = 0; q <= Q; q++) { ll x; scanf("%lld", &x); }
    ll WQ = max(1LL, min(300LL, Pmax / (ll)Q));
    int q0 = max(1, Q / 3);
    printf("%lld 1 %d 0 %lld\n", WQ, q0, Pmax);
    return 0;
}
