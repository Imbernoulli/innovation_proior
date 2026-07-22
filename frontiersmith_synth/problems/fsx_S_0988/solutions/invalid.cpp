// TIER: invalid
// Deliberately infeasible: WP=0 violates the required 1<=WP<=300 bound
// (a zero price weight also breaks the "monotone non-increasing in price"
// requirement of a scoring rule) -> checker must reject with 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int Q, n; ll Pmax;
    if (scanf("%d %d %lld", &Q, &n, &Pmax) != 3) return 0;
    for (int q = 0; q <= Q; q++) { ll x; scanf("%lld", &x); }
    for (int j = 0; j < n; j++)
        for (int q = 0; q <= Q; q++) { ll x; scanf("%lld", &x); }
    printf("5 0 %d 0 %lld\n", Q, Pmax);
    return 0;
}
