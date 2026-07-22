// TIER: invalid
// Deliberately infeasible: every line repeats the same element against
// itself (u == v), which the checker must reject immediately -> score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N; ll T;
    scanf("%d %lld", &N, &T);
    for (int i = 0; i < N; i++) { ll x, y, d; scanf("%lld %lld %lld", &x, &y, &d); }
    for (ll t = 0; t < T; t++) printf("0 0\n");
    return 0;
}
