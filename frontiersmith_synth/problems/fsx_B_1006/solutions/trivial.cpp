// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Release nothing, ever. Reproduces the checker's own internal baseline B
// exactly, so this scores ratio == 0.1 on every test by construction.
int main() {
    int K, T;
    scanf("%d %d", &K, &T);
    for (int i = 1; i <= K; i++) { ll a, b, c, d; scanf("%lld %lld %lld %lld", &a, &b, &c, &d); }
    for (int i = 1; i < K; i++) { ll x; scanf("%lld", &x); }
    for (int t = 1; t <= T; t++) { ll x; scanf("%lld", &x); }
    for (int i = 1; i <= K; i++)
        for (int t = 1; t <= T; t++) { ll x; scanf("%lld", &x); }

    for (int t = 1; t <= T; t++)
        for (int i = 1; i <= K; i++)
            printf("0%c", (i == K) ? '\n' : ' ');
    return 0;
}
