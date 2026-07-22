// TIER: trivial
// The unsophisticated but obviously "safe" first instinct: grab the TOPK=6
// highest-value points one at a time, homing before each pickup so a reversal can
// never spoil it. No tour planning, no thought about the other points at all.
// This is deliberately identical to the checker's own internal baseline
// construction B, so it always reproduces ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const int TOPK = 6;

int main() {
    int n;
    ll bxp, bxm, byp, bym, C, Lmax;
    scanf("%d %lld %lld %lld %lld %lld %lld", &n, &bxp, &bxm, &byp, &bym, &C, &Lmax);
    vector<ll> V(n + 1);
    for (int i = 1; i <= n; i++) {
        ll x, y, v, tol;
        scanf("%lld %lld %lld %lld", &x, &y, &v, &tol);
        V[i] = v;
    }
    int K = min(n, TOPK);
    vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i + 1;
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (V[a] != V[b]) return V[a] > V[b];
        return a < b;
    });
    printf("%d\n", 2 * K);
    for (int k = 0; k < K; k++) printf("0 %d ", idx[k]);
    printf("\n");
    return 0;
}
