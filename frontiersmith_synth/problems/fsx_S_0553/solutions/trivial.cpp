// TIER: trivial
// Reproduces the checker's internal baseline B EXACTLY: widely-spaced grid drop,
// cell = 3*maxW by 3*maxH, first min(M, cols*rows) parts in input order, cut in
// input order. -> F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ll N, M, DA, DB, PW, PG, THR;
    if (!(cin >> N >> M >> DA >> DB >> PW >> PG >> THR)) return 0;
    vector<ll> w(M + 1), h(M + 1), v(M + 1), q(M + 1);
    ll maxW = 1, maxH = 1;
    for (int i = 1; i <= M; i++){
        cin >> w[i] >> h[i] >> v[i] >> q[i];
        maxW = max(maxW, w[i]); maxH = max(maxH, h[i]);
    }
    ll cellW = 3 * maxW, cellH = 3 * maxH;
    ll cols = (cellW > 0) ? N / cellW : 0;
    ll rows = (cellH > 0) ? N / cellH : 0;
    ll cap  = cols * rows;
    ll place = min<ll>(min<ll>(M, cap), 25);   // K0 = 25 reference parts (matches checker)

    // print in the identical order the checker uses for B
    printf("%lld\n", place);
    for (ll k = 0; k < place; k++){
        ll col = k % cols, row = k / cols;
        ll x = col * cellW, y = row * cellH;
        printf("%lld %lld %lld\n", k + 1, x, y);
    }
    return 0;
}
