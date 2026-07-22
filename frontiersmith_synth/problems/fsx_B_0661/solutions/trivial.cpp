// TIER: trivial
// Pick the M ignitable charges with the largest own value, ignoring reach entirely.
// This is exactly the checker's baseline construction -> ratio 0.1 by design.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<ll> V(N);
    vector<int> C(N);
    for (int i = 0; i < N; i++){
        ll x, y, r, t;
        scanf("%lld %lld %lld %lld %lld %d", &x, &y, &V[i], &r, &t, &C[i]);
    }
    vector<int> ignitable;
    for (int i = 0; i < N; i++) if (C[i] == 1) ignitable.push_back(i);
    sort(ignitable.begin(), ignitable.end(), [&](int a, int b){
        if (V[a] != V[b]) return V[a] > V[b];
        return a < b;
    });
    for (int k = 0; k < M; k++) printf("%d%c", ignitable[k] + 1, k + 1 < M ? ' ' : '\n');
    return 0;
}
