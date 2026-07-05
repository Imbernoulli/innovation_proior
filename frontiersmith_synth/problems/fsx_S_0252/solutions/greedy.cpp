// TIER: greedy
// Reactive: each step produce only the minimum needed to keep every reserve >= 0,
// buying spot power when solar is available that step, grid otherwise. No pre-storing.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, T; ll Q, S, P;
    scanf("%d %d %lld %lld %lld", &N, &T, &Q, &S, &P);
    vector<int> solar(T); for (auto& z : solar) scanf("%d", &z);
    vector<ll> pr(T); for (auto& z : pr) scanf("%lld", &z);
    vector<ll> Cap(N), I(N);
    vector<vector<ll>> L(N, vector<ll>(T));
    for (int i = 0; i < N; i++) {
        scanf("%lld %lld", &Cap[i], &I[i]);
        for (int t = 0; t < T; t++) scanf("%lld", &L[i][t]);
    }
    vector<ll> x(N);
    for (int t = 0; t < T; t++) {
        ll total = 0;
        for (int i = 0; i < N; i++) {
            ll need = L[i][t] - I[i];
            if (need < 0) need = 0;
            x[i] = need;
            total += need;
        }
        int src = 2;
        if (total > 0 && solar[t] == 1) src = 1;   // use cheap spot only if we must produce now
        printf("%d", src);
        for (int i = 0; i < N; i++) printf(" %lld", x[i]);
        printf("\n");
        // apply
        for (int i = 0; i < N; i++) {
            ll v = I[i] + x[i]; if (v > Cap[i]) v = Cap[i]; v -= L[i][t];
            I[i] = v;
        }
    }
    return 0;
}
