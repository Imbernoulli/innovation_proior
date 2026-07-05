// TIER: strong
// Lookahead pre-store: during cheap spot windows fill rooms toward capacity (subject to the
// plant throughput cap) so that grid use during no-solar stretches is minimized. Coast on the
// stored cold otherwise, buying grid only for the unavoidable minimum.
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
        for (int i = 0; i < N; i++) x[i] = 0;
        int src = 2;

        if (solar[t] == 1 && pr[t] < P) {
            // cheap window: cover needs, then pre-store toward capacity with the leftover throughput
            src = 1;
            ll rem = Q;
            // 1) mandatory: cover the load that the reserve cannot
            for (int i = 0; i < N; i++) {
                ll need = L[i][t] - I[i];
                if (need < 0) need = 0;
                if (need > rem) need = rem;   // (guaranteed to fit: sum needs <= Q)
                x[i] = need; rem -= need;
            }
            // 2) opportunistic: top rooms toward capacity, lowest reserve first
            vector<int> order(N);
            for (int i = 0; i < N; i++) order[i] = i;
            sort(order.begin(), order.end(),
                 [&](int a, int b) { return I[a] + x[a] < I[b] + x[b]; });
            for (int idx : order) {
                if (rem <= 0) break;
                ll room = Cap[idx] - (I[idx] + x[idx]);
                if (room < 0) room = 0;
                ll add = min(rem, room);
                x[idx] += add; rem -= add;
            }
        } else {
            // no cheap spot: buy only the minimum from grid, pause if reserves suffice
            src = 2;
            ll total = 0;
            for (int i = 0; i < N; i++) {
                ll need = L[i][t] - I[i];
                if (need < 0) need = 0;
                x[i] = need; total += need;
            }
            // if spot is available but pricier than grid (rare), still prefer spot for the needs
            if (total > 0 && solar[t] == 1) src = 1;
        }

        printf("%d", src);
        for (int i = 0; i < N; i++) printf(" %lld", x[i]);
        printf("\n");
        for (int i = 0; i < N; i++) {
            ll v = I[i] + x[i]; if (v > Cap[i]) v = Cap[i]; v -= L[i][t];
            I[i] = v;
        }
    }
    return 0;
}
