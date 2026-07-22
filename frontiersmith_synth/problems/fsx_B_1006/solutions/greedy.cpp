// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Reactive, current-period-only policy (the "obvious first approach"):
//   - try to satisfy the downstream reservoir's CURRENTLY REPORTED demand
//     right now, ignoring that it will actually take `lag` periods to arrive
//     (no lag-shifting / no anticipation);
//   - additionally dump anything above a fixed high-fill threshold so the
//     reservoir "feels safe", but only AFTER this period's forced spill has
//     already been computed by the checker, so it can never prevent the
//     overflow that a big inflow spike causes in the very period it lands.
// This never looks at future periods at all.
int main() {
    int K, T;
    scanf("%d %d", &K, &T);
    vector<ll> Cap(K + 1), S0(K + 1), fw(K + 1), dw(K + 1), lag(K + 1);
    for (int i = 1; i <= K; i++) scanf("%lld %lld %lld %lld", &Cap[i], &S0[i], &fw[i], &dw[i]);
    for (int i = 1; i < K; i++) scanf("%lld", &lag[i]);
    vector<ll> I(T + 1);
    for (int t = 1; t <= T; t++) scanf("%lld", &I[t]);
    vector<vector<ll>> D(K + 1, vector<ll>(T + 1));
    for (int i = 1; i <= K; i++)
        for (int t = 1; t <= T; t++) scanf("%lld", &D[i][t]);

    vector<vector<ll>> arrival(K + 2, vector<ll>(T + 1, 0));
    for (int t = 1; t <= T; t++) arrival[1][t] = I[t];
    vector<ll> storI(K + 1);
    for (int i = 1; i <= K; i++) storI[i] = S0[i];

    string out;
    out.reserve((size_t)T * K * 7);
    char buf[32];
    for (int t = 1; t <= T; t++) {
        for (int i = 1; i <= K; i++) {
            ll pre = storI[i] + arrival[i][t];
            ll draw = min(pre, D[i][t]);
            ll postdraw = pre - draw;
            ll spill = max(0LL, postdraw - Cap[i]);
            ll avail = postdraw - spill;

            ll target = (i < K) ? D[i + 1][t] : 0;           // ignores the lag it needs
            ll thresh = Cap[i] * 4 / 5;                        // reactive high-fill dump
            ll overflowGuard = max(0LL, avail - thresh);
            ll r = min(avail, max(target, overflowGuard));

            int n = snprintf(buf, sizeof(buf), "%lld", r);
            out.append(buf, n);
            out.push_back((i == K) ? '\n' : ' ');

            storI[i] = avail - r;
            if (i < K) {
                ll ft = t + lag[i];
                if (ft <= T) arrival[i + 1][ft] += r + spill;
            }
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
