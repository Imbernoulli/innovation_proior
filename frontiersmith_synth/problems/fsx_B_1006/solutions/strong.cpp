// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: because a release must travel `lag_i` periods before it counts as
// arrival at reservoir i+1, "what reservoir i should release at period t" is
// really a backward-shifted pull of ALL downstream local demand, independent
// of physical simulation. Precompute, purely from D and lag (a closed-form
// reformulation, not a simulation):
//   target[K][t]   = 0                                          (no downstream)
//   target[i][t]   = D[i+1][t+lag_i] + target[i+1][t+lag_i]      (i<K)
// target[i][t] is exactly the volume reservoir i must release AT t so it (and
// everything it must forward) lands on time. This also gives a free forecast
// of future arrivals at i (forecast_i(t') = I(t') for i=1, else
// target[i-1][t'-lag_{i-1}]), used to release extra now whenever the forecast
// over a lookahead window would otherwise overflow capacity -- pre-emptying
// ahead of a spike, not reacting to it once it has already landed.
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

    // ---- backward-shifted release target (pure function of D, lag) ----
    vector<vector<ll>> target(K + 2, vector<ll>(T + 2, 0));
    for (int i = K - 1; i >= 1; i--) {
        for (int t = 1; t <= T; t++) {
            ll ft = t + lag[i];
            target[i][t] = (ft <= T) ? (D[i + 1][ft] + target[i + 1][ft]) : 0;
        }
    }
    auto forecast = [&](int i, ll tp) -> ll {
        if (tp < 1 || tp > T) return 0;
        if (i == 1) return I[tp];
        ll s = tp - lag[i - 1];
        if (s < 1 || s > T) return 0;
        return target[i - 1][s];
    };

    const int WIN = 8; // forecast lookahead window (periods)

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

            ll base = min(avail, target[i][t]);
            ll remaining = avail - base;

            ll forecastSum = 0;
            for (int w = 1; w <= WIN; w++) forecastSum += forecast(i, (ll)t + w);
            ll leftoverIfBase = avail - base;
            ll extraNeeded = max(0LL, leftoverIfBase + forecastSum - Cap[i]);
            ll extra = min(remaining, extraNeeded);

            ll r = base + extra;
            if (r > avail) r = avail;
            if (r < 0) r = 0;

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
