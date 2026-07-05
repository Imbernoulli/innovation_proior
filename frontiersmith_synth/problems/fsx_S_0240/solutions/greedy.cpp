// TIER: greedy
// Cost-aware marginal greedy: process tasks in deadline order; place each at the
// in-window step (with spare throughput) that minimizes the marginal cost delta.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T, N, M; ll G;
vector<ll> f, s; vector<int> cap;

static inline ll stepCost(int t, int y) {
    if (y <= 0) return 0;
    ll c = f[t];
    int sp = min(y, cap[t]);
    c += s[t] * (ll)sp + G * (ll)(y - sp);
    return c;
}

int main() {
    if (scanf("%d %d %d", &T, &N, &M) != 3) return 0;
    scanf("%lld", &G);
    f.assign(T + 1, 0); s.assign(T + 1, 0); cap.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) scanf("%lld %lld %d", &f[t], &s[t], &cap[t]);
    vector<int> r(N), dl(N);
    for (int i = 0; i < N; i++) scanf("%d %d", &r[i], &dl[i]);

    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (dl[a] != dl[b]) return dl[a] < dl[b];
        if (r[a] != r[b]) return r[a] < r[b];
        return a < b;
    });
    vector<int> y(T + 1, 0), assign(N, -1);
    for (int idx : order) {
        int best = -1; ll bestDelta = LLONG_MAX;
        for (int t = r[idx]; t <= dl[idx]; t++) {
            if (y[t] >= M) continue;
            ll delta = stepCost(t, y[t] + 1) - stepCost(t, y[t]);
            if (delta < bestDelta) { bestDelta = delta; best = t; }
        }
        if (best < 0) { // fallback: earliest in window
            for (int t = r[idx]; t <= dl[idx]; t++) if (y[t] < M) { best = t; break; }
            if (best < 0) best = r[idx];
        }
        assign[idx] = best; if (best >= 1 && best <= T) y[best]++;
    }
    for (int i = 0; i < N; i++) printf("%d\n", assign[i]);
    return 0;
}
