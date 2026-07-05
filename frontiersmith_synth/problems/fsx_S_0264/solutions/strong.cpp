// TIER: strong
// Assign each bed to the cheapest-rain step in its window (like greedy), then run a
// storage-aware supply pass: pre-harvest cheap rain and hoard it in the cistern (bounded
// by tank capacity and by remaining future demand) to serve later expensive / zero-catchment
// steps for free. This is the lookahead the offline replay rewards.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int T, N; ll R, G;
    scanf("%d %d %lld", &T, &N, &R);
    scanf("%lld", &G);
    vector<ll> f(T + 1), s(T + 1), cap(T + 1);
    for (int t = 1; t <= T; t++) scanf("%lld %lld %lld", &f[t], &s[t], &cap[t]);
    vector<int> r(N), dl(N); vector<ll> d(N);
    vector<int> assign(N);
    vector<ll> Demand(T + 1, 0);
    for (int i = 0; i < N; i++) {
        scanf("%d %d %lld", &r[i], &dl[i], &d[i]);
        int best = r[i]; ll bs = s[r[i]];
        for (int t = r[i] + 1; t <= dl[i]; t++) {
            if (s[t] < bs) { bs = s[t]; best = t; }
        }
        assign[i] = best;
        Demand[best] += d[i];
    }

    // suffix demand: SD[t] = sum_{t' >= t} Demand[t']
    vector<ll> SD(T + 2, 0);
    for (int t = T; t >= 1; t--) SD[t] = SD[t + 1] + Demand[t];

    // storage-aware supply
    vector<ll> rainOut(T + 1, 0), mainsOut(T + 1, 0);
    ll L = 0;
    for (int t = 1; t <= T; t++) {
        ll need = Demand[t];
        ll futureNeed = SD[t + 1];           // demand strictly after t
        ll storeRoom = min(R, futureNeed);   // never hoard more than the tank or the future demand
        ll rain;
        if (s[t] < G) {
            // cheap rain: harvest enough to cover this step AND top the cistern toward storeRoom
            ll want = need + storeRoom - L;
            if (want < 0) want = 0;
            rain = min(cap[t], want);
        } else {
            // expensive rain (s == G): only harvest what is needed now
            ll want = need - L;
            if (want < 0) want = 0;
            rain = min(cap[t], want);
        }
        ll avail = L + rain;
        ll mains = 0;
        if (avail < need) { mains = need - avail; avail += mains; }
        L = avail - need;                    // <= storeRoom <= R by construction
        rainOut[t] = rain;
        mainsOut[t] = mains;
    }

    string out; out.reserve((size_t)N * 4);
    char buf[32];
    for (int i = 0; i < N; i++) {
        int len = sprintf(buf, "%d\n", assign[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    for (int t = 1; t <= T; t++)
        printf("%lld %lld\n", rainOut[t], mainsOut[t]);
    return 0;
}
