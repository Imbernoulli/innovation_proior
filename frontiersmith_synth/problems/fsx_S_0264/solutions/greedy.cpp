// TIER: greedy
// Assign each bed to the cheapest-rain step inside its window; source each step's demand
// with no storage (rain up to cap, then mains). Beats trivial by consolidating onto cheap
// catchment steps and saving activations.
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
        // cheapest s_t in window, tie -> earliest
        int best = r[i]; ll bs = s[r[i]];
        for (int t = r[i] + 1; t <= dl[i]; t++) {
            if (s[t] < bs) { bs = s[t]; best = t; }
        }
        assign[i] = best;
        Demand[best] += d[i];
    }
    string out; out.reserve((size_t)N * 4);
    char buf[32];
    for (int i = 0; i < N; i++) {
        int len = sprintf(buf, "%d\n", assign[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    for (int t = 1; t <= T; t++) {
        ll rain = min(cap[t], Demand[t]);
        ll mains = Demand[t] - rain;
        printf("%lld %lld\n", rain, mains);
    }
    return 0;
}
