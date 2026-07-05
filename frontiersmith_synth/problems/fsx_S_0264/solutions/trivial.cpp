// TIER: trivial
// Reproduce the checker's baseline exactly: water each bed at its release step r_i,
// and source each step's demand with no storage (rain up to cap, then mains). Scores ~0.1.
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
    vector<ll> Demand(T + 1, 0);
    for (int i = 0; i < N; i++) {
        scanf("%d %d %lld", &r[i], &dl[i], &d[i]);
        Demand[r[i]] += d[i];
    }
    // assignments
    string out; out.reserve((size_t)N * 4);
    char buf[32];
    for (int i = 0; i < N; i++) {
        int len = sprintf(buf, "%d\n", r[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    // supply: no storage, rain-first
    for (int t = 1; t <= T; t++) {
        ll rain = min(cap[t], Demand[t]);
        ll mains = Demand[t] - rain;
        printf("%lld %lld\n", rain, mains);
    }
    return 0;
}
