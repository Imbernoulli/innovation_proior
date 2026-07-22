// TIER: trivial
// The laziest feasible recipe: try to pull each order EXACTLY at its own
// arrival tick, in input (arrival) order, and give up on it entirely the
// moment that tick is already taken by an earlier order -- no searching for
// a later free tick, no rescheduling, no temperature check, no flush ever.
// Whenever several orders share (or nearly share) an arrival tick -- which
// happens often, since orders queue up for the same boiler -- only the first
// one is ever served; the rest are simply abandoned.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, Tmax, H0, FMAX;
    ll TLO, THI, THOT, PH, QH, TCOLD, PC, QC, T0, FLUSH_DROP;
    if (scanf("%d %d %lld %lld %lld %lld %lld %lld %lld %lld %lld %d %lld %d",
               &N, &Tmax, &TLO, &THI, &THOT, &PH, &QH, &TCOLD, &PC, &QC, &T0, &H0, &FLUSH_DROP, &FMAX) != 14) return 0;
    vector<ll> a(N), d(N);
    for (int i = 0; i < N; i++) {
        ll lo, hi, w;
        scanf("%lld %lld %lld %lld %lld", &a[i], &d[i], &lo, &hi, &w);
    }

    vector<char> usedTick(Tmax, 0);
    vector<pair<ll,int>> plan;
    for (int i = 0; i < N; i++) {
        ll t = a[i];
        if (t < 0 || t >= Tmax || usedTick[t]) continue;   // no retry, just give up
        usedTick[t] = 1;
        plan.push_back({t, i});
    }
    sort(plan.begin(), plan.end());

    printf("%d\n", (int) plan.size());
    for (auto &pr : plan) printf("%lld %d\n", pr.first, pr.second);
    return 0;
}
