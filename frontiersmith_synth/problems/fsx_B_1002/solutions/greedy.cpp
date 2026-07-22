// TIER: greedy
// The obvious first attempt: classic earliest-deadline-first scheduling. Sort
// orders by deadline, serve each as soon as the boiler is free within its own
// window [a_i,d_i]. This is a perfectly reasonable SCHEDULING heuristic (it
// respects deadlines well and rarely starves an order for TIME), but it never
// looks at the temperature at all and never flushes -- it fights the boiler's
// natural oscillation instead of riding it, so on orders whose window sits
// several ticks after their arrival (planted sweeps), or that are only
// reachable at all via a flush (trap orders), it lands far from the target.
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

    vector<int> ord(N);
    iota(ord.begin(), ord.end(), 0);
    stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return d[x] < d[y]; });

    vector<pair<ll,int>> plan;
    ll cur = 0;
    for (int i : ord) {
        ll t = max(a[i], cur);
        if (t > d[i]) continue;
        plan.push_back({t, i});
        cur = t + 1;
    }
    sort(plan.begin(), plan.end());

    printf("%d\n", (int) plan.size());
    for (auto &pr : plan) printf("%lld %d\n", pr.first, pr.second);
    return 0;
}
