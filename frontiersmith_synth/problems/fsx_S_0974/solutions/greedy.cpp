// TIER: greedy
// The obvious "protect what you can afford" heuristic: for each segment compute the
// ducat cost of fully armouring it against the observed peak inflow (raising its
// capacity to at least peak), sort segments by that cost ascending, and fully armour as
// many as the budget affords, cheapest first (spending any leftover partially on the
// first segment it can no longer fully afford). This is a completely reasonable
// "protect more, starting from what's cheap" instinct -- it never deliberately leaves a
// segment weak on purpose, and does not know that an un-armoured cheap segment quietly
// shaves the wave for everyone downstream. On skewed maps (a handful of very valuable,
// low-tolerance segments among many cheap high-tolerance ones) it burns the budget
// fully protecting numerous cheap pastures -- which also kills their free flood
// detention -- leaving the valuable segments to face the (nearly) undiminished wave.
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long N, T, Budget;
    if (scanf("%lld %lld %lld", &N, &T, &Budget) != 3) return 0;
    vector<long long> base(N + 1), Hmax(N + 1), cost(N + 1), value(N + 1), store(N + 1);
    for (long long i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &base[i], &Hmax[i], &cost[i], &value[i], &store[i]);
    vector<long long> q(T + 1);
    long long peak = 0;
    for (long long t = 1; t <= T; t++) { scanf("%lld", &q[t]); peak = max(peak, q[t]); }

    vector<long long> neededFull(N + 1), costFull(N + 1);
    for (long long i = 1; i <= N; i++) {
        neededFull[i] = min(Hmax[i], max(0LL, peak - base[i]));
        costFull[i] = cost[i] * neededFull[i];
    }
    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (costFull[a] != costFull[b]) return costFull[a] < costFull[b];
        return a < b;
    });

    vector<long long> h(N + 1, 0);
    long long remaining = Budget;
    for (int idx : order) {
        if (remaining >= costFull[idx]) {
            h[idx] = neededFull[idx];
            remaining -= costFull[idx];
        } else {
            long long afford = (cost[idx] > 0) ? remaining / cost[idx] : 0;
            afford = min(afford, neededFull[idx]);
            h[idx] = afford;
            remaining -= afford * cost[idx];
        }
    }

    for (long long i = 1; i <= N; i++) printf("%lld%c", h[i], (i == N) ? '\n' : ' ');
    return 0;
}
