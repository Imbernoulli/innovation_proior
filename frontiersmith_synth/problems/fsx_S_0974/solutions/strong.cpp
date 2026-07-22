// TIER: strong
// The insight: overtopping an unprotected segment is a FREE detention reservoir -- a
// segment that is never armoured keeps absorbing volume into its own floodplain (at its
// own damage rate) and that volume is permanently removed from the wave before it can
// reach anything further downstream. So the question is not "what can I afford to
// protect, cheapest first" (the monotone protect-more trap) but "which segments are
// actually the load-bearing damage sinks once the free sacrifice zones have done their
// work". We answer that by literally RUNNING the same do-nothing routing simulation the
// checker itself uses and recording how much REALIZED damage each segment absorbs (its
// value times the volume actually diverted there once upstream storage has already
// shaved the wave, including segments whose storage quietly absorbs everything and so
// contribute nothing -- those must NOT be funded, no matter how large their raw
// "value" field is). We then spend the budget armouring the segments responsible for
// the largest slice of that realized baseline damage, most expensive-to-flood first,
// deliberately leaving every other segment at height 0 so it keeps doing its job as a
// sacrifice zone. This is a decomposition of the baseline's own damage total, not a
// static sort on an input field, so it correctly ignores high-value segments the river's
// own upstream storage already protects for free, and correctly prioritises segments
// that truly bear damage in the unprotected run.
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

    // ---- run the do-nothing baseline routing pass, attributing damage per segment ----
    vector<long long> in(T + 1), out(T + 1), damage0(N + 1, 0);
    for (long long t = 1; t <= T; t++) in[t] = q[t];
    for (long long i = 1; i <= N; i++) {
        long long cap = base[i];  // h=0
        long long left = store[i];
        for (long long t = 1; t <= T; t++) {
            long long overflow = max(0LL, in[t] - cap);
            long long divert = min(overflow, left);
            left -= divert;
            damage0[i] += divert * value[i];
            out[t] = in[t] - divert;
        }
        in.swap(out);
    }

    // ---- how much it costs to fully armour a segment against the observed peak ----
    vector<long long> neededFull(N + 1), costFull(N + 1);
    for (long long i = 1; i <= N; i++) {
        neededFull[i] = min(Hmax[i], max(0LL, peak - base[i]));
        costFull[i] = cost[i] * neededFull[i];
    }

    // ---- rank by REALIZED baseline damage (a decomposition of B, not the raw value
    // field), so a high-value segment the river already protects for free is correctly
    // skipped, and cheap-to-fix big realized losses are still preferred on ties ----
    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (damage0[a] != damage0[b]) return damage0[a] > damage0[b];
        if (costFull[a] != costFull[b]) return costFull[a] < costFull[b];
        return a < b;
    });

    vector<long long> h(N + 1, 0);
    long long remaining = Budget;
    for (int idx : order) {
        if (damage0[idx] <= 0) continue;  // never funds a segment that floods for free with h=0
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
