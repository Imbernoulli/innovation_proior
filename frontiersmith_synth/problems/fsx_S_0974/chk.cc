#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- Duchy Flood Ledger scorer (minimization).
// Reads the levee heights, validates feasibility (range + budget), then simulates the
// deterministic flood-wave routing (segment by segment, time step by time step, with
// cumulative per-segment sacrificial storage) to get the participant's total damage F,
// and separately simulates the same river with every height = 0 to get baseline damage B.
//   ratio = min(1, 0.1 * B / max(1,F))
// so ratio ~= 0.1 for the do-nothing baseline, higher for genuinely lower damage.

static int N, T;
static vector<long long> base_cap, Hmax, cost, value, store, q;

static long long simulate(const vector<long long>& h) {
    vector<long long> in(T + 1), out(T + 1);
    for (int t = 1; t <= T; t++) in[t] = q[t];
    long long damage = 0;
    for (int i = 1; i <= N; i++) {
        long long cap = base_cap[i] + h[i];
        long long left = store[i];
        for (int t = 1; t <= T; t++) {
            long long overflow = in[t] - cap;
            if (overflow < 0) overflow = 0;
            long long divert = min(overflow, left);
            left -= divert;
            damage += divert * value[i];
            out[t] = in[t] - divert;
        }
        in.swap(out);
    }
    return damage;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    T = inf.readInt();
    long long Budget = inf.readLong();

    base_cap.assign(N + 1, 0); Hmax.assign(N + 1, 0); cost.assign(N + 1, 0);
    value.assign(N + 1, 0); store.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) {
        base_cap[i] = inf.readLong();
        Hmax[i] = inf.readLong();
        cost[i] = inf.readLong();
        value[i] = inf.readLong();
        store[i] = inf.readLong();
    }
    q.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) q[t] = inf.readLong();

    // ---- read + validate participant heights ----
    vector<long long> h(N + 1, 0);
    long long spent = 0;
    for (int i = 1; i <= N; i++) {
        long long v = ouf.readLong();
        if (v < 0 || v > Hmax[i])
            quitf(_wa, "height at segment %d = %lld out of range [0,%lld]", i, v, Hmax[i]);
        h[i] = v;
        // defensive overflow guard: cost[i]>=0 and v<=Hmax[i] are both bounded by the
        // (trusted) input, but never let the product silently wrap regardless.
        if (cost[i] > 0 && v > (LLONG_MAX / cost[i]) - 1)
            quitf(_wa, "height %lld at segment %d makes the budget cost overflow", v, i);
        spent += cost[i] * v;
        if (spent > Budget)
            quitf(_wa, "cumulative budget spent %lld exceeds Budget %lld by segment %d",
                  spent, Budget, i);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after %d heights", N);

    long long F = simulate(h);
    vector<long long> zero(N + 1, 0);
    long long B = simulate(zero);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
