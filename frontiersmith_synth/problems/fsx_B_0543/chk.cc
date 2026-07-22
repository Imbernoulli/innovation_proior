#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer -- family: taxiway-crossing-waves.
//
// Input:  N P W d ; then N lines "rel tau w alley".
//   Slot: tick t crossable iff (t % P) < W.  Alley: same-alley pushbacks >= d.
//
// Output (participant): N lines "s_i x_i" in input order.
//   s_i = pushback tick (>= rel_i), x_i = crossing tick.
// Feasible iff: s>=rel; same-alley pushbacks >= d apart; x >= s+tau; x in a slot;
//   all x distinct.  Objective (MIN): F = sum w_i*(x_i - (rel_i+tau_i)).
//
// Baseline B (checker-built do-nothing): input order; pushback ASAP subject to
//   release + alley; then cross ALONE in the next free slot -- one aircraft per
//   slot, at the slot opening tick (usable offset 0 only).  B = F of that.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000  (trivial -> 0.1).
// -----------------------------------------------------------------------------

static const ll XMAX = 1000000000LL; // 1e9 cap on s_i, x_i (keeps F within ll)

// smallest tick t >= a with (t % P) < cap and t not in `used`
static ll nextFreeTick(ll a, ll P, ll cap, unordered_set<ll>& used) {
    ll t = a < 0 ? 0 : a;
    while (true) {
        ll off = t % P;
        if (off >= cap) { t += (P - off); continue; }  // jump to next slot start
        if (used.count(t)) { t++; continue; }
        return t;
    }
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int P = inf.readInt();
    int W = inf.readInt();
    int d = inf.readInt();
    vector<ll> rel(N), tau(N), w(N);
    vector<int> alley(N);
    int maxA = 0;
    for (int i = 0; i < N; i++) {
        rel[i]   = inf.readLong();
        tau[i]   = inf.readLong();
        w[i]     = inf.readLong();
        alley[i] = inf.readInt();
        maxA = max(maxA, alley[i]);
    }

    // ---- internal baseline B (one aircraft per slot, input order) ----
    ll B = 0;
    {
        vector<ll> lastPB(maxA + 1, LLONG_MIN);
        unordered_set<ll> used;
        used.reserve(N * 2);
        for (int i = 0; i < N; i++) {
            ll s = rel[i];
            if (lastPB[alley[i]] != LLONG_MIN) s = max(s, lastPB[alley[i]] + d);
            lastPB[alley[i]] = s;
            ll a = s + tau[i];
            ll x = nextFreeTick(a, P, 1, used); // cap=1 => one per slot
            used.insert(x);
            B += w[i] * (x - (rel[i] + tau[i]));
        }
    }
    if (B < 1) B = 1;

    // ---- read + validate participant output ----
    vector<ll> s(N), x(N);
    for (int i = 0; i < N; i++) {
        s[i] = ouf.readLong(rel[i], XMAX, format("s_%d", i).c_str()); // enforces s>=rel
        x[i] = ouf.readLong(0, XMAX, format("x_%d", i).c_str());
        if (x[i] % P >= W)
            quitf(_wa, "aircraft %d crosses at tick %lld which is not in an open slot (t mod P = %lld >= W=%d)",
                  i, x[i], x[i] % P, W);
        if (x[i] < s[i] + tau[i])
            quitf(_wa, "aircraft %d crosses at %lld before reaching crossing (s+tau=%lld)",
                  i, x[i], s[i] + tau[i]);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // pushback blocking: same-alley pushbacks must be >= d apart
    {
        vector<vector<ll>> byAlley(maxA + 1);
        for (int i = 0; i < N; i++) byAlley[alley[i]].push_back(s[i]);
        for (int a = 0; a <= maxA; a++) {
            sort(byAlley[a].begin(), byAlley[a].end());
            for (size_t k = 1; k < byAlley[a].size(); k++)
                if (byAlley[a][k] - byAlley[a][k - 1] < d)
                    quitf(_wa, "alley %d pushbacks %lld and %lld are < d=%d apart",
                          a, byAlley[a][k - 1], byAlley[a][k], d);
        }
    }

    // one lane: all crossing ticks distinct
    {
        unordered_set<ll> seen;
        seen.reserve(N * 2);
        for (int i = 0; i < N; i++)
            if (!seen.insert(x[i]).second)
                quitf(_wa, "two aircraft cross at the same tick %lld", x[i]);
    }

    // ---- objective ----
    ll F = 0;
    for (int i = 0; i < N; i++) {
        ll delay = x[i] - (rel[i] + tau[i]); // >= 0 since x >= s+tau >= rel+tau
        F += w[i] * delay;
    }
    if (F < 0) F = 0;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
