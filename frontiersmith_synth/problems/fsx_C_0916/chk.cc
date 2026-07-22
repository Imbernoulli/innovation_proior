#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Benchmark Stakes on the Mountain Road".
//
// Input:  L K Q W ; then Q ascending lines  pos cnt.
// Output: exactly K integers in [0,L] (the chosen stake positions).
//
// Objective (MIN): F = sum cnt_i * ceil((pos_i - s_i)/W), s_i = the largest
//   value among {0} U {printed stakes} that is <= pos_i (always exists: 0).
// Baseline B (checker-computed, UNIFORM spacing reference -- "one every
//   n/K meters", the textbook layout that ignores the trace entirely):
//   stakes at floor(L*i/(K+1)) for i=1..K, plus the free trailhead 0;
//   B = the resulting probe cost under that fixed layout.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static inline ll ceilDiv(ll a, ll b) { return (a + b - 1) / b; }

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    ll L = inf.readLong();
    ll K = inf.readLong();
    int Q = inf.readInt();
    ll W = inf.readLong();

    vector<ll> pos(Q), cnt(Q);
    for (int i = 0; i < Q; i++) {
        pos[i] = inf.readLong();
        cnt[i] = inf.readLong();
    }

    // ---- internal baseline B: uniform-spacing reference (ignores the trace) ----
    vector<ll> ustakes;
    ustakes.push_back(0);
    for (ll i = 1; i <= K; i++) ustakes.push_back(L * i / (K + 1));
    sort(ustakes.begin(), ustakes.end());
    ustakes.erase(unique(ustakes.begin(), ustakes.end()), ustakes.end());
    ll B = 0;
    for (int i = 0; i < Q; i++) {
        auto it = upper_bound(ustakes.begin(), ustakes.end(), pos[i]);
        --it;
        B += cnt[i] * ceilDiv(pos[i] - *it, W);
    }
    if (B <= 0) B = 1;   // generator guarantees positive weights/positions

    // ---- read & validate the participant's K stake positions ----
    vector<ll> stakes;
    stakes.reserve((size_t)K + 1);
    stakes.push_back(0);  // free trailhead
    for (ll i = 0; i < K; i++) {
        ll p = ouf.readLong(0, L, "stake");
        stakes.push_back(p);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the %lld-th stake", K);

    sort(stakes.begin(), stakes.end());
    stakes.erase(unique(stakes.begin(), stakes.end()), stakes.end());

    // ---- compute F: for each query, nearest stake <= pos_i ----
    ll F = 0;
    for (int i = 0; i < Q; i++) {
        // largest stake <= pos[i]; stakes is sorted ascending and contains 0
        auto it = upper_bound(stakes.begin(), stakes.end(), pos[i]);
        // it points to first element > pos[i]; the answer is the one before it
        --it;
        ll s = *it;  // guaranteed valid since stakes[0] == 0 <= pos[i] (pos[i] >= 1)
        F += cnt[i] * ceilDiv(pos[i] - s, W);
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
