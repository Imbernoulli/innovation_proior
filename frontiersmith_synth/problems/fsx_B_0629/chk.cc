#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Master Keys for the Wing Complex". family: masterkey-template-cover
//
// Input:  K M_max L Rmax Budget ; then for each of K wings: n_k, followed by n_k
//         demand strings of length L over {'0'..'3'}.
// Output: T  (0 <= T <= M_max), then T lines "template radius" with
//         0 <= radius <= Rmax and template a length-L string over {'0'..'3'}.
//         Feasibility: sum_i (radius_i+1)^2 <= Budget.
//
// A demand is SERVED if it is within Hamming distance radius_i of template_i for
// some i. F = MIN over wings of (served fraction), scaled by 1e6 for integer math.
// Baseline B (checker-internal, from the input only): for each wing k, the fraction
// of wing k's own demands within a fixed radius_B = max(1,Rmax/3) of wing k's FIRST
// demand string, averaged (MEAN, not MIN) over wings -- always positive since the
// anchor covers itself. Score: sc = min(1000, 100*F/max(1,B)); Ratio = sc/1000.
// -----------------------------------------------------------------------------

static const ll SCALE = 1000000;

static inline int hammingDist(const string &a, const string &b, int L) {
    int d = 0;
    for (int i = 0; i < L; i++) if (a[i] != b[i]) d++;
    return d;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int K = inf.readInt();
    int M_max = inf.readInt();
    int L = inf.readInt();
    int Rmax = inf.readInt();
    ll Budget = inf.readLong();

    vector<vector<string>> demand(K);
    vector<int> n(K);
    for (int k = 0; k < K; k++) {
        n[k] = inf.readInt();
        demand[k].resize(n[k]);
        for (int i = 0; i < n[k]; i++) demand[k][i] = inf.readWord();
    }

    // ---- internal baseline B: per-wing self-anchor ball, MEAN over wings ----
    int radiusB = max(1, Rmax / 3);
    ll Bsum = 0;
    for (int k = 0; k < K; k++) {
        const string &anchor = demand[k][0];
        ll cnt = 0;
        for (int i = 0; i < n[k]; i++)
            if (hammingDist(demand[k][i], anchor, L) <= radiusB) cnt++;
        Bsum += cnt * SCALE / n[k];
    }
    ll B = Bsum / K;
    if (B <= 0) B = 1;

    // ---- read + validate participant output ----
    int T = ouf.readInt(0, 1000000, "T");
    if (T > M_max) quitf(_wa, "T=%d exceeds M_max=%d", T, M_max);
    vector<string> tmpl(T);
    vector<int> rad(T);
    ll totalCost = 0;
    for (int i = 0; i < T; i++) {
        string s = ouf.readWord();
        if ((int)s.size() != L) quitf(_wa, "template %d has length %d, expected %d", i, (int)s.size(), L);
        for (char c : s)
            if (c < '0' || c > '3') quitf(_wa, "template %d has invalid symbol '%c' (need 0..3)", i, c);
        int r = ouf.readInt(0, Rmax, "radius");
        ll cost = (ll)(r + 1) * (r + 1);
        totalCost += cost;
        if (totalCost > Budget)
            quitf(_wa, "budget exceeded: used %lld > Budget=%lld after template %d", totalCost, Budget, i);
        tmpl[i] = s;
        rad[i] = r;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after %d templates", T);

    // ---- objective F: MIN over wings of served fraction ----
    ll F = SCALE; // start at max possible, take min
    for (int k = 0; k < K; k++) {
        ll covered = 0;
        for (int i = 0; i < n[k]; i++) {
            const string &d = demand[k][i];
            bool served = false;
            for (int j = 0; j < T; j++) {
                if (hammingDist(d, tmpl[j], L) <= rad[j]) { served = true; break; }
            }
            if (served) covered++;
        }
        ll fracK = covered * SCALE / n[k];
        F = min(F, fracK);
    }
    if (T == 0) F = 0;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld T=%d cost=%lld/%lld Ratio: %.6f",
          F, B, T, totalCost, Budget, sc / 1000.0);
    return 0;
}
