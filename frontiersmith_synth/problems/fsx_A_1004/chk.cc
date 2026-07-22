#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Checker/scorer for "Rotating Star Vertex: Symmetric Crease Selection"
// (family: crease-star-flatfold).
//
// Input:  k S mmax beta1000 gamma1000
//         v_1 ... v_{S-1}
// Output: extra
//         r_1 .. r_extra      (strictly increasing, in [1,S-1])
//         label_1 .. label_d  (d=k*(extra+1), each 'M' or 'V', cyclic angular
//                              order: sector0 spoke, sector0 interior creases
//                              ascending, sector1 spoke, ... )
//
// Feasibility (checked EXACTLY, integer arithmetic for gaps):
//   Maekawa: |#M - #V| == 2
//   Kawasaki: sum of even-indexed gaps == sum of odd-indexed gaps (cyclic,
//             starting at index 0 = gap right after the sector-0 spoke)
//
// Objective (max): F = Value + beta*Transitions + gamma*Entropy
//   Value       = k * sum(v[r_i])
//   Transitions = # cyclically-adjacent label pairs that differ
//   Entropy     = -sum_i (g_i/(k*S)) * ln(g_i/(k*S))   (gaps in slot units)
//
// Baseline B (checker-internal, always feasible & always > 0): a NAIVE
// (index-order, never value-ranked) interior-slot fill of ~mmax/2 creases,
// parity-adjusted per k (see buildNaiveBaseline below), same formula.
// Score: sc = min(1000, 100*F/max(eps,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

// Naive (no value information) baseline slot set, sized ~mmax/2, rounded to
// whichever parity is REQUIRED for feasibility given k (see statement/gen
// for the fold conditions): even k -> even count via plain index order
// (always Kawasaki-safe for even k); any k -> mirror-symmetric odd count
// (midpoint + naive low-index pairs) is always Kawasaki-safe too. We use
// the even/index-order form when k is even (simpler) and the mirror form
// when k is odd (required for feasibility at all).
static vector<int> buildNaiveBaseline(int k, int S, int mmax) {
    int target = mmax / 2;
    int mid = S / 2;
    vector<int> rb;
    if (k % 2 == 0) {
        int extra_b = target - (target % 2); // round down to even, >=0
        for (int i = 1; i <= extra_b; i++) rb.push_back(i);
    } else {
        int extra_b = (target % 2 == 0) ? max(1, target - 1) : target; // round down to odd, >=1
        int numpairs = (extra_b - 1) / 2; // always <= mid-1 given k>=2,S>=6,mmax<=S-1
        rb.push_back(mid);
        for (int j = 1; j <= numpairs; j++) { rb.push_back(j); rb.push_back(S - j); }
        sort(rb.begin(), rb.end());
    }
    if (rb.empty()) rb.push_back(mid); // tiny-budget floor: never a zero-crease baseline
    return rb;
}

static double computeF(int k, int S, const vector<int>& r, const vector<int>& v,
                        double beta, double gamma, const vector<int>& label /*0/1, 1=M*/) {
    int extra = (int)r.size();
    long long d = (long long)k * (extra + 1);
    // positions in global slot units (already ascending by construction)
    vector<long long> pos;
    pos.reserve(d);
    for (int t = 0; t < k; t++) {
        pos.push_back((long long)t * S + 0);
        for (int x : r) pos.push_back((long long)t * S + x);
    }
    long long total = (long long)k * S;
    vector<long long> gap(d);
    for (long long i = 0; i < d; i++) {
        long long nxt = (i + 1 < d) ? pos[i + 1] : pos[0] + total;
        gap[i] = nxt - pos[i];
    }
    long long sumEven = 0, sumOdd = 0;
    for (long long i = 0; i < d; i++) {
        if (i % 2 == 0) sumEven += gap[i]; else sumOdd += gap[i];
    }
    (void)sumEven; (void)sumOdd; // feasibility already validated by caller when needed

    long long value = 0;
    for (int x : r) value += v[x];
    value *= k;

    long long transitions = 0;
    for (long long i = 0; i < d; i++) {
        long long j = (i + 1 < d) ? i + 1 : 0;
        if (label[i] != label[j]) transitions++;
    }

    double entropy = 0.0;
    for (long long i = 0; i < d; i++) {
        double p = (double)gap[i] / (double)total;
        if (p > 0) entropy -= p * log(p);
    }

    return (double)value + beta * (double)transitions + gamma * entropy;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int k = inf.readInt();
    int S = inf.readInt();
    int mmax = inf.readInt();
    int beta1000 = inf.readInt();
    int gamma1000 = inf.readInt();
    double beta = beta1000 / 1000.0;
    double gamma = gamma1000 / 1000.0;

    vector<int> v(S, 0);
    for (int i = 1; i <= S - 1; i++) v[i] = inf.readInt();

    // ---- read participant output ----
    int extra = ouf.readInt(0, mmax, "extra");
    vector<int> r(extra);
    int prev = 0;
    for (int i = 0; i < extra; i++) {
        r[i] = ouf.readInt(1, S - 1, "slot");
        if (r[i] <= prev) quitf(_wa, "interior slots must be strictly increasing, got %d after %d", r[i], prev);
        prev = r[i];
    }
    long long d = (long long)k * (extra + 1);
    if (d <= 0 || d > 200000) quitf(_wa, "degree out of sane range: d=%lld", d);

    vector<int> label(d);
    long long cntM = 0, cntV = 0;
    for (long long i = 0; i < d; i++) {
        string tok = ouf.readToken();
        if (tok == "M") { label[i] = 1; cntM++; }
        else if (tok == "V") { label[i] = 0; cntV++; }
        else quitf(_wa, "label %lld must be 'M' or 'V', got '%s'", i, tok.c_str());
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the expected %lld labels", d);

    // Maekawa
    if (llabs(cntM - cntV) != 2) quitf(_wa, "Maekawa violated: |M-V|=|%lld-%lld|!=2", cntM, cntV);

    // Kawasaki (exact integer gaps)
    {
        vector<long long> pos;
        pos.reserve(d);
        for (int t = 0; t < k; t++) {
            pos.push_back((long long)t * S + 0);
            for (int x : r) pos.push_back((long long)t * S + x);
        }
        long long total = (long long)k * S;
        long long sumEven = 0, sumOdd = 0;
        for (long long i = 0; i < d; i++) {
            long long nxt = (i + 1 < d) ? pos[i + 1] : pos[0] + total;
            long long g = nxt - pos[i];
            if (g <= 0) quitf(_wa, "non-positive gap at index %lld", i);
            if (i % 2 == 0) sumEven += g; else sumOdd += g;
        }
        if (sumEven != sumOdd)
            quitf(_wa, "Kawasaki violated: even-gap-sum=%lld != odd-gap-sum=%lld", sumEven, sumOdd);
    }

    double F = computeF(k, S, r, v, beta, gamma, label);

    // ---- internal baseline: a NAIVE (index-order, not value-ranked) fill
    // sized to roughly mmax/2 interior creases -- big enough that a value-
    // aware construction can't trivially blow past the 10x score cap, but
    // built with no value information (a fair, easy-to-reproduce reference).
    vector<int> rb = buildNaiveBaseline(k, S, mmax);
    long long db = (long long)k * ((long long)rb.size() + 1);
    vector<int> lb(db);
    for (long long i = 0; i < db; i++) lb[i] = (i % 2 == 0) ? 1 : 0; // M,V,M,V,...
    lb[db - 1] = 1; // flip the last V->M to satisfy Maekawa with max transitions
    double B = computeF(k, S, rb, v, beta, gamma, lb);
    if (B < 1e-9) B = 1e-9;

    double sc = min(1000.0, 100.0 * F / B);
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
}
