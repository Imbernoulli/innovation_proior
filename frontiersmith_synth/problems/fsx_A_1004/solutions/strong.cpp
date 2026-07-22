// TIER: strong
// The insight: the fold conditions bind at the ORBIT SUM around the fixed
// point O of the k-fold rotation, which forces an exact modular-parity
// classification of which interior-crease counts are even reachable at all,
// and which of those get Kawasaki "for free":
//   - extra (interior slots per sector) EVEN  =>  d=k*(extra+1) needs k EVEN
//     to satisfy Maekawa; and when it does, the alternating-gap condition at
//     O is satisfied AUTOMATICALLY for ANY choice of slot positions (the
//     period length extra+1 is odd, so stepping through the k repeated
//     periods hits every gap once at an even global index and once at an
//     odd one) -- so just grab the top-value slots freely.
//   - extra ODD  =>  d is even for ANY k, but Kawasaki is NOT automatic; a
//     mirror-symmetric (palindromic) slot set -- pairs (j,S-j) plus the
//     self-mirrored midpoint slot S/2 -- makes the gap sequence a palindrome
//     of even length, which provably satisfies the alternating-sum balance
//     for ANY value table (no search needed).
// We evaluate every feasible extra under both constructions and keep the one
// maximizing F, then label with the interleave-with-one-flip pattern, which
// is the maximum-transitions labeling achievable under Maekawa's forced
// M/V split (d/2+1, d/2-1) -> 2*min = d-2 transitions.
#include <bits/stdc++.h>
using namespace std;

static bool kawasakiBalanced(int k, int S, const vector<int>& r) {
    int extra = (int)r.size();
    long long d = (long long)k * (extra + 1);
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
        if (g <= 0) return false;
        if (i % 2 == 0) sumEven += g; else sumOdd += g;
    }
    return sumEven == sumOdd;
}

static double evalF(int k, int S, const vector<int>& r, const vector<int>& v,
                     double beta, double gamma) {
    int extra = (int)r.size();
    long long d = (long long)k * (extra + 1);
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
    long long value = 0;
    for (int x : r) value += v[x];
    value *= k;
    long long transitions = d - 2; // interleave-with-one-flip achieves the max
    double entropy = 0.0;
    for (long long i = 0; i < d; i++) {
        double p = (double)gap[i] / (double)total;
        if (p > 0) entropy -= p * log(p);
    }
    return (double)value + beta * (double)transitions + gamma * entropy;
}

static void printLabels(long long d) {
    for (long long i = 0; i < d; i++) {
        int lab = (i % 2 == 0) ? 1 : 0;
        if (i == d - 1) lab = 1; // flip last V -> M for |M-V|=2
        cout << (lab ? 'M' : 'V') << " \n"[i + 1 == d];
    }
}

int main() {
    int k, S, mmax, beta1000, gamma1000;
    cin >> k >> S >> mmax >> beta1000 >> gamma1000;
    vector<int> v(S, 0);
    for (int i = 1; i <= S - 1; i++) cin >> v[i];
    double beta = beta1000 / 1000.0, gamma = gamma1000 / 1000.0;

    // slots sorted by value desc, for the "free" (auto-regime) construction
    vector<int> byValue;
    for (int i = 1; i <= S - 1; i++) byValue.push_back(i);
    sort(byValue.begin(), byValue.end(), [&](int a, int b) { return v[a] > v[b]; });

    int mid = S / 2;
    // mirror pairs (j, S-j) for j=1..mid-1, ranked by combined value desc
    vector<int> pairJ;
    for (int j = 1; j < mid; j++) pairJ.push_back(j);
    sort(pairJ.begin(), pairJ.end(), [&](int a, int b) {
        return v[a] + v[S - a] > v[b] + v[S - b];
    });

    vector<int> best;
    double bestF = -1e18;
    bool haveBest = false;

    auto consider = [&](vector<int> r) {
        if ((int)r.size() > mmax) return;
        sort(r.begin(), r.end());
        r.erase(unique(r.begin(), r.end()), r.end());
        if ((int)r.size() != (int)0) {} // no-op, keep structure simple
        long long extra = (long long)r.size();
        long long d = (long long)k * (extra + 1);
        if (d % 2 != 0) return; // Maekawa impossible
        if (!kawasakiBalanced(k, S, r)) return;
        double F = evalF(k, S, r, v, beta, gamma);
        if (F > bestF) { bestF = F; best = r; haveBest = true; }
    };

    // "auto" candidates: extra even, top-value slots at arbitrary positions
    for (int extra = 0; extra <= mmax; extra += 2) {
        vector<int> r(byValue.begin(), byValue.begin() + extra);
        consider(r);
    }
    // "mirror" candidates: extra odd = 2*numpairs+1 (midpoint + top pairs)
    for (int extra = 1; extra <= mmax; extra += 2) {
        int numpairs = (extra - 1) / 2;
        if (numpairs > (int)pairJ.size()) continue;
        if (mid < 1 || mid > S - 1) continue; // S even guarantees mid valid
        vector<int> r;
        r.push_back(mid);
        for (int t = 0; t < numpairs; t++) {
            r.push_back(pairJ[t]);
            r.push_back(S - pairJ[t]);
        }
        consider(r);
    }
    // safety net: the always-feasible single-midpoint construction
    if (!haveBest) consider(vector<int>{mid});

    long long extra = (long long)best.size();
    long long d = (long long)k * (extra + 1);
    cout << extra << "\n";
    for (size_t i = 0; i < best.size(); i++) cout << best[i] << " \n"[i + 1 == best.size()];
    if (best.empty()) cout << "\n";
    printLabels(d);
    return 0;
}
