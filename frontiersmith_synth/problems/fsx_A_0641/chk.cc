#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Codebook Shuffle Against Residue Spies. Minimization.
// Validates sigma is a genuine permutation of {0,...,p-1}. Computes
//   Leak(sigma) = max_{d=1}^{p-1} W[d] * |T(d)|,  T(d) = sum_i chi(sigma[i]-sigma[(i-d)%p])
// via the same O(p^2) routine used for both the participant's F and the checker's own
// baseline B (the leak of the identity shuffle, provably the worst possible: every T(d) for
// a pure shift equals p*chi(d), the maximum magnitude, simultaneously on every channel).
// Score is a log-scaled relative-improvement ratio (see statement.txt), so it stays well
// calibrated even though Leak can shrink by orders of magnitude below B.

static int P;
static vector<int> chiTab;   // chiTab[x] in {-1,0,1}, x in [0,p-1]
static vector<int> W;        // W[1..p-1]

static void buildChi(int p) {
    chiTab.assign(p, -1);
    chiTab[0] = 0;
    for (long long x = 1; x < p; x++) chiTab[(int)((x * x) % p)] = 1;
}

// O(p^2) worst-channel leak of a given permutation.
static long long leakOf(const vector<int>& sigma) {
    int p = P;
    long long best = 0;
    for (int d = 1; d <= p - 1; d++) {
        long long T = 0;
        for (int i = 0; i < p; i++) {
            int j = i - d; if (j < 0) j += p;
            int diff = sigma[i] - sigma[j];
            diff %= p; if (diff < 0) diff += p;
            T += chiTab[diff];
        }
        long long val = (long long)W[d] * llabs(T);
        if (val > best) best = val;
    }
    return best;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    P = inf.readInt(3, 5000, "p");
    W.assign(P, 0);
    for (int d = 1; d <= P - 1; d++) W[d] = inf.readInt(1, 1000, "W");
    buildChi(P);

    // ---- read + validate participant's permutation ----
    vector<int> sigma(P);
    vector<char> seen(P, 0);
    for (int i = 0; i < P; i++) {
        int v = ouf.readInt(0, P - 1, "sigma_i");
        if (seen[v]) quitf(_wa, "value %d repeated in the shuffle (not a permutation)", v);
        seen[v] = 1;
        sigma[i] = v;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the permutation");

    long long F = leakOf(sigma);

    // ---- baseline: identity shuffle (a pure shift; provably worst-case leak) ----
    vector<int> ident(P);
    for (int i = 0; i < P; i++) ident[i] = i;
    long long B = leakOf(ident);
    if (B <= 0) B = 1;

    long long Fc = max(1LL, F);
    double lnB = log((double)B);
    double lnF = log((double)Fc);
    const double lnCap = log(0.1);
    double ratio = 0.1 + 0.9 * (lnB - lnF) / (lnB - lnCap);
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;

    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
    return 0;
}
