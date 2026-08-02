#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Everyone Guesses, One Must Be Right".
//
// Input:  N K ; then one line w_1 .. w_N (positive integer information weights).
// Output: for each player i = 1..N (in order), K^(N-1) integers in [0,K-1]: the
//   guess table over all combinations of colours of the OTHER N-1 players. The
//   t-th entry (0-indexed, t in [0, K^(N-1))) corresponds to the "seen" tuple
//   obtained by writing t in base K with N-1 digits, MOST significant digit
//   first, where digit m (0-indexed) is the colour of the m-th smallest-indexed
//   OTHER player (i.e. players 1..N except i, in increasing order).
//
// Objective (MAX):  F = the WORST CASE, over ALL K^N colour assignments
//   X in {0,...,K-1}^N, of  sum_i w_i * [guess_i(X restricted to others) == X_i].
//   The checker brute-forces every assignment (K^N <= 4^9, always small).
//
// Baseline B (checker-computed): the value achieved by the "index round-robin"
//   reference -- player i is pre-assigned target residue r_i = (i-1) mod K and
//   guesses assuming the total of ALL N colours is congruent to r_i (mod K); by
//   a standard identity this reference's worst case equals
//   B = min_{r in 0..K-1} (sum of w_i over players with (i-1) mod K == r).
//   B is guaranteed positive because N >= K+1 in every generated test (every
//   residue gets >=1 player, and all weights are >=1). This is exactly what the
//   trivial reference solution reproduces (-> ratio 0.1).
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int K = inf.readInt();
    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) w[i] = inf.readLong();

    // ---- internal baseline B: index round-robin residue assignment ----
    vector<ll> gB(K, 0);
    for (int i = 1; i <= N; i++) gB[(i - 1) % K] += w[i];
    ll B = *min_element(gB.begin(), gB.end());
    if (B <= 0) B = 1;   // generator guarantees B>0; defensive only

    // ---- read participant's guess tables ----
    ll tableSize = 1;
    for (int e = 0; e < N - 1; e++) tableSize *= K;
    if (tableSize > 20000000LL) quitf(_fail, "test too large (internal)");

    vector<vector<int>> table(N + 1);
    for (int i = 1; i <= N; i++) {
        table[i].resize((size_t)tableSize);
        for (ll t = 0; t < tableSize; t++)
            table[i][(size_t)t] = ouf.readInt(0, K - 1, "guess");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the tables");

    // ---- brute force over all K^N colour assignments ----
    vector<int> X(N + 1, 0);
    ll F = -1;   // will be set to the true minimum on the first assignment
    bool done = false;
    while (!done) {
        ll cur = 0;
        for (int i = 1; i <= N; i++) {
            ll t = 0;
            for (int j = 1; j <= N; j++) {
                if (j == i) continue;
                t = t * K + X[j];
            }
            if (table[i][(size_t)t] == X[i]) cur += w[i];
        }
        if (F < 0 || cur < F) F = cur;

        // odometer increment over X[1..N] in base K
        int pos = N;
        while (pos >= 1) {
            X[pos]++;
            if (X[pos] == K) { X[pos] = 0; pos--; }
            else break;
        }
        if (pos == 0) done = true;
    }
    if (F < 0) F = 0;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
