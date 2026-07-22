#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Cyclic Cup Seeding"  (cyclic-outcome-bracket-seeding).
//
// Input:  N K
//         N lines of N chars '0'/'1' (row i, col j = '1' iff player i beats j)
//         K ids (sponsor players), K bounty values (aligned)
//
// Output (participant): N integers -- a permutation of 1..N, the player placed
//   in bracket slot 1, 2, ..., N.
//
// Evaluation: play the single-elimination bracket (slot 2i-1 vs slot 2i each
//   round, winners advance) for k = log2(N) rounds. For each sponsor p let
//   rounds(p) = number of consecutive rounds p wins (0 if eliminated round 1,
//   k if champion). Objective (MAX):
//       F = sum over sponsors p of  bounty(p) * (rounds(p) + 1)
//
// Baseline B (checker-computed do-nothing): simulate with the IDENTITY seeding
//   (player i in slot i). B >= sum(bounty) > 0 always, since rounds(p) >= 0.
//   This is exactly what the trivial reference reproduces -> ratio 0.1.
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int N, K;
vector<string> M;          // M[i][j-1] = '1' iff player i beats player j
vector<int> sponsorId, sponsorVal;
vector<char> isSponsor;
vector<int> sponsorIdxOf;  // sponsorIdxOf[playerId] = index into sponsorId/sponsorVal, or -1

static inline bool beats(int a, int b) { return M[a][b - 1] == '1'; }

// Simulate the bracket given an initial slot order `cur` (size must be a power
// of two dividing down to 1). Returns F.
static ll simulate(vector<int> cur) {
    vector<ll> roundsWon(N + 1, 0);
    int r = 0;
    while ((int)cur.size() > 1) {
        r++;
        int half = (int)cur.size() / 2;
        vector<int> nxt(half);
        for (int j = 0; j < half; j++) {
            int a = cur[2 * j], b = cur[2 * j + 1];
            int w = beats(a, b) ? a : b;
            nxt[j] = w;
            if (isSponsor[w]) roundsWon[w] = r;
        }
        cur.swap(nxt);
    }
    ll F = 0;
    for (int i = 0; i < K; i++) {
        int id = sponsorId[i];
        ll rw = roundsWon[id];
        F += (ll)sponsorVal[i] * (rw + 1);
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    K = inf.readInt();
    M.assign(N + 1, string());
    for (int i = 1; i <= N; i++) {
        string row = inf.readToken();
        if ((int)row.size() != N) quitf(_fail, "bad test data: row %d has length %d, expected %d", i, (int)row.size(), N);
        M[i] = row;
    }
    sponsorId.assign(K, 0);
    sponsorVal.assign(K, 0);
    isSponsor.assign(N + 1, 0);
    for (int i = 0; i < K; i++) {
        sponsorId[i] = inf.readInt();
        isSponsor[sponsorId[i]] = 1;
    }
    for (int i = 0; i < K; i++) sponsorVal[i] = inf.readInt();

    // ---- internal baseline B: identity seeding ----
    vector<int> idPerm(N);
    for (int i = 0; i < N; i++) idPerm[i] = i + 1;
    ll B = simulate(idPerm);
    if (B < 1) B = 1;

    // ---- read participant permutation (strict feasibility) ----
    vector<int> perm(N);
    vector<char> seen(N + 1, 0);
    for (int i = 0; i < N; i++) {
        int v = ouf.readInt(1, N, "perm_i");
        if (seen[v]) quitf(_wa, "duplicate player id %d in seeding", v);
        seen[v] = 1;
        perm[i] = v;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after permutation");

    ll F = simulate(perm);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
