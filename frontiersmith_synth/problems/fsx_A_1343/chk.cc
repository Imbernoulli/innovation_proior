#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Certifying the Losing Positions" (family: nim-variant-invariant).
//
// GAME: N independent piles play a disjunctive-sum subtraction game (normal play:
// last player to move wins). Pile i has its own finite move menu S_i (a set of
// legal removal amounts). The whole game's Sprague-Grundy value of pile size a is
// g_i(a) = mex{ g_i(a-s) : s in S_i, s<=a }, g_i(0)=0. A position (a_1..a_N) is a
// LOSS for the player about to move ("P-position") iff XOR_i g_i(a_i) == 0. Every
// finite-menu subtraction game's Grundy sequence is guaranteed eventually periodic
// (finite window of previous values -> finite state -> pigeonhole), but the lead-in
// and period are NOT given by any simple formula in general (only for the classic
// interval S_i={1..C} is period exactly C+1; for a general/sparse S_i it is not).
//
// PARTICIPANT OUTPUT (the invariant certificate): for each pile i, a period p_i and
// a table T_i[0..p_i-1] of claimed Grundy-residue values, used as predicted_g_i(a) =
// T_i[a mod p_i]. A query position is predicted a LOSS for the mover iff
// XOR_i predicted_g_i(a_i) == 0.
//
// SCORE = accuracy term (ACC: fraction of held-out query positions -- including
// astronomically large pile sizes no DP can touch directly -- where the predicted
// P/N label matches the true one, computed here by the checker via DP + verified
// period-detection) PLUS a closure term (CLOS: fraction of residues r in each
// submitted table that satisfy the exact cyclic mex recurrence implied by that
// pile's OWN move menu -- T_i[r] == mex{T_i[(r-s) mod p_i] : s in S_i} -- i.e. is
// your table actually a self-consistent game invariant, not merely a fit).
//
// F = round(10000*ACC + 4000*CLOS). BASELINE B: the checker's own "always predict
// a loss" construction (p_i=1, T_i=[0] for every pile) evaluated the same way.
// sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const int A_PROBE = 8000;      // DP prefix length used to establish ground truth
static const int TAIL = 4000;         // suffix window checked for periodicity
static const int PMAX_SEARCH = 1500;  // max candidate period the checker searches
static const int R_START = A_PROBE - TAIL;   // anchor index, deep past any transient
static const int PMAX_SUBMIT = 2000;  // max period a participant may declare
static const int TMAX = 63;           // max table value a participant may declare

vector<int> grundyDP(const vector<int>& S) {
    vector<int> g(A_PROBE, 0);
    for (int a = 1; a < A_PROBE; a++) {
        bool seenArr[16] = {false};
        for (int s : S) {
            if (s <= a) {
                int v = g[a - s];
                if (v >= 0 && v < 16) seenArr[v] = true;
            }
        }
        int m = 0;
        while (m < 16 && seenArr[m]) m++;
        g[a] = m;
    }
    return g;
}

// smallest period p in [1,PMAX_SEARCH] with g[a]==g[a-p] for all a in [R_START+p, A_PROBE-1].
// Returns -1 if none found within the search bound (should not happen for this
// problem's generator parameter ranges -- verified empirically at design time).
int findPeriod(const vector<int>& g) {
    for (int p = 1; p <= PMAX_SEARCH; p++) {
        bool ok = true;
        for (int a = R_START + p; a < A_PROBE; a++) {
            if (g[a] != g[a - p]) { ok = false; break; }
        }
        if (ok) return p;
    }
    return -1;
}

inline int trueGrundy(ll a, const vector<int>& g, int p) {
    if (a < A_PROBE) return g[(int)a];
    ll r = R_START + (a - R_START) % p;
    return g[(int)r];
}

inline int emod(ll x, int m) {  // Euclidean mod, x can be "negative-ish" (small range here)
    ll r = x % m;
    if (r < 0) r += m;
    return (int)r;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    vector<vector<int>> S(N);
    for (int i = 0; i < N; i++) {
        int K = inf.readInt();
        S[i].resize(K);
        for (int j = 0; j < K; j++) S[i][j] = inf.readInt();
    }

    // ---- ground truth per pile ----
    vector<vector<int>> garr(N);
    vector<int> trueP(N);
    for (int i = 0; i < N; i++) {
        garr[i] = grundyDP(S[i]);
        trueP[i] = findPeriod(garr[i]);
        if (trueP[i] == -1) trueP[i] = 1;  // defensive fallback; not expected to trigger
    }

    // ---- read participant certificate ----
    vector<int> p(N);
    vector<vector<int>> T(N);
    for (int i = 0; i < N; i++) {
        p[i] = ouf.readInt(1, PMAX_SUBMIT, "period");
        T[i].resize(p[i]);
        for (int j = 0; j < p[i]; j++) T[i][j] = ouf.readInt(0, TMAX, "table value");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the N certificate lines");

    // ---- CLOS: cyclic self-consistency of the submitted tables against each pile's OWN move menu ----
    ll closTotal = 0, closMatch = 0;
    for (int i = 0; i < N; i++) {
        int pi = p[i];
        for (int r = 0; r < pi; r++) {
            bool seenArr[16] = {false};
            for (int s : S[i]) {
                int idx = emod((ll)r - s, pi);
                int v = T[i][idx];
                if (v >= 0 && v < 16) seenArr[v] = true;
            }
            int mexv = 0;
            while (mexv < 16 && seenArr[mexv]) mexv++;
            closTotal++;
            if (mexv == T[i][r]) closMatch++;
        }
    }
    double CLOS = closTotal > 0 ? (double)closMatch / (double)closTotal : 0.0;

    // ---- read queries, compute ACC (participant certificate) and BACC (baseline "always a loss") ----
    int NQ = inf.readInt();
    ll accMatch = 0, baseMatch = 0;
    for (int q = 0; q < NQ; q++) {
        int predXor = 0, trueXor = 0;
        for (int i = 0; i < N; i++) {
            ll a = inf.readLong();
            predXor ^= T[i][(int)(a % p[i])];
            trueXor ^= trueGrundy(a, garr[i], trueP[i]);
        }
        bool predLoss = (predXor == 0);
        bool trueLoss = (trueXor == 0);
        if (predLoss == trueLoss) accMatch++;
        if (trueLoss) baseMatch++;  // baseline always predicts "loss"
    }
    double ACC = NQ > 0 ? (double)accMatch / (double)NQ : 0.0;
    double BACC = NQ > 0 ? (double)baseMatch / (double)NQ : 0.0;

    ll F = llround(10000.0 * ACC + 4000.0 * CLOS);
    ll B = llround(10000.0 * BACC + 4000.0 * 0.0);  // baseline table fails every closure check
    if (F < 0) F = 0;
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld ACC=%.4f CLOS=%.4f Ratio: %.6f", F, B, ACC, CLOS, sc / 1000.0);
    return 0;
}
