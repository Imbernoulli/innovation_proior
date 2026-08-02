// chk.cc -- Splinter Cairns (fsx_A_1365)
// Ground truth uses Conway's tame-sum misere theorem (rigorously correct for
// bounded-subtraction cairns: every subtraction game is "tame" -- Ferguson's
// theorem -- and tame-game disjunctive sums obey the classic P/N rule below).
#include "testlib.h"
#include <vector>
#include <cmath>
using namespace std;

static int grundy(int r, int n) { return n % (r + 1); }

static bool theoremIsP(const vector<int>& gs) {
    bool anyWild = false;
    for (int g : gs) if (g >= 2) { anyWild = true; break; }
    if (anyWild) {
        int x = 0;
        for (int g : gs) x ^= g;
        return x == 0;
    }
    int c1 = 0;
    for (int g : gs) if (g == 1) c1++;
    return (c1 % 2) == 1;
}

int main(int argc, char* argv[]) {
    setName("Splinter Cairns checker");
    registerTestlibCmd(argc, argv);

    int T = inf.readInt(1, 1000000, "T");
    vector<int> M(T);
    vector<vector<int>> R(T), N(T);
    for (int k = 0; k < T; k++) {
        M[k] = inf.readInt(2, 6, "M");
        R[k].resize(M[k]); N[k].resize(M[k]);
        for (int i = 0; i < M[k]; i++) {
            R[k][i] = inf.readInt(2, 5, "r_i");
            N[k][i] = inf.readInt(0, 5000, "n_i");
        }
    }

    int F = 0, B = 0;
    for (int k = 0; k < T; k++) {
        int m = M[k];
        // ---- participant's move (strict feasibility) ----
        int idx = ouf.readInt(0, m - 1, "cairn index");
        long long nOld = N[k][idx];
        long long s = ouf.readLong(-1000000000LL, 1000000000LL, "new count");
        if (s < 0 || s >= nOld) quitf(_wa, "duel %d: new count %lld not in [0, %lld)", k, s, nOld);
        long long removed = nOld - s;
        if (removed < 1 || removed > R[k][idx])
            quitf(_wa, "duel %d: removed %lld stones from cairn %d, but fracture limit is %d",
                  k, removed, idx, R[k][idx]);

        vector<int> gs(m);
        for (int i = 0; i < m; i++) gs[i] = grundy(R[k][i], N[k][i]);
        gs[idx] = grundy(R[k][idx], (int)s);
        if (theoremIsP(gs)) F++;

        // ---- checker's own blind baseline: remove 1 stone from lowest-index
        //      nonempty cairn (recomputed independently, ignores all theory) ----
        int bIdx = -1;
        for (int i = 0; i < m; i++) if (N[k][i] >= 1) { bIdx = i; break; }
        // every duel is guaranteed to have >=1 nonzero cairn by construction
        vector<int> gsB(m);
        for (int i = 0; i < m; i++) gsB[i] = grundy(R[k][i], N[k][i]);
        gsB[bIdx] = grundy(R[k][bIdx], N[k][bIdx] - 1);
        if (theoremIsP(gsB)) B++;
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    double ratio;
    if (F >= B) ratio = 0.1 + 0.75 * (double)(F - B) / (double)max(1, T - B);
    else ratio = 0.1 * (double)F / (double)max(1, B);
    if (!isfinite(ratio)) quitf(_wa, "non-finite ratio");
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;

    quitp(ratio, "F=%d B=%d T=%d Ratio: %.6f", F, B, T, ratio);
    return 0;
}
