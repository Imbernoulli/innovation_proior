#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Checker / scorer for purge-carryover-scheduling (rainbow 3D print purge
// volume, MINIMIZE).
//
// Reads L layers of regions, each with a tolerance class (2-3 acceptable
// concrete colors) and a C x C asymmetric purge matrix P. The participant
// outputs, per layer, a print-order permutation of that layer's regions and
// a chosen concrete color per region (must lie in that region's class). The
// objective F is the sum of P[colorOf(prev)][colorOf(next)] over EVERY
// consecutive pair in the full concatenated job sequence (layer 1's order,
// then layer 2's order, ...), so the transition from a layer's last-printed
// color into the next layer's first-printed color (the carryover) is
// charged exactly like any other transition.
//
// Internal baseline B: the checker's own "do-nothing" construction -- print
// each layer's regions in the LISTED (input) order, and for every region
// use the FIRST color listed in its tolerance class (ignore the recoloring
// freedom entirely). This is exactly what solutions/trivial.cpp does.
//
// Score (minimization, standard convention): Ratio = min(1, B / (10 * max(1,F))).
//   B itself (matching the do-nothing baseline)      -> Ratio 0.1
//   10*F <= B                                          -> Ratio capped 1.0
//   (F is floored at 1 before dividing, so a degenerate F=0 instance is
//   scored as if F=1 rather than dividing by zero)

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int L = inf.readInt(1, 100000, "L");
    int C = inf.readInt(2, 100, "C");

    vector<vector<ll>> P(C, vector<ll>(C, 0));
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++)
            P[i][j] = inf.readLong(0, (ll)2000000, "P");

    vector<int> Rl(L);
    vector<vector<vector<int>>> cls(L); // cls[l][r] = list of allowed colors

    for (int l = 0; l < L; l++) {
        Rl[l] = inf.readInt(1, 1000000, "R_l");
        cls[l].resize(Rl[l]);
        for (int r = 0; r < Rl[l]; r++) {
            int k = inf.readInt(2, 3, "class size");
            cls[l][r].resize(k);
            for (int i = 0; i < k; i++)
                cls[l][r][i] = inf.readInt(0, C - 1, "class color");
        }
    }

    // ---- internal baseline B: input order + first-listed color everywhere ----
    ll B = 0;
    {
        int prevColor = -1;
        bool have = false;
        for (int l = 0; l < L; l++) {
            for (int r = 0; r < Rl[l]; r++) {
                int col = cls[l][r][0];
                if (have) B += P[prevColor][col];
                prevColor = col; have = true;
            }
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld <= 0", B);

    // ---- read participant output ----
    ll F = 0;
    int prevColor = -1;
    bool have = false;

    for (int l = 0; l < L; l++) {
        int R = Rl[l];
        vector<int> perm(R);
        vector<char> seen(R, 0);
        for (int i = 0; i < R; i++) {
            int idx = ouf.readInt(0, R - 1, "perm index");
            if (seen[idx]) quitf(_wa, "layer %d: duplicate region index %d in print order", l, idx);
            seen[idx] = 1;
            perm[i] = idx;
        }
        vector<int> colors(R);
        for (int i = 0; i < R; i++) {
            int col = ouf.readInt(0, C - 1, "chosen color");
            int regionIdx = perm[i];
            bool ok = false;
            for (int c : cls[l][regionIdx]) if (c == col) { ok = true; break; }
            if (!ok) quitf(_wa, "layer %d: color %d not in tolerance class of region %d", l, col, regionIdx);
            colors[i] = col;
        }
        for (int i = 0; i < R; i++) {
            if (have) F += P[prevColor][colors[i]];
            prevColor = colors[i]; have = true;
        }
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing output after last layer");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    double ratio = sc / 1000.0;
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
    return 0;
}
