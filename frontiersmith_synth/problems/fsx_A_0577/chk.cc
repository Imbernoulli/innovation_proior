#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Frustrated Wang Tiling".
//
// Input:  R C T ; then T lines  W E N S  (tile types).
// Output: R*C tile indices (row-major), each in [0, T-1].  Tiles are reusable.
//
// Objective (MAX): F = number of matched borders, where
//   horizontal border (r,c)-(r,c+1) matches iff tile[r][c].E == tile[r][c+1].W,
//   vertical   border (r,c)-(r+1,c) matches iff tile[r][c].S == tile[r+1][c].N.
//
// Baseline B (checker-computed): the "alternating-domino" tiling -- fill every row with
//   a matched horizontal pair A,B repeated (A,B,A,B,...). Each pair contributes one
//   matched horizontal border, no vertical/inter-pair border matches, so it realizes
//   exactly R*floor(C/2) matches. This is a genuine feasible lower-bound construction
//   and is what the trivial reference reproduces (-> ratio 0.1).
// Score (max): sc = min(1000, 100 * F / max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int R = inf.readInt();
    int C = inf.readInt();
    int T = inf.readInt();
    vector<int> W(T), E(T), N(T), S(T);
    for (int i = 0; i < T; i++){
        W[i] = inf.readInt();
        E[i] = inf.readInt();
        N[i] = inf.readInt();
        S[i] = inf.readInt();
    }

    // ---- read the participant grid (strict bounded reads) ----
    vector<vector<int>> g(R, vector<int>(C));
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            g[r][c] = ouf.readInt(0, T - 1, "tile");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the grid");

    // ---- objective F = matched borders ----
    ll F = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++){
            int t = g[r][c];
            if (c + 1 < C && E[t] == W[g[r][c + 1]]) F++;   // horizontal
            if (r + 1 < R && S[t] == N[g[r + 1][c]]) F++;   // vertical
        }

    // ---- baseline B = R * floor(C/2)  (alternating-domino construction) ----
    ll B = (ll)R * (ll)(C / 2);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
