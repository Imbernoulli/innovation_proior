#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Dicing a Scarred Wafer into Unequal Chips".
//
// Input : N K ; K lines "a_i b_i v_i" (die catalog, exactly one die is 1x1) ;
//         D ; D lines "r c" (1-indexed defective cells).
// Output: H p_1..p_H ; V q_1..q_V ; then (H+1)*(V+1) integers in row-major
//         order over the resulting rectangles (row-bands top->bottom, then
//         column-bands left->right within each row-band). 0 = bare rectangle,
//         j in [1,K] = realize die j there (rotation allowed, excess wasted).
//
// Feasibility: cut positions strictly increasing in [1,N-1]; every nonzero
//   assignment must FIT the rectangle (in either orientation) and the
//   rectangle must be entirely defect-free.
//
// Objective (MAX): F = sum of v_j over every rectangle assigned a fitting,
//   defect-free die (dies are unlimited supply).
//
// Baseline B (checker-internal, "any-feasible" construction): dice the
//   FINEST possible grid (every 1x1 cell its own rectangle) and realize the
//   base 1x1 die on every defect-free cell. B = (#clean cells) * value(1x1).
//   This exactly matches solutions/trivial.cpp, so trivial always scores
//   ratio == 0.100000.
//
// Score: sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int N, K;
vector<ll> A_, Bd_, Val_;      // 1-indexed: die i has dims A_[i] x Bd_[i], value Val_[i]
vector<vector<ll>> pre;        // (N+1) x (N+1) 2D prefix sum of defect grid

inline ll rectDefects(int r1, int r2, int c1, int c2) {
    return pre[r2][c2] - pre[r1 - 1][c2] - pre[r2][c1 - 1] + pre[r1 - 1][c1 - 1];
}

inline bool fits(int idx, ll w, ll h) {
    ll a = A_[idx], b = Bd_[idx];
    return (a <= w && b <= h) || (b <= w && a <= h);
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt(1, 1000000, "N");
    K = inf.readInt(1, 1000000, "K");
    A_.assign(K + 1, 0);
    Bd_.assign(K + 1, 0);
    Val_.assign(K + 1, 0);
    for (int i = 1; i <= K; i++) {
        A_[i] = inf.readInt(1, N, "a_i");
        Bd_[i] = inf.readInt(1, N, "b_i");
        Val_[i] = inf.readLong(1, (ll)2e9, "v_i");
    }

    int D = inf.readInt(0, N * N, "D");
    vector<vector<char>> grid(N + 2, vector<char>(N + 2, 0));
    for (int i = 0; i < D; i++) {
        int r = inf.readInt(1, N, "r");
        int c = inf.readInt(1, N, "c");
        grid[r][c] = 1;
    }

    pre.assign(N + 1, vector<ll>(N + 1, 0));
    ll cleanCells = 0;
    for (int r = 1; r <= N; r++) {
        for (int c = 1; c <= N; c++) {
            pre[r][c] = (ll)grid[r][c] + pre[r - 1][c] + pre[r][c - 1] - pre[r - 1][c - 1];
            if (!grid[r][c]) cleanCells++;
        }
    }

    // ---- baseline B: finest grid, base 1x1 die on every clean cell ----
    ll unitValue = -1;
    for (int i = 1; i <= K; i++) {
        if (A_[i] == 1 && Bd_[i] == 1 && Val_[i] > unitValue) unitValue = Val_[i];
    }
    if (unitValue < 0) quitf(_fail, "input malformed: catalog has no 1x1 die");
    ll Bval = cleanCells * unitValue;
    if (Bval <= 0) quitf(_fail, "input malformed: baseline is not positive");

    // ---- read participant output ----
    int H = ouf.readInt(0, N - 1, "H");
    vector<int> hcut(H);
    for (int i = 0; i < H; i++) hcut[i] = ouf.readInt(1, N - 1, "h_cut");
    for (int i = 1; i < H; i++)
        if (hcut[i] <= hcut[i - 1]) quitf(_wa, "horizontal cut positions not strictly increasing");

    int Vn = ouf.readInt(0, N - 1, "V");
    vector<int> vcut(Vn);
    for (int i = 0; i < Vn; i++) vcut[i] = ouf.readInt(1, N - 1, "v_cut");
    for (int i = 1; i < Vn; i++)
        if (vcut[i] <= vcut[i - 1]) quitf(_wa, "vertical cut positions not strictly increasing");

    vector<pair<int, int>> rowBands, colBands;
    {
        int prev = 1;
        for (int i = 0; i < H; i++) { rowBands.push_back({prev, hcut[i]}); prev = hcut[i] + 1; }
        rowBands.push_back({prev, N});
    }
    {
        int prev = 1;
        for (int i = 0; i < Vn; i++) { colBands.push_back({prev, vcut[i]}); prev = vcut[i] + 1; }
        colBands.push_back({prev, N});
    }

    ll F = 0;
    for (auto &rb : rowBands) {
        int h = rb.second - rb.first + 1;
        for (auto &cb : colBands) {
            int w = cb.second - cb.first + 1;
            int idx = ouf.readInt(0, K, "assign");
            if (idx == 0) continue;
            if (!fits(idx, w, h))
                quitf(_wa, "die %d (%lldx%lld) does not fit rectangle %dx%d", idx, A_[idx], Bd_[idx], w, h);
            ll defc = rectDefects(rb.first, rb.second, cb.first, cb.second);
            if (defc > 0)
                quitf(_wa, "die %d placed on a rectangle containing %lld defective cell(s)", idx, defc);
            F += Val_[idx];
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, Bval));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, Bval, sc / 1000.0);
}
