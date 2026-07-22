#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "One rule paints the whole mural"  (ca-target-painter).
//
// Input:  N K T r0 c0 b  ; then N lines of N chars ('0'/'1') = target image.
//   Seed box = rows [r0,r0+b), cols [c0,c0+b).
//
// Output (participant): describe ONE outer-totalistic CA + a bounded sparse seed.
//   Line 1: 18 integers  B0 B1 ... B8 S0 S1 ... S8  (each 0 or 1).
//           A dead cell with exactly j live Moore-neighbours becomes live iff Bj=1;
//           a live cell with exactly j live neighbours stays live iff Sj=1.
//   Line 2: m  (0 <= m <= K)  number of initially-live seed cells.
//   Next m lines: r c   (each inside the box; all distinct).
//
// Evaluation: run the CA synchronously for T steps (cells outside the grid are
//   dead).  Let TP = live cells that are ON in the target, FP = live cells that
//   are OFF in the target.  Objective (MAX):  F = max(0, TP - FP).
//
// Baseline B (checker-computed do-nothing): B = min(K, #target cells in the box).
//   This is exactly the "freeze <=K target pixels in the seed, use the identity
//   rule" construction the trivial reference reproduces  ->  ratio 0.1.
// Score (max): sc = min(1000, 100 * F / max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N  = inf.readInt();
    int K  = inf.readInt();
    int T  = inf.readInt();
    int r0 = inf.readInt();
    int c0 = inf.readInt();
    int b  = inf.readInt();
    vector<uint8_t> tgt(N * N, 0);
    for (int r = 0; r < N; r++){
        string row = inf.readToken();
        for (int c = 0; c < N; c++) tgt[r * N + c] = (row[c] == '1') ? 1 : 0;
    }

    // ---- internal baseline B = min(K, target cells inside the box) ----
    ll boxCnt = 0;
    for (int r = r0; r < r0 + b; r++) for (int c = c0; c < c0 + b; c++)
        if (tgt[r * N + c]) boxCnt++;
    ll B = min((ll)K, boxCnt);
    if (B < 1) B = 1;

    // ---- read participant CA rule + seed (strict feasibility) ----
    int Bmask = 0, Smask = 0;
    for (int j = 0; j <= 8; j++){
        int v = ouf.readInt(0, 1, format("B%d", j).c_str());
        Bmask |= (v << j);
    }
    for (int j = 0; j <= 8; j++){
        int v = ouf.readInt(0, 1, format("S%d", j).c_str());
        Smask |= (v << j);
    }
    int m = ouf.readInt(0, K, "m");
    vector<uint8_t> g(N * N, 0);
    set<pair<int,int>> seen;
    for (int i = 0; i < m; i++){
        int r = ouf.readInt(r0, r0 + b - 1, "seed_r");
        int c = ouf.readInt(c0, c0 + b - 1, "seed_c");
        if (!seen.insert({r, c}).second)
            quitf(_wa, "duplicate seed cell (%d,%d)", r, c);
        g[r * N + c] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after seed");

    // ---- simulate T steps (outer-totalistic, dead boundary) ----
    vector<uint8_t> h(N * N, 0);
    for (int s = 0; s < T; s++){
        for (int r = 0; r < N; r++) for (int c = 0; c < N; c++){
            int cnt = 0;
            for (int dr = -1; dr <= 1; dr++) for (int dc = -1; dc <= 1; dc++){
                if (!dr && !dc) continue;
                int nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
                cnt += g[nr * N + nc];
            }
            int self = g[r * N + c];
            h[r * N + c] = (uint8_t)(self ? ((Smask >> cnt) & 1) : ((Bmask >> cnt) & 1));
        }
        g.swap(h);
    }

    ll TP = 0, FP = 0;
    for (int i = 0; i < N * N; i++){
        if (g[i]){ if (tgt[i]) TP++; else FP++; }
    }
    ll F = TP - FP;
    if (F < 0) F = 0;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld TP=%lld FP=%lld B=%lld Ratio: %.6f",
          F, TP, FP, B, sc / 1000.0);
    return 0;
}
