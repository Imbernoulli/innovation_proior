#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "Aim One Ray to Sweep a Mirrored Grid" (maximization).
// The participant picks an initial diagonal direction out of the fixed boundary
// source S, plus up to B internal reflectors ('H' or 'V') to drop into empty
// cells. We simulate the 45-degree billiard ray (edge bounces flip one velocity
// component exactly like a wall; an 'H' reflector flips the vertical component,
// a 'V' reflector flips the horizontal component -- same rule as an edge, just
// placed anywhere) until its (cell,direction) state repeats, and score the
// total weight of distinct cells visited. Baseline B = the same simulation
// with DIR=DR and zero reflectors (always feasible, always positive since S
// itself carries positive weight).
//   ratio = min(1, (100 * F / B) / 1000), capped at 1.0.

static int R, C, Bbudget;
static int sr, sc_;
static vector<string> grid;              // original chars: '.', '#', 'S'
static vector<vector<int>> W;            // weights

static int dirOf(const string &tok, int &dr, int &dc) {
    if (tok == "UL") { dr = -1; dc = -1; return 1; }
    if (tok == "UR") { dr = -1; dc = 1; return 1; }
    if (tok == "DL") { dr = 1; dc = -1; return 1; }
    if (tok == "DR") { dr = 1; dc = 1; return 1; }
    return 0;
}

static int encDir(int dr, int dc) {
    // dr,dc in {-1,1} -> 0..3
    int a = (dr == 1) ? 1 : 0;
    int b = (dc == 1) ? 1 : 0;
    return a * 2 + b;
}

// Simulate on a working grid (with mirrors placed); return total weight of
// distinct visited cells (starting with S).
static long long simulate(const vector<string> &wg, int dr0, int dc0) {
    int dr = dr0, dc = dc0;
    int r = sr, c = sc_;
    vector<char> visitedCell(R * C, 0);
    vector<char> visitedState(R * C * 4, 0);
    long long total = 0;

    visitedCell[r * C + c] = 1;
    total += W[r][c];
    visitedState[(r * C + c) * 4 + encDir(dr, dc)] = 1;

    long long maxSteps = 4LL * R * C + 10;
    for (long long step = 0; step < maxSteps; step++) {
        if (r + dr < 0 || r + dr >= R) dr = -dr;
        if (c + dc < 0 || c + dc >= C) dc = -dc;
        int nr = r + dr, nc = c + dc;
        // Defensive clamp (should already be valid for R,C>=3).
        if (nr < 0 || nr >= R || nc < 0 || nc >= C) break;
        if (wg[nr][nc] == '#') break; // absorbed, blocked cell not counted
        if (wg[nr][nc] == 'H') dr = -dr;
        else if (wg[nr][nc] == 'V') dc = -dc;
        r = nr; c = nc;
        int stIdx = (r * C + c) * 4 + encDir(dr, dc);
        if (visitedState[stIdx]) break; // loop detected
        visitedState[stIdx] = 1;
        if (!visitedCell[r * C + c]) {
            visitedCell[r * C + c] = 1;
            total += W[r][c];
        }
    }
    return total;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    R = inf.readInt();
    C = inf.readInt();
    Bbudget = inf.readInt();
    sr = inf.readInt();
    sc_ = inf.readInt();

    grid.resize(R);
    for (int i = 0; i < R; i++) grid[i] = inf.readToken();
    for (int i = 0; i < R; i++) {
        if ((int)grid[i].size() != C) quitf(_fail, "bad input grid row length");
    }
    W.assign(R, vector<int>(C, 0));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            W[i][j] = inf.readInt();

    if (grid[sr][sc_] != 'S') quitf(_fail, "bad input: S not at declared location");

    // ---- read participant output ----
    string dtok = ouf.readToken();
    int dr0, dc0;
    if (!dirOf(dtok, dr0, dc0)) quitf(_wa, "unknown direction token '%s'", dtok.c_str());

    int k = ouf.readInt(0, Bbudget, "k");
    vector<string> work = grid;
    // Replace S with '.' for mirror/absorb lookups (S itself never blocks/mirrors).
    work[sr][sc_] = '.';

    set<pair<int,int>> placed;
    for (int i = 0; i < k; i++) {
        int r = ouf.readInt(0, R - 1, "r");
        int c = ouf.readInt(0, C - 1, "c");
        string t = ouf.readToken();
        if (t != "H" && t != "V") quitf(_wa, "reflector %d has unknown type '%s'", i, t.c_str());
        if (r == sr && c == sc_) quitf(_wa, "reflector %d placed on the source cell", i);
        if (grid[r][c] != '.') quitf(_wa, "reflector %d placed on a non-empty original cell (%d,%d)", i, r, c);
        if (!placed.insert({r, c}).second) quitf(_wa, "duplicate reflector placement at (%d,%d)", r, c);
        work[r][c] = t[0];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    long long F = simulate(work, dr0, dc0);

    // ---- baseline: DIR=DR, zero mirrors, on the original (mirror-free) grid ----
    vector<string> baseGrid = grid;
    baseGrid[sr][sc_] = '.';
    long long Bline = simulate(baseGrid, 1, 1);
    if (Bline <= 0) Bline = 1;

    double sc_ratio = min(1000.0, 100.0 * (double)F / (double)max(1LL, Bline));
    quitp(sc_ratio / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, Bline, sc_ratio / 1000.0);
    return 0;
}
