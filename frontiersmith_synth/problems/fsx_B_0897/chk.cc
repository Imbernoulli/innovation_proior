#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Stencil Counters: Bridging the Floating Middles".
// family: bridge-anchored-stencil
//
// Grid R x C (1-indexed). '#' = frame (rigid, always grounded). digit 1..K = island
// id (a floating stencil piece that must be structurally anchored). '.' = background
// (buildable). A second RxC digit grid gives a visual cost 0-9 per cell.
//
// A bridge is an axis-aligned rectangle of background cells that becomes solid
// material, connecting exactly two DIFFERENT structures (frame or island) that
// each present a flat face flush with one short end of the rectangle. Long axis
// (rows spanned >= cols spanned -> "vertical"; else "horizontal") gives LENGTH L
// (cells along the long axis) and WIDTH W (cells across).
//
// Stiffness rule (the physics twist): a rod resists stretching along its OWN axis
// efficiently (axial, ~ W/L) but resists bending ACROSS its axis far less
// efficiently (~ W^3/L^3, classic cantilever beam scaling). A vertical bridge
// therefore feeds its axial term into the island's ROW-stiffness and its (weak)
// bending term into the island's COL-stiffness; a horizontal bridge does the
// opposite. Every island needs BOTH stiffness totals >= its required threshold.
//
// This means ONE bridge must satisfy one axis via the cheap axial term and the
// OTHER axis via the expensive cubic bending term (width blows up on long spans),
// while TWO orthogonal bridges can each satisfy their own axis via the cheap axial
// term alone -- geometry (a triangulated pair) buys stiffness far more cheaply
// than material (one fat rod). That is exactly what the checker's internal
// baseline (a single generously-oversized rod) fails to notice, and what a smart
// solver exploits.
//
// Output: M, then M lines "o r1 c1 r2 c2" (1-indexed rectangle, inclusive;
// o=0 vertical (long axis = rows) / o=1 horizontal (long axis = cols) -- the
// solver STATES orientation explicitly, since required widths can legitimately
// exceed the span length, so shape alone cannot disambiguate a thin-vs-fat rod).
// Objective (MIN): F = total visual cost of all bridge cells (cost counted once
// per cell; bridges may not overlap each other or existing structure).
// Baseline B (checker-internal): for each island, take its geometrically NEAREST
// single direction and build ONE oversized (1.4x safety margin) rod there; sum its
// real visual cost. B is what an unimaginative, safety-first engineer would build.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int R, C, K, A, BD;
vector<string> S;      // 1..R, 1..C  structure grid
vector<string> Cst;    // 1..R, 1..C  cost grid (digit chars)
vector<int> ir1, ir2, ic1, ic2;   // island bounding boxes, 1..K
vector<double> Sreq;              // 1..K

inline char cellS(int r, int c) { return S[r][c]; }
inline int cellCost(int r, int c) { return Cst[r][c] - '0'; }

// direction codes
enum Dir { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

// geometric gap length from island id in a given direction
ll gapLen(int id, int dir) {
    if (dir == UP) return (ll)ir1[id] - 2;
    if (dir == DOWN) return (ll)R - ir2[id] - 1;
    if (dir == LEFT) return (ll)ic1[id] - 2;
    return (ll)C - ic2[id] - 1;
}
// max feasible width in that direction (bounded by the island's own face)
ll maxWidth(int id, int dir) {
    if (dir == UP || dir == DOWN) return ic2[id] - ic1[id] + 1;
    return ir2[id] - ir1[id] + 1;
}
// rectangle (r1,c1,r2,c2) for a hypothetical bridge of given direction/width w,
// centered on the island's face.
void rectFor(int id, int dir, ll L, ll w, int &r1, int &c1, int &r2, int &c2) {
    if (dir == UP) {
        r1 = ir1[id] - (int)L; r2 = ir1[id] - 1;
        int iw = ic2[id] - ic1[id] + 1;
        c1 = ic1[id] + (iw - (int)w) / 2; c2 = c1 + (int)w - 1;
    } else if (dir == DOWN) {
        r1 = ir2[id] + 1; r2 = ir2[id] + (int)L;
        int iw = ic2[id] - ic1[id] + 1;
        c1 = ic1[id] + (iw - (int)w) / 2; c2 = c1 + (int)w - 1;
    } else if (dir == LEFT) {
        c1 = ic1[id] - (int)L; c2 = ic1[id] - 1;
        int ih = ir2[id] - ir1[id] + 1;
        r1 = ir1[id] + (ih - (int)w) / 2; r2 = r1 + (int)w - 1;
    } else {
        c1 = ic2[id] + 1; c2 = ic2[id] + (int)L;
        int ih = ir2[id] - ir1[id] + 1;
        r1 = ir1[id] + (ih - (int)w) / 2; r2 = r1 + (int)w - 1;
    }
}
ll sumCost(int r1, int c1, int r2, int c2) {
    ll s = 0;
    for (int r = r1; r <= r2; r++)
        for (int c = c1; c <= c2; c++)
            s += cellCost(r, c);
    return s;
}
ll ceilDiv_(double x) { return (ll)ceil(x - 1e-9); }

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    R = inf.readInt(); C = inf.readInt(); K = inf.readInt();
    A = inf.readInt(); BD = inf.readInt();

    S.assign(R + 1, "");
    for (int r = 1; r <= R; r++) {
        string row = inf.readToken();
        S[r] = " " + row; // 1-index columns
    }
    Cst.assign(R + 1, "");
    for (int r = 1; r <= R; r++) {
        string row = inf.readToken();
        Cst[r] = " " + row;
    }
    ir1.assign(K + 1, INT_MAX); ir2.assign(K + 1, -1);
    ic1.assign(K + 1, INT_MAX); ic2.assign(K + 1, -1);
    for (int r = 1; r <= R; r++)
        for (int c = 1; c <= C; c++) {
            char ch = S[r][c];
            if (ch >= '1' && ch <= '9') {
                int id = ch - '0';
                if (id > K) quitf(_fail, "input structure grid has island id %d > K=%d", id, K);
                ir1[id] = min(ir1[id], r); ir2[id] = max(ir2[id], r);
                ic1[id] = min(ic1[id], c); ic2[id] = max(ic2[id], c);
            }
        }
    Sreq.assign(K + 1, 0.0);
    for (int i = 1; i <= K; i++) {
        int id = inf.readInt();
        double s = inf.readInt();
        if (id < 1 || id > K) quitf(_fail, "bad island id in Sreq list");
        Sreq[id] = s;
    }

    // ---- read participant bridges ----
    const int MCAP = 400;
    int M = ouf.readInt(0, MCAP, "M");
    vector<vector<char>> used(R + 2, vector<char>(C + 2, 0));
    vector<int> parent(K + 1);
    for (int i = 0; i <= K; i++) parent[i] = i;
    function<int(int)> find = [&](int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    auto unite = [&](int a, int b) { a = find(a); b = find(b); if (a != b) parent[a] = b; };

    vector<double> Krow(K + 1, 0.0), Kcol(K + 1, 0.0);
    ll F = 0;

    for (int b = 0; b < M; b++) {
        int o = ouf.readInt(0, 1, "orient");
        int r1 = ouf.readInt(1, R, "r1");
        int c1 = ouf.readInt(1, C, "c1");
        int r2 = ouf.readInt(1, R, "r2");
        int c2 = ouf.readInt(1, C, "c2");
        if (r1 > r2) quitf(_wa, "bridge %d: r1 > r2", b);
        if (c1 > c2) quitf(_wa, "bridge %d: c1 > c2", b);
        ll h = r2 - r1 + 1, wd = c2 - c1 + 1;
        bool vertical = (o == 0);

        // interior cells must be free background, not overlapping structure or
        // a previous bridge
        for (int r = r1; r <= r2; r++)
            for (int c = c1; c <= c2; c++) {
                if (S[r][c] != '.') quitf(_wa, "bridge %d: cell (%d,%d) is not background", b, r, c);
                if (used[r][c]) quitf(_wa, "bridge %d: cell (%d,%d) overlaps another bridge", b, r, c);
            }

        int frontId, backId; // -1 = frame(0 encoded as -1 not used), we use 0 for frame
        char frontCh = 0, backCh = 0;
        int fr1, fr2, fc1, fc2, br1, br2, bc1, bc2;
        if (vertical) {
            int frow = r1 - 1, brow = r2 + 1;
            if (frow < 1 || brow > R) quitf(_wa, "bridge %d: vertical bridge runs off the grid", b);
            fr1 = fr2 = frow; fc1 = c1; fc2 = c2;
            br1 = br2 = brow; bc1 = c1; bc2 = c2;
        } else {
            int fcol = c1 - 1, bcol = c2 + 1;
            if (fcol < 1 || bcol > C) quitf(_wa, "bridge %d: horizontal bridge runs off the grid", b);
            fr1 = r1; fr2 = r2; fc1 = fc2 = fcol;
            br1 = r1; br2 = r2; bc1 = bc2 = bcol;
        }
        // front face: must be a single uniform structure
        for (int r = fr1; r <= fr2; r++)
            for (int c = fc1; c <= fc2; c++) {
                char ch = S[r][c];
                if (ch == '.') quitf(_wa, "bridge %d: front face touches background at (%d,%d)", b, r, c);
                if (frontCh == 0) frontCh = ch;
                else if (frontCh != ch) quitf(_wa, "bridge %d: front face touches two different structures", b);
            }
        for (int r = br1; r <= br2; r++)
            for (int c = bc1; c <= bc2; c++) {
                char ch = S[r][c];
                if (ch == '.') quitf(_wa, "bridge %d: back face touches background at (%d,%d)", b, r, c);
                if (backCh == 0) backCh = ch;
                else if (backCh != ch) quitf(_wa, "bridge %d: back face touches two different structures", b);
            }
        frontId = (frontCh == '#') ? 0 : (frontCh - '0');
        backId  = (backCh  == '#') ? 0 : (backCh  - '0');
        if (frontId == backId) quitf(_wa, "bridge %d: connects a structure to itself", b);

        for (int r = r1; r <= r2; r++)
            for (int c = c1; c <= c2; c++) used[r][c] = 1;

        double L = vertical ? (double)h : (double)wd;
        double W = vertical ? (double)wd : (double)h;
        double axial = A * W / L;
        double bending = (W * W * W) / ((double)BD * L * L * L);

        if (vertical) {
            if (frontId >= 1) { Krow[frontId] += axial; Kcol[frontId] += bending; }
            if (backId  >= 1) { Krow[backId]  += axial; Kcol[backId]  += bending; }
        } else {
            if (frontId >= 1) { Kcol[frontId] += axial; Krow[frontId] += bending; }
            if (backId  >= 1) { Kcol[backId]  += axial; Krow[backId]  += bending; }
        }
        unite(frontId, backId);
        F += sumCost(r1, c1, r2, c2);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the bridge list");

    for (int id = 1; id <= K; id++)
        if (find(id) != find(0)) quitf(_wa, "island %d is not connected to the frame", id);

    const double EPS = 1e-6;
    for (int id = 1; id <= K; id++) {
        if (Krow[id] < Sreq[id] - EPS)
            quitf(_wa, "island %d row-stiffness %.4f < required %.4f", id, Krow[id], Sreq[id]);
        if (Kcol[id] < Sreq[id] - EPS)
            quitf(_wa, "island %d col-stiffness %.4f < required %.4f", id, Kcol[id], Sreq[id]);
    }

    // ---- internal baseline B: nearest single direction, 1.4x safety margin ----
    ll Btot = 0;
    for (int id = 1; id <= K; id++) {
        int bestDir = UP; ll bestL = gapLen(id, UP);
        for (int d = DOWN; d <= RIGHT; d++) {
            ll g = gapLen(id, d);
            if (g < bestL) { bestL = g; bestDir = d; }
        }
        double Lm = (double)bestL;
        ll wa = ceilDiv_(Sreq[id] * Lm / A);
        ll wb = ceilDiv_(Lm * cbrt(Sreq[id] * (double)BD));
        ll singleW = max((ll)1, max(wa, wb));
        ll wTriv = max((ll)1, ceilDiv_(1.4 * (double)singleW));
        ll capW = maxWidth(id, bestDir);
        if (wTriv > capW) wTriv = capW; // generator guarantees this doesn't bind
        int r1, c1, r2, c2;
        rectFor(id, bestDir, bestL, wTriv, r1, c1, r2, c2);
        Btot += sumCost(r1, c1, r2, c2);
    }
    if (Btot <= 0) Btot = 1;

    double sc = min(1000.0, 100.0 * (double)Btot / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, Btot, sc / 1000.0);
    return 0;
}
