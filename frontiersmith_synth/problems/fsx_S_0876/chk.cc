#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Flow-Matched Marquetry".
//
// Input: Ws Hs ; Hs x Ws source grain angles ; Wp Hp ; Hp mask lines ; Hp x Wp
//        target flow angles ; K shapes (ncells then ncells "dx dy") ; lambda mu Cu.
//
// Output: M, then M lines "shape_id xs ys rs xp yp rp" -- one physical PIECE
//   INSTANCE per line, cut from the sheet at pose (xs,ys,rs) and glued into the
//   panel at pose (xp,yp,rp). A cell at canonical offset (dx,dy) of the shape
//   lands on the sheet at (xs,ys)+rot(dx,dy,rs) and, for the SAME shape index j,
//   on the panel at (xp,yp)+rot(dx,dy,rp). Its panel-grain angle is the sheet
//   angle at that sheet cell, shifted by 90*(rp-rs) degrees, mod 180.
//
// Feasibility: sheet footprints pairwise disjoint & inside [0,Ws)x[0,Hs);
//   panel footprints pairwise disjoint, inside the '#' mask. Full panel
//   coverage is NOT required (uncovered cells are simply penalized).
//
// Objective (MIN):
//   F = SeamMismatch + TargetFlowMismatch + Cu*UncoveredPanelCells
//       + lambda*UnusedSheetCells - mu*DistinctShapesUsed          (floored at 1)
//   SeamMismatch          = sum over 4-adjacent panel cell pairs covered by TWO
//                            DIFFERENT instances of their angular mismatch (deg, in [0,90]).
//   TargetFlowMismatch    = sum over covered panel cells of |instance angle - desired angle| (mod-180, in [0,90]).
// Baseline B (do-nothing): B = Cu*(#masked panel cells) + lambda*(Ws*Hs).
// Score: sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int rotDx(int dx, int dy, int r){
    switch (r & 3){
        case 0: return dx;
        case 1: return -dy;
        case 2: return -dx;
        default: return dy;
    }
}
int rotDy(int dx, int dy, int r){
    switch (r & 3){
        case 0: return dy;
        case 1: return dx;
        case 2: return -dy;
        default: return -dx;
    }
}
int angDiff(int a, int b){
    int d = abs(a - b) % 180;
    return min(d, 180 - d);
}
int wrap180(long long a){ long long m = a % 180; if (m < 0) m += 180; return (int)m; }

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int Ws = inf.readInt(), Hs = inf.readInt();
    vector<vector<int>> src(Hs, vector<int>(Ws));
    for (int y = 0; y < Hs; y++)
        for (int x = 0; x < Ws; x++)
            src[y][x] = inf.readInt(0, 179);

    int Wp = inf.readInt(), Hp = inf.readInt();
    vector<string> maskRows(Hp);
    for (int y = 0; y < Hp; y++) maskRows[y] = inf.readWord();
    vector<vector<int>> tgt(Hp, vector<int>(Wp));
    ll maskCnt = 0;
    for (int y = 0; y < Hp; y++)
        for (int x = 0; x < Wp; x++){
            tgt[y][x] = inf.readInt(0, 179);
            if (maskRows[y][x] == '#') maskCnt++;
        }

    int K = inf.readInt(1, 100);
    vector<vector<pair<int,int>>> cat(K);
    for (int i = 0; i < K; i++){
        int nc = inf.readInt(1, 16);
        cat[i].resize(nc);
        for (int j = 0; j < nc; j++){
            cat[i][j].first  = inf.readInt(-100, 100);
            cat[i][j].second = inf.readInt(-100, 100);
        }
    }
    ll lambda = inf.readLong();
    ll mu     = inf.readLong();
    ll Cu     = inf.readLong();

    // ---- baseline: do-nothing ----
    ll B = Cu * maskCnt + lambda * (ll)Ws * Hs;
    if (B <= 0) B = 1;

    // ---- read participant output ----
    const int MAXM = 4000;
    int M = ouf.readInt(0, MAXM, "M");

    vector<vector<char>> sheetUsed(Hs, vector<char>(Ws, 0));
    vector<vector<int>>  panelOwner(Hp, vector<int>(Wp, -1));
    vector<vector<int>>  panelAngle(Hp, vector<int>(Wp, -1));
    vector<char> shapeUsed(K, 0);
    ll sheetCovered = 0, panelCovered = 0;

    for (int inst = 0; inst < M; inst++){
        int sid = ouf.readInt(0, K - 1, "shape_id");
        int xs  = ouf.readInt(-1000, 1000, "xs");
        int ys  = ouf.readInt(-1000, 1000, "ys");
        int rs  = ouf.readInt(0, 3, "rs");
        int xp  = ouf.readInt(-1000, 1000, "xp");
        int yp  = ouf.readInt(-1000, 1000, "yp");
        int rp  = ouf.readInt(0, 3, "rp");

        auto &cells = cat[sid];
        int nc = (int)cells.size();
        vector<pair<int,int>> sheetCells(nc), panelCells(nc);
        for (int j = 0; j < nc; j++){
            int dx = cells[j].first, dy = cells[j].second;
            int sx = xs + rotDx(dx, dy, rs), sy = ys + rotDy(dx, dy, rs);
            int px = xp + rotDx(dx, dy, rp), py = yp + rotDy(dx, dy, rp);
            if (sx < 0 || sx >= Ws || sy < 0 || sy >= Hs)
                quitf(_wa, "instance %d: sheet cell (%d,%d) out of bounds", inst, sx, sy);
            if (px < 0 || px >= Wp || py < 0 || py >= Hp)
                quitf(_wa, "instance %d: panel cell (%d,%d) out of bounds", inst, px, py);
            if (maskRows[py][px] != '#')
                quitf(_wa, "instance %d: panel cell (%d,%d) is outside the panel mask", inst, px, py);
            sheetCells[j] = {sx, sy};
            panelCells[j] = {px, py};
        }
        for (int j = 0; j < nc; j++){
            int sx = sheetCells[j].first, sy = sheetCells[j].second;
            if (sheetUsed[sy][sx])
                quitf(_wa, "instance %d: sheet cell (%d,%d) already cut by another instance", inst, sx, sy);
        }
        for (int j = 0; j < nc; j++){
            int px = panelCells[j].first, py = panelCells[j].second;
            if (panelOwner[py][px] != -1)
                quitf(_wa, "instance %d: panel cell (%d,%d) already occupied", inst, px, py);
        }

        int netSteps = ((rp - rs) % 4 + 4) % 4;
        int netDeg = 90 * netSteps;
        for (int j = 0; j < nc; j++){
            int sx = sheetCells[j].first, sy = sheetCells[j].second;
            int px = panelCells[j].first, py = panelCells[j].second;
            sheetUsed[sy][sx] = 1; sheetCovered++;
            int ang = wrap180((long long)src[sy][sx] + netDeg);
            panelOwner[py][px] = inst;
            panelAngle[py][px] = ang;
            panelCovered++;
        }
        shapeUsed[sid] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- compute objective terms ----
    ll seamMismatch = 0;
    for (int y = 0; y < Hp; y++)
        for (int x = 0; x < Wp; x++){
            if (maskRows[y][x] != '#' || panelOwner[y][x] == -1) continue;
            if (x + 1 < Wp && maskRows[y][x+1] == '#' && panelOwner[y][x+1] != -1 &&
                panelOwner[y][x+1] != panelOwner[y][x])
                seamMismatch += angDiff(panelAngle[y][x], panelAngle[y][x+1]);
            if (y + 1 < Hp && maskRows[y+1][x] == '#' && panelOwner[y+1][x] != -1 &&
                panelOwner[y+1][x] != panelOwner[y][x])
                seamMismatch += angDiff(panelAngle[y][x], panelAngle[y+1][x]);
        }

    ll targetMismatch = 0;
    for (int y = 0; y < Hp; y++)
        for (int x = 0; x < Wp; x++)
            if (maskRows[y][x] == '#' && panelOwner[y][x] != -1)
                targetMismatch += angDiff(panelAngle[y][x], tgt[y][x]);

    ll uncoveredPanel = maskCnt - panelCovered;
    ll sheetWaste = (ll)Ws * Hs - sheetCovered;
    ll distinctShapes = 0;
    for (int i = 0; i < K; i++) if (shapeUsed[i]) distinctShapes++;

    ll F = seamMismatch + targetMismatch + Cu * uncoveredPanel + lambda * sheetWaste - mu * distinctShapes;
    if (F < 1) F = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)F);
    quitp(sc / 1000.0, "OK F=%lld B=%lld seam=%lld target=%lld uncov=%lld waste=%lld distinct=%lld Ratio: %.6f",
          F, B, seamMismatch, targetMismatch, uncoveredPanel, sheetWaste, distinctShapes, sc / 1000.0);
    return 0;
}
