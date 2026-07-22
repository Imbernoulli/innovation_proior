#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Checker / scorer for "Dragon-Forge Radiator Sculpting".  family:
// fin-taper-perimeter-tradeoff
//
// The wall (row 0, columns [BC,BC+BW-1]) is an infinite reservoir fixed at
// T_HOT. The solver's chosen set S of at most M blocks (row in [1,H], column
// in [0,W-1], one 4-connected piece attached to the wall) settles at a steady
// state found by relaxing, for every block i:
//   T_i <- [K_COND * sum_{j in ConductNbr(i)} T_j + K_LOSS*exposed(i)*T_AMB]
//          / [K_COND*|ConductNbr(i)| + K_LOSS*exposed(i)]
// where ConductNbr(i) are i's orthogonal neighbors that are also in S (or the
// wall itself, fixed at T_HOT), and exposed(i) = 4 - |ConductNbr(i)| counts
// the remaining faces radiating into T_AMB (mechanism: exposure-perimeter).
// A one-cell-wide path has only ONE parallel conductive channel per step
// (mechanism: conduction-bottleneck) so its temperature -- and hence what it
// can shed -- decays fast with distance from the wall; a wider cross-section
// carries more flux but "wastes" material once the local flux has already
// decayed (mechanism: shape-taper) -- the winning width profile is set by the
// conduction-to-loss ratio K_COND/K_LOSS, not by raw exposed-face count.
//
// Internal baseline B: the checker relaxes its OWN reference radiator -- the
// single cheapest feasible one, ONE block on the wall's center column -- to
// get a modest, always-positive baseline ("best-single-unit").
// -----------------------------------------------------------------------------

static const int ITER = 4000;

// Relax the steady-state temperatures of `cells` (given as a boolean grid
// `occ`, 1-indexed rows 1..H, 0-indexed columns 0..W-1) and return the total
// heat shed F. `occ` values outside the given cell list must be false.
static double relaxAndScore(const vector<pair<int,int>> &cells, vector<vector<char>> &occ,
                             int H, int W, int BW, int BC, double Thot, double Tamb,
                             double Kcond, double Kloss) {
    if (cells.empty()) return 0.0;
    vector<pair<int,int>> order = cells;
    sort(order.begin(), order.end());
    // dense temperature grid, indexed [row][col], row in [1,H]
    vector<vector<double>> T(H + 1, vector<double>(W, Tamb));

    auto isBaseNbr = [&](int r, int c) {
        return r == 1 && c >= BC && c < BC + BW;
    };

    for (int it = 0; it < ITER; it++) {
        for (auto &pr : order) {
            int r = pr.first, c = pr.second;
            double sumT = 0.0;
            int degCond = 0;
            static const int dr[4] = {-1, 1, 0, 0};
            static const int dc[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr == 0) {
                    if (isBaseNbr(r, c) && dr[d] == -1) { // north face touches wall row
                        sumT += Thot; degCond++;
                    }
                    continue;
                }
                if (nr < 1 || nr > H || nc < 0 || nc >= W) continue;
                if (occ[nr][nc]) { sumT += T[nr][nc]; degCond++; }
            }
            int exposed = 4 - degCond;
            double denom = Kcond * degCond + Kloss * exposed;
            double num = Kcond * sumT + Kloss * exposed * Tamb;
            T[r][c] = num / denom; // degCond >= 1 always (connected), denom > 0
        }
    }

    double F = 0.0;
    for (auto &pr : order) {
        int r = pr.first, c = pr.second;
        int degCond = 0;
        static const int dr[4] = {-1, 1, 0, 0};
        static const int dc[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr == 0) { if (isBaseNbr(r, c) && dr[d] == -1) degCond++; continue; }
            if (nr < 1 || nr > H || nc < 0 || nc >= W) continue;
            if (occ[nr][nc]) degCond++;
        }
        int exposed = 4 - degCond;
        F += Kloss * exposed * (T[r][c] - Tamb);
    }
    return F;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int H = inf.readInt(4, 40, "H");
    int W = inf.readInt(4, 40, "W");
    int M = inf.readInt(1, H * W, "M");
    long long Thot = inf.readLong((long long) 50, (long long) 5000, "Thot");
    long long Tamb = inf.readLong((long long) 0, (long long) 5000, "Tamb");
    if (Thot < Tamb + 50) quitf(_fail, "generator invariant violated: T_HOT must be >= T_AMB+50");
    long long Kcond = inf.readLong((long long) 1, (long long) 60, "Kcond");
    long long Kloss = inf.readLong((long long) 1, (long long) 60, "Kloss");
    int BW = inf.readInt(1, W, "BW");
    int BC = inf.readInt(0, W - BW, "BC");

    // ---- internal baseline: the single cheapest feasible radiator -- ONE block
    // attached to the wall's center column ("best-single-unit"). This is a
    // deliberately unambitious, always-positive reference: it spends almost no
    // material and makes no claim about length, width, or taper at all.
    int cb = BC + BW / 2;
    vector<vector<char>> occB(H + 1, vector<char>(W, 0));
    vector<pair<int,int>> baseCells;
    occB[1][cb] = 1; baseCells.push_back({1, cb});
    double B = relaxAndScore(baseCells, occB, H, W, BW, BC, (double) Thot, (double) Tamb,
                              (double) Kcond, (double) Kloss);
    if (B <= 1e-9) B = 1e-9;

    // ---- read + validate participant output ----
    int K = ouf.readInt(0, M, "K");
    vector<vector<char>> occ(H + 1, vector<char>(W, 0));
    vector<pair<int,int>> cells(K);
    for (int i = 0; i < K; i++) {
        int r = ouf.readInt(1, H, "r");
        int c = ouf.readInt(0, W - 1, "c");
        if (occ[r][c]) quitf(_wa, "block (%d,%d) listed twice", r, c);
        occ[r][c] = 1;
        cells[i] = {r, c};
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after %d blocks", K);

    // ---- connectivity: S must be ONE 4-connected piece, and that single piece
    // must touch the wall. Seed the flood fill from exactly ONE wall-adjacent
    // cell (not every wall-adjacent cell) so two separate blobs that each
    // happen to touch a different wall column, but are not connected to each
    // other through any placed block, are correctly rejected as two pieces.
    if (K > 0) {
        vector<vector<char>> vis(H + 1, vector<char>(W, 0));
        queue<pair<int,int>> q;
        for (int c = BC; c < BC + BW && q.empty(); c++) {
            if (occ[1][c]) { vis[1][c] = 1; q.push({1, c}); }
        }
        int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        while (!q.empty()) {
            auto [r, c] = q.front(); q.pop();
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr < 1 || nr > H || nc < 0 || nc >= W) continue;
                if (occ[nr][nc] && !vis[nr][nc]) { vis[nr][nc] = 1; q.push({nr, nc}); }
            }
        }
        int visited = 0;
        for (int r = 1; r <= H; r++) for (int c = 0; c < W; c++) if (occ[r][c]) visited++;
        int reached = 0;
        for (int r = 1; r <= H; r++) for (int c = 0; c < W; c++) if (vis[r][c]) reached++;
        if (reached != visited) quitf(_wa, "output is not one connected piece attached to the wall");
    }

    // ---- steady state + objective on the participant's shape ----
    double F = relaxAndScore(cells, occ, H, W, BW, BC, (double) Thot, (double) Tamb,
                              (double) Kcond, (double) Kloss);
    if (!isfinite(F)) quitf(_wa, "non-finite objective");

    double sc = min(1000.0, 100.0 * F / B);
    quitp(sc / 1000.0, "OK F=%.4f B=%.4f K=%d Ratio: %.6f", F, B, K, sc / 1000.0);
    return 0;
}
