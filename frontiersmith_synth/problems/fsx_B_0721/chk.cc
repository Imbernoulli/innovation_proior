#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "One Broken Mat: Auspicious Tatami Court Layout".
//
// Input:  H W ; then H lines of W chars ('.' free, '#' pillar). H,W odd, the
//         centre cell (H-1)/2,(W-1)/2 is always free.
// Output: M ; then M lines "T r c" with T in {H,V,S} (0-indexed r,c):
//   H r c : mat covers (r,c)-(r,c+1)      V r c : mat covers (r,c)-(r+1,c)
//   S r c : the ONE required 1x1 mat, covers (r,c) alone
//
// Feasibility: cells in range, free, each free cell covered at most once,
// exactly one S mat total, and no point where FOUR distinct mats' corners
// meet (the tatami rule) -- checked at every interior lattice point whose
// four surrounding cells are all covered.
//
// Objective F (max), weights fixed and stated in the statement:
//   coverage  = covered_free_cells / total_free_cells
//   entropy   = mean, over every 4x4 window containing >=1 H/V mat anchor,
//               of the Shannon entropy (base 2) of that window's H-vs-V mix
//   motif     = (#interior points with EXACTLY 3 distinct mats among the
//               4 covered corners) / (#interior points with all 4 corners
//               covered)
//   symmetry  = among adjacent-cell edges whose own two cells AND their
//               180-degree point-mirror image cells are all covered, the
//               fraction where "is this a seam?" (different mats) agrees
//               with the mirrored edge
//   F = 1000 * (0.18*coverage + 0.37*entropy + 0.10*motif + 0.35*symmetry)
//
// Baseline B: the checker's own even-rows-only reference tiling (same
// construction as solutions/trivial.cpp), scored with the identical formula.
// sc = min(1000, 100*F/max(eps,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static int H, W;
static vector<string> grid;

struct Metrics { double coverage, entropy, motif, symmetry, F; };

static Metrics scoreTiling(const vector<vector<int>>& matId,
                            const vector<vector<char>>& anchorType, // 'H','V', or 0
                            ll freeCells) {
    Metrics m{};
    ll covered = 0;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (matId[r][c] != -1) covered++;
    m.coverage = freeCells > 0 ? (double)covered / (double)freeCells : 0.0;

    // ---- entropy over 4x4 windows via 2D prefix sums ----
    const int WIN = 4;
    vector<vector<int>> PH(H + 1, vector<int>(W + 1, 0)), PV(H + 1, vector<int>(W + 1, 0));
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            int hAdd = (anchorType[r][c] == 'H') ? 1 : 0;
            int vAdd = (anchorType[r][c] == 'V') ? 1 : 0;
            PH[r + 1][c + 1] = PH[r][c + 1] + PH[r + 1][c] - PH[r][c] + hAdd;
            PV[r + 1][c + 1] = PV[r][c + 1] + PV[r + 1][c] - PV[r][c] + vAdd;
        }
    double entSum = 0.0;
    ll entCnt = 0;
    if (H >= WIN && W >= WIN) {
        for (int i0 = 0; i0 + WIN <= H; i0++) {
            for (int j0 = 0; j0 + WIN <= W; j0++) {
                int i1 = i0 + WIN, j1 = j0 + WIN;
                int nH = PH[i1][j1] - PH[i0][j1] - PH[i1][j0] + PH[i0][j0];
                int nV = PV[i1][j1] - PV[i0][j1] - PV[i1][j0] + PV[i0][j0];
                int tot = nH + nV;
                if (tot == 0) continue;
                double pH = (double)nH / tot, pV = (double)nV / tot;
                double e = 0.0;
                if (pH > 0) e -= pH * log2(pH);
                if (pV > 0) e -= pV * log2(pV);
                entSum += e;
                entCnt++;
            }
        }
    }
    m.entropy = entCnt > 0 ? entSum / (double)entCnt : 0.0;

    // ---- motif fraction + junction validity (four-corner rule) ----
    ll totalJ = 0, motif3 = 0, viol = 0;
    for (int i = 1; i < H; i++) {
        for (int j = 1; j < W; j++) {
            int a = matId[i - 1][j - 1], b = matId[i - 1][j];
            int c_ = matId[i][j - 1], d = matId[i][j];
            if (a == -1 || b == -1 || c_ == -1 || d == -1) continue;
            totalJ++;
            int distinct = 1;
            int vals[4] = {a, b, c_, d};
            sort(vals, vals + 4);
            distinct = (int)(unique(vals, vals + 4) - vals);
            if (distinct == 4) viol++;
            else if (distinct == 3) motif3++;
        }
    }
    m.motif = totalJ > 0 ? (double)motif3 / (double)totalJ : 0.0;
    // NOTE: caller is responsible for rejecting `viol>0` outputs before
    // trusting this Metrics struct for the participant's own tiling; the
    // checker's own internal baseline is always violation-free by construction.

    // ---- mirror symmetry over edges (only when all 4 relevant cells covered) ----
    ll totalE = 0, matches = 0;
    auto same = [&](int r1, int c1, int r2, int c2) {
        return matId[r1][c1] == matId[r2][c2];
    };
    for (int r = 0; r < H; r++) {
        for (int c = 0; c + 1 < W; c++) {
            int r2 = r, c2 = c + 1;
            int mr = H - 1 - r, mc = W - 1 - c, mr2 = H - 1 - r2, mc2 = W - 1 - c2;
            if (matId[r][c] == -1 || matId[r2][c2] == -1 ||
                matId[mr][mc] == -1 || matId[mr2][mc2] == -1) continue;
            bool seam1 = !same(r, c, r2, c2);
            bool seam2 = !same(mr, mc, mr2, mc2);
            totalE++;
            if (seam1 == seam2) matches++;
        }
    }
    for (int r = 0; r + 1 < H; r++) {
        for (int c = 0; c < W; c++) {
            int r2 = r + 1, c2 = c;
            int mr = H - 1 - r, mc = W - 1 - c, mr2 = H - 1 - r2, mc2 = W - 1 - c2;
            if (matId[r][c] == -1 || matId[r2][c2] == -1 ||
                matId[mr][mc] == -1 || matId[mr2][mc2] == -1) continue;
            bool seam1 = !same(r, c, r2, c2);
            bool seam2 = !same(mr, mc, mr2, mc2);
            totalE++;
            if (seam1 == seam2) matches++;
        }
    }
    m.symmetry = totalE > 0 ? (double)matches / (double)totalE : 0.0;

    m.F = 1000.0 * (0.18 * m.coverage + 0.37 * m.entropy + 0.10 * m.motif + 0.35 * m.symmetry);
    return m;
}

// checker's own baseline reference: even rows only, fixed-phase pairing
// (0,1),(2,3),...; monomino at the first free cell scanning odd rows, with a
// defensive fallback. Always violation-free by construction (never mixes two
// covered cells with two uncovered ones in the same interior point, because
// entire odd rows are always fully uncovered).
static Metrics baselineMetrics(ll freeCells) {
    vector<vector<int>> matId(H, vector<int>(W, -1));
    vector<vector<char>> anchor(H, vector<char>(W, 0));
    int mid = 0;
    for (int r = 0; r < H; r += 2) {
        for (int c = 0; c + 1 < W; c += 2) {
            if (grid[r][c] == '.' && grid[r][c + 1] == '.') {
                matId[r][c] = mid; matId[r][c + 1] = mid; anchor[r][c] = 'H'; mid++;
            }
        }
    }
    bool placedMono = false;
    for (int r = 1; r < H && !placedMono; r += 2) {
        for (int c = 0; c < W; c++) {
            if (grid[r][c] == '.') { matId[r][c] = mid++; placedMono = true; break; }
        }
    }
    if (!placedMono) {
        for (int r = 0; r < H && !placedMono; r++)
            for (int c = 0; c < W; c++)
                if (grid[r][c] == '.' && matId[r][c] == -1) { matId[r][c] = mid++; placedMono = true; break; }
    }
    if (!placedMono) {
        int cr = (H - 1) / 2, cc = (W - 1) / 2;
        // free the pair covering the centre if needed, then place there
        int old = matId[cr][cc];
        if (old != -1) {
            for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++)
                    if (matId[r][c] == old) { matId[r][c] = -1; anchor[r][c] = 0; }
        }
        matId[cr][cc] = mid++;
    }
    return scoreTiling(matId, anchor, freeCells);
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    H = inf.readInt();
    W = inf.readInt();
    grid.resize(H);
    for (int r = 0; r < H; r++) grid[r] = inf.readToken();
    for (int r = 0; r < H; r++)
        if ((int)grid[r].size() != W) quitf(_fail, "bad input row length");

    ll freeCells = 0;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (grid[r][c] == '.') freeCells++;

    int M = ouf.readInt(1, H * W, "M");
    vector<vector<int>> matId(H, vector<int>(W, -1));
    vector<vector<char>> anchor(H, vector<char>(W, 0));
    int countS = 0;

    for (int k = 0; k < M; k++) {
        string t = ouf.readToken("[HVS]", "type");
        int r = ouf.readInt(0, H - 1, "r");
        int c = ouf.readInt(0, W - 1, "c");
        if (grid[r][c] != '.') quitf(_wa, "mat %d placed on a blocked/pillar cell (%d,%d)", k, r, c);
        if (matId[r][c] != -1) quitf(_wa, "mat %d overlaps an already-covered cell (%d,%d)", k, r, c);
        char ty = t[0];
        if (ty == 'H') {
            if (c + 1 >= W) quitf(_wa, "mat %d horizontal domino out of bounds at (%d,%d)", k, r, c);
            if (grid[r][c + 1] != '.') quitf(_wa, "mat %d covers blocked cell (%d,%d)", k, r, c + 1);
            if (matId[r][c + 1] != -1) quitf(_wa, "mat %d overlaps already-covered cell (%d,%d)", k, r, c + 1);
            matId[r][c] = k; matId[r][c + 1] = k; anchor[r][c] = 'H';
        } else if (ty == 'V') {
            if (r + 1 >= H) quitf(_wa, "mat %d vertical domino out of bounds at (%d,%d)", k, r, c);
            if (grid[r + 1][c] != '.') quitf(_wa, "mat %d covers blocked cell (%d,%d)", k, r + 1, c);
            if (matId[r + 1][c] != -1) quitf(_wa, "mat %d overlaps already-covered cell (%d,%d)", k, r + 1, c);
            matId[r][c] = k; matId[r + 1][c] = k; anchor[r][c] = 'V';
        } else { // 'S'
            matId[r][c] = k;
            countS++;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after %d mats", M);
    if (countS != 1) quitf(_wa, "expected exactly one 1x1 (S) mat, found %d", countS);

    // four-corner (tatami) rule -- reject any real violation outright
    for (int i = 1; i < H; i++) {
        for (int j = 1; j < W; j++) {
            int a = matId[i - 1][j - 1], b = matId[i - 1][j];
            int c_ = matId[i][j - 1], d = matId[i][j];
            if (a == -1 || b == -1 || c_ == -1 || d == -1) continue;
            if (a != b && a != c_ && a != d && b != c_ && b != d && c_ != d)
                quitf(_wa, "four mat corners meet at point (%d,%d)", i, j);
        }
    }

    Metrics part = scoreTiling(matId, anchor, freeCells);
    Metrics base = baselineMetrics(freeCells);
    double B = base.F;
    if (B < 1e-9) B = 1e-9;
    double sc = min(1000.0, 100.0 * part.F / B);

    quitp(sc / 1000.0,
          "OK F=%.4f B=%.4f cov=%.4f ent=%.4f mot=%.4f sym=%.4f Ratio: %.6f",
          part.F, B, part.coverage, part.entropy, part.motif, part.symmetry, sc / 1000.0);
    return 0;
}
