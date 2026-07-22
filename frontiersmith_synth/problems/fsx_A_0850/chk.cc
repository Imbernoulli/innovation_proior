#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> Cell;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Veiled Fourfold Hedge" (hedge-maze-veiled-symmetry).
//
// Input:  "n wSym wTurn wOsc" (n odd; entrance=(0,0), center=((n-1)/2,(n-1)/2)).
// Output: n lines of (n-1) bits (Hwall[r][c], 1=wall between (r,c)-(r,c+1)), then
//         n-1 lines of n bits (Vwall[r][c], 1=wall between (r,c)-(r+1,c)).
//
// Feasibility (unicursal near-Hamiltonian corridor):
//   - every cell's open-edge degree in {0,1,2}
//   - exactly two degree-1 cells, and they must be entrance & center
//   - walking open edges from entrance (never re-entering a visited cell) reaches
//     center after L cells with no branch/cycle
//   - total open edges across the WHOLE grid == L-1 (no stray component elsewhere)
//   - L >= ceil(0.9 * n^2)
// Any violation -> quitf(_wa, ...) -> Ratio 0.
//
// Objective (MAX), all in [0,1], weighted average per input weights:
//   symScore  = fraction of (edge, D4-symmetry) pairs (over the 7 non-identity
//               symmetries) whose image edge has the SAME open/wall state.
//   turnScore = Shannon entropy (base-3, normalized) of the path's per-step
//               Straight/Left/Right turn labels.
//   oscScore  = min(1, (TV - depth(entrance)) / (n^2/5)) where TV is the total
//               variation of Chebyshev-distance-to-center along the path.
// Baseline B = same weighted score for the checker's own plain spiral-matrix
// construction from entrance to center (identical to solutions/trivial.cpp).
// Score: sc = min(1000, 100*F/max(1e-9,B));  Ratio = sc/1000.
// -----------------------------------------------------------------------------

static inline int edgeIdOf(int n, int r1, int c1, int r2, int c2) {
    if (r1 == r2) { int c = min(c1, c2); return r1 * (n - 1) + c; }
    int r = min(r1, r2); return n * (n - 1) + r * n + c1;
}

static inline Cell applyG(int n, int r, int c, int g) {
    switch (g) {
        case 0: return {c, n - 1 - r};
        case 1: return {n - 1 - r, n - 1 - c};
        case 2: return {n - 1 - c, r};
        case 3: return {r, n - 1 - c};
        case 4: return {n - 1 - r, c};
        case 5: return {c, r};
        default: return {n - 1 - c, n - 1 - r};
    }
}

static inline void applyGDir(int dr, int dc, int g, int& odr, int& odc) {
    switch (g) {
        case 0: odr = dc; odc = -dr; break;
        case 1: odr = -dr; odc = -dc; break;
        case 2: odr = -dc; odc = dr; break;
        case 3: odr = dr; odc = -dc; break;
        case 4: odr = -dr; odc = dc; break;
        case 5: odr = dc; odc = dr; break;
        default: odr = -dc; odc = -dr; break;
    }
}
static inline int vec2code(int dr, int dc) {
    if (dr == 1) return 0; if (dr == -1) return 1;
    if (dc == 1) return 2; return 3;
}
static const int CODE_DR[4] = {1, -1, 0, 0};
static const int CODE_DC[4] = {0, 0, 1, -1};

static vector<array<int,7>> buildImageTable(int n) {
    int E = 2 * n * (n - 1);
    vector<array<int,7>> image(E);
    for (int r = 0; r < n; r++)
        for (int c = 0; c + 1 < n; c++) {
            int id = edgeIdOf(n, r, c, r, c + 1);
            for (int g = 0; g < 7; g++) {
                Cell a = applyG(n, r, c, g), b = applyG(n, r, c + 1, g);
                image[id][g] = edgeIdOf(n, a.first, a.second, b.first, b.second);
            }
        }
    for (int r = 0; r + 1 < n; r++)
        for (int c = 0; c < n; c++) {
            int id = edgeIdOf(n, r, c, r + 1, c);
            for (int g = 0; g < 7; g++) {
                Cell a = applyG(n, r, c, g), b = applyG(n, r + 1, c, g);
                image[id][g] = edgeIdOf(n, a.first, a.second, b.first, b.second);
            }
        }
    return image;
}

static vector<Cell> spiralPath(int n) {
    vector<Cell> path; path.reserve((size_t)n * n);
    int top = 0, bottom = n - 1, left = 0, right = n - 1;
    while (top <= bottom && left <= right) {
        for (int c = left; c <= right; c++) path.push_back({top, c});
        top++;
        for (int r = top; r <= bottom; r++) path.push_back({r, right});
        right--;
        if (top <= bottom) {
            for (int c = right; c >= left; c--) path.push_back({bottom, c});
            bottom--;
        }
        if (left <= right) {
            for (int r = bottom; r >= top; r--) path.push_back({r, left});
            left++;
        }
    }
    return path;
}

static inline double oscCap(int n) { return (double)n * (double)n / 5.0; }

static double scoreOf(const vector<Cell>& path, int n, const vector<array<int,7>>& image,
                       ll wSym, ll wTurn, ll wOsc) {
    int L = (int)path.size();
    int E = 2 * n * (n - 1);
    int m = (n - 1) / 2;
    vector<int8_t> dirState(E, -1);
    for (int i = 0; i + 1 < L; i++) {
        int r1 = path[i].first, c1 = path[i].second, r2 = path[i + 1].first, c2 = path[i + 1].second;
        int e = edgeIdOf(n, r1, c1, r2, c2);
        dirState[e] = (int8_t)vec2code(r2 - r1, c2 - c1);
    }
    ll matchCount = 0;
    for (int e = 0; e < E; e++) {
        for (int g = 0; g < 7; g++) {
            int im = image[e][g];
            if (dirState[e] < 0) { if (dirState[im] < 0) matchCount++; continue; }
            int odr, odc; applyGDir(CODE_DR[dirState[e]], CODE_DC[dirState[e]], g, odr, odc);
            int expect = vec2code(odr, odc);
            if (dirState[im] == expect) matchCount++;
        }
    }
    double sym = (double)matchCount / ((double)E * 7.0);

    ll nS = 0, nL = 0, nR = 0;
    for (int i = 1; i + 1 < L; i++) {
        int dr1 = path[i].first - path[i - 1].first, dc1 = path[i].second - path[i - 1].second;
        int dr2 = path[i + 1].first - path[i].first, dc2 = path[i + 1].second - path[i].second;
        if (dr1 == dr2 && dc1 == dc2) nS++;
        else {
            ll cross = (ll)dr1 * dc2 - (ll)dc1 * dr2;
            if (cross > 0) nL++; else nR++;
        }
    }
    ll totalTurns = nS + nL + nR;
    double turn = 0.0;
    if (totalTurns > 0) {
        double H = 0.0;
        ll cnts[3] = {nS, nL, nR};
        for (ll cnt : cnts) if (cnt > 0) {
            double p = (double)cnt / (double)totalTurns;
            H -= p * log2(p);
        }
        turn = H / log2(3.0);
    }

    vector<int> depth(L);
    for (int i = 0; i < L; i++) depth[i] = max(abs(path[i].first - m), abs(path[i].second - m));
    ll TV = 0;
    for (int i = 0; i + 1 < L; i++) TV += llabs((ll)depth[i + 1] - depth[i]);
    ll minTV = depth[0];
    double oscExcess = (double)(TV - minTV);
    double osc = min(1.0, oscExcess / oscCap(n));

    double wsum = (double)(wSym + wTurn + wOsc);
    return ((double)wSym * sym + (double)wTurn * turn + (double)wOsc * osc) / wsum;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(5, 30, "n");
    ll wSym = inf.readLong(1LL, 12LL, "wSym");
    ll wTurn = inf.readLong(1LL, 12LL, "wTurn");
    ll wOsc = inf.readLong(1LL, 12LL, "wOsc");
    if (n % 2 == 0) quitf(_fail, "bad test: n=%d must be odd", n);
    int m = (n - 1) / 2;
    ll E = 2LL * n * (n - 1);

    // ---- read participant output strictly ----
    vector<vector<int>> Hwall(n, vector<int>(n - 1));
    for (int r = 0; r < n; r++)
        for (int c = 0; c + 1 < n; c++)
            Hwall[r][c] = ouf.readInt(0, 1, "Hwall");
    vector<vector<int>> Vwall(n - 1, vector<int>(n));
    for (int r = 0; r + 1 < n; r++)
        for (int c = 0; c < n; c++)
            Vwall[r][c] = ouf.readInt(0, 1, "Vwall");
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the maze");

    // ---- feasibility ----
    vector<vector<int>> degree(n, vector<int>(n, 0));
    auto openH = [&](int r, int c) { return Hwall[r][c] == 0; };   // edge (r,c)-(r,c+1)
    auto openV = [&](int r, int c) { return Vwall[r][c] == 0; };   // edge (r,c)-(r+1,c)
    ll totalOpen = 0;
    for (int r = 0; r < n; r++)
        for (int c = 0; c + 1 < n; c++)
            if (openH(r, c)) { degree[r][c]++; degree[r][c + 1]++; totalOpen++; }
    for (int r = 0; r + 1 < n; r++)
        for (int c = 0; c < n; c++)
            if (openV(r, c)) { degree[r][c]++; degree[r + 1][c]++; totalOpen++; }

    int deg1Count = 0; bool entranceDeg1 = false, centerDeg1 = false;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            if (degree[r][c] < 0 || degree[r][c] > 2)
                quitf(_wa, "cell (%d,%d) has corridor-degree %d > 2 (branch)", r, c, degree[r][c]);
            if (degree[r][c] == 1) {
                deg1Count++;
                if (r == 0 && c == 0) entranceDeg1 = true;
                if (r == m && c == m) centerDeg1 = true;
            }
        }
    if (deg1Count != 2 || !entranceDeg1 || !centerDeg1)
        quitf(_wa, "need exactly two degree-1 cells: entrance(0,0) and center(%d,%d); got %d degree-1 cells (entranceOK=%d centerOK=%d)",
              m, m, deg1Count, (int)entranceDeg1, (int)centerDeg1);

    // walk from entrance following open edges, never revisiting
    vector<vector<int>> visited(n, vector<int>(n, 0));
    vector<Cell> path;
    Cell cur = {0, 0}, prev = {-1, -1};
    while (true) {
        visited[cur.first][cur.second] = 1;
        path.push_back(cur);
        if (cur.first == m && cur.second == m) break;
        // find the open neighbor that is not prev
        Cell nxt = {-1, -1}; int found = 0;
        int r = cur.first, c = cur.second;
        static const int dr[4] = {0, 0, 1, -1}, dc[4] = {1, -1, 0, 0};
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr < 0 || nr >= n || nc < 0 || nc >= n) continue;
            bool isOpen = (dr[d] == 0) ? (dc[d] == 1 ? openH(r, c) : openH(nr, nc))
                                       : (dr[d] == 1 ? openV(r, c) : openV(nr, nc));
            if (!isOpen) continue;
            if (nr == prev.first && nc == prev.second) continue;
            found++; nxt = {nr, nc};
        }
        if (found == 0) quitf(_wa, "corridor from entrance dead-ends at (%d,%d) before reaching center", r, c);
        if (found > 1) quitf(_wa, "cell (%d,%d) branches (should be unreachable given degree checks)", r, c);
        if (visited[nxt.first][nxt.second]) quitf(_wa, "corridor revisits (%d,%d): cycle detected", nxt.first, nxt.second);
        prev = cur; cur = nxt;
        if ((int)path.size() > n * n + 5) quitf(_wa, "corridor walk exceeded n^2 cells: infinite loop");
    }
    ll L = (ll)path.size();
    if (totalOpen != L - 1)
        quitf(_wa, "total open edges (%lld) != path length-1 (%lld): stray corridor/loop elsewhere", totalOpen, L - 1);
    ll need = (ll)ceil(0.9 * (double)n * (double)n);
    if (L < need)
        quitf(_wa, "corridor covers only %lld of %d cells, need >= %lld (90%%)", L, n * n, need);

    // ---- score ----
    auto image = buildImageTable(n);
    double F = scoreOf(path, n, image, wSym, wTurn, wOsc);
    vector<Cell> base = spiralPath(n);
    double B = scoreOf(base, n, image, wSym, wTurn, wOsc);
    if (B <= 0) quitf(_fail, "internal baseline B non-positive (n=%d)", n);

    double sc = min(1000.0, 100.0 * F / B);
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f L=%lld Ratio: %.6f", F, B, L, sc / 1000.0);
    return 0;
}
