// Checker/scorer for fsx_B_0806 "Mirror Hall Laserglyph".
#include "testlib.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <array>
#include <utility>

using namespace std;

static int N;
static string GROUP;

// Directions: 0=N 1=E 2=S 3=W
static const int DR[4] = {-1, 0, 1, 0};
static const int DC[4] = {0, 1, 0, -1};

// '/' swaps N<->E, S<->W  ; '\' swaps N<->W, S<->E
static int reflect(int dir, char mtype) {
    if (mtype == '/') {
        static const int m[4] = {1, 0, 3, 2}; // N->E,E->N,S->W,W->S
        return m[dir];
    } else {
        static const int m[4] = {3, 2, 1, 0}; // N->W,W->N,S->E,E->S
        return m[dir];
    }
}

// Apply group-element `code` to cell (r,c) on an n x n grid.
// codes: 0 identity, 1 rot90, 2 rot180, 3 rot270, 4 flipH, 5 flipV, 6 flipDiag, 7 flipAntiDiag
static pair<int,int> applyCode(int code, int r, int c, int n) {
    switch (code) {
        case 0: return {r, c};
        case 1: return {c, n - 1 - r};
        case 2: return {n - 1 - r, n - 1 - c};
        case 3: return {n - 1 - c, r};
        case 4: return {n - 1 - r, c};
        case 5: return {r, n - 1 - c};
        case 6: return {c, r};
        default: return {n - 1 - c, n - 1 - r};
    }
}

// Simulate the beam starting at (0, ec) heading South, through `mirror` (0 = none,
// else '/' or '\'), and return the illuminated cell mask.
static vector<vector<char>> simulate(int n, int ec, const vector<vector<char>>& mirror) {
    vector<vector<char>> illum(n, vector<char>(n, 0));
    // visited[r][c][dir]
    vector<vector<array<bool,4>>> visited(n, vector<array<bool,4>>(n, {false,false,false,false}));
    int r = 0, c = ec, dir = 2; // South
    long long cap = 6LL * n * n + 50;
    for (long long step = 0; step < cap; step++) {
        if (r < 0 || r >= n || c < 0 || c >= n) break;
        illum[r][c] = 1;
        char mt = mirror[r][c];
        int nd = (mt == '/' || mt == '\\') ? reflect(dir, mt) : dir;
        if (visited[r][c][nd]) break;
        visited[r][c][nd] = true;
        r += DR[nd];
        c += DC[nd];
        dir = nd;
    }
    return illum;
}

struct ScoreParts { long long F; double sym; int motif; };

static ScoreParts scoreOf(int n, const string& group, const vector<vector<char>>& illum,
                           const vector<int>& codes) {
    // orbit id per cell
    vector<vector<int>> orbitId(n, vector<int>(n, -1));
    vector<int> orbitSize;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            if (orbitId[r][c] != -1) continue;
            vector<pair<int,int>> members;
            for (int code : codes) {
                auto p = applyCode(code, r, c, n);
                bool dup = false;
                for (auto& m : members) if (m == p) { dup = true; break; }
                if (!dup) members.push_back(p);
            }
            int id = (int)orbitSize.size();
            for (auto& p : members) orbitId[p.first][p.second] = id;
            orbitSize.push_back((int)members.size());
        }
    }

    vector<int> touchedCount(orbitSize.size(), 0);
    vector<char> touched(orbitSize.size(), 0);
    long long sCount = 0;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            if (illum[r][c]) {
                sCount++;
                int id = orbitId[r][c];
                touchedCount[id]++;
                touched[id] = 1;
            }
        }
    }
    double symSum = 0.0; int nTouched = 0;
    for (size_t id = 0; id < orbitSize.size(); id++) {
        if (touched[id]) {
            symSum += (double)touchedCount[id] / (double)orbitSize[id];
            nTouched++;
        }
    }
    double sym = (nTouched > 0) ? (symSum / nTouched) : 0.0;

    // motif: distinct 3x3 illumination patterns among nonempty windows, capped at 15
    bool seenMask[512] = {false};
    int motif = 0;
    for (int r0 = 0; r0 + 3 <= n; r0++) {
        for (int c0 = 0; c0 + 3 <= n; c0++) {
            int mask = 0, bit = 0; bool any1 = false;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    if (illum[r0+i][c0+j]) { mask |= (1 << bit); any1 = true; }
                    bit++;
                }
            if (any1 && !seenMask[mask]) { seenMask[mask] = true; motif++; }
        }
    }
    int motifCapped = min(motif, 15);
    long long F = (long long)llround(sym * 100.0) * (1 + motifCapped);
    return {F, sym, motifCapped};
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt(8, 40, "n");
    int M = inf.readInt(1, 10000, "M");
    GROUP = inf.readToken();
    if (GROUP != "C2" && GROUP != "D4") quitf(_fail, "bad input: group token");
    int EC = inf.readInt(0, N - 1, "ec");

    vector<int> codes;
    if (GROUP == "C2") codes = {0, 2};
    else codes = {0, 1, 2, 3, 4, 5, 6, 7};

    // ---- read + validate participant output ----
    int K = ouf.readInt(0, M, "K");
    vector<vector<char>> mirror(N, vector<char>(N, 0));
    vector<vector<char>> used(N, vector<char>(N, 0));
    for (int i = 0; i < K; i++) {
        int r = ouf.readInt(0, N - 1, "r");
        int c = ouf.readInt(0, N - 1, "c");
        string t = ouf.readToken();
        if (t != "/" && t != "\\") quitf(_wa, "bad mirror type at mirror %d", i);
        if (used[r][c]) quitf(_wa, "duplicate mirror position (%d,%d)", r, c);
        used[r][c] = 1;
        mirror[r][c] = t[0];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    // ---- score participant ----
    auto illumP = simulate(N, EC, mirror);
    ScoreParts sp = scoreOf(N, GROUP, illumP, codes);
    long long F = sp.F;

    // ---- baseline: no mirrors at all ----
    vector<vector<char>> noMirror(N, vector<char>(N, 0));
    auto illumB = simulate(N, EC, noMirror);
    ScoreParts bp = scoreOf(N, GROUP, illumB, codes);
    long long B = max(1LL, bp.F);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld sym=%.4f motif=%d Ratio: %.6f",
          F, B, sp.sym, sp.motif, sc / 1000.0);
    return 0;
}
