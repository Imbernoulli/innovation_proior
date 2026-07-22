#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;

// Scoring constants (see statement.txt "Scoring"): the shape (floor + periodicity
// term + motif term) is stated; these exact numbers are the "tunable constants"
// the statement deliberately leaves to be read from behaviour, not memorised.
static const double FLOORC = 10.0;
static const double W1 = 60.0;   // periodicity-penalty weight (dominant mechanism)
static const double W2 = 40.0;   // substitution/motif-diversity weight
static const int MOTIF_CAP = 30; // distinct-4x4-window count normalizer

struct Tile { int N, E, S, W; };

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int R = inf.readInt();
    int W = inf.readInt();
    int T = inf.readInt();
    vector<Tile> tiles(T);
    for (int i = 0; i < T; i++) {
        tiles[i].N = inf.readInt();
        tiles[i].E = inf.readInt();
        tiles[i].S = inf.readInt();
        tiles[i].W = inf.readInt();
    }

    // ---- internal baseline B: repeat the smallest-index self-loop tile everywhere ----
    int loopIdx = -1;
    for (int i = 0; i < T; i++) {
        if (tiles[i].N == tiles[i].S && tiles[i].E == tiles[i].W) { loopIdx = i; break; }
    }
    if (loopIdx < 0) quitf(_fail, "bad instance: no self-loop tile in catalog");
    double motifTermBase = (R >= 4 && W >= 4) ? min(1.0, 1.0 / (double)MOTIF_CAP) : 0.0;
    double Bd = FLOORC + W1 * 0.0 + W2 * motifTermBase;
    if (Bd <= 0) quitf(_fail, "bad instance: B<=0");

    // ---- read & validate participant's tiling (bounded reads only) ----
    vector<vector<int>> grid(R, vector<int>(W));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < W; j++)
            grid[i][j] = ouf.readInt(1, T, "tileIndex") - 1; // store 0-indexed

    for (int i = 0; i < R; i++) {
        for (int j = 0; j < W; j++) {
            const Tile& cur = tiles[grid[i][j]];
            if (j + 1 < W) {
                const Tile& rt = tiles[grid[i][j + 1]];
                if (cur.E != rt.W)
                    quitf(_wa, "horizontal edge mismatch at row %d col %d: E=%d vs W=%d", i, j, cur.E, rt.W);
            }
            if (i + 1 < R) {
                const Tile& dn = tiles[grid[i + 1][j]];
                if (cur.S != dn.N)
                    quitf(_wa, "vertical edge mismatch at row %d col %d: S=%d vs N=%d", i, j, cur.S, dn.N);
            }
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- periodicity penalty: 1 - max over horizontal shifts p=1..W/2 of the
    // fraction of cells (i,j) with grid[i][j] == grid[i][j+p] ----
    int P = W / 2;
    long long best = -1, bestDen = 1;
    for (int p = 1; p <= P; p++) {
        long long matches = 0;
        int cols = W - p;
        for (int i = 0; i < R; i++) {
            const vector<int>& row = grid[i];
            for (int j = 0; j < cols; j++)
                if (row[j] == row[j + p]) matches++;
        }
        long long den = (long long)R * cols;
        if (den > 0 && matches * bestDen > best * den) { best = matches; bestDen = den; }
    }
    double maxRep = (bestDen > 0 && best >= 0) ? (double)best / (double)bestDen : 1.0;
    double periodicityTerm = 1.0 - maxRep;

    // ---- motif diversity: distinct 4x4 windows of tile indices (packed into a
    // 64-bit key, 4 bits/cell since T<=16 fits exactly; falls back to a string
    // key if a future variant ever raises T beyond 16) ----
    int distinctCount = 0;
    if (R >= 4 && W >= 4) {
        int bits = 1;
        while ((1 << bits) < T) bits++;
        if (16 * bits <= 64) {
            unordered_set<u64> seen;
            seen.reserve(4096);
            for (int i = 0; i + 4 <= R; i++) {
                for (int j = 0; j + 4 <= W; j++) {
                    u64 key = 0;
                    for (int di = 0; di < 4; di++)
                        for (int dj = 0; dj < 4; dj++)
                            key = (key << bits) | (u64)grid[i + di][j + dj];
                    seen.insert(key);
                }
            }
            distinctCount = (int)seen.size();
        } else {
            unordered_set<string> seen;
            for (int i = 0; i + 4 <= R; i++) {
                for (int j = 0; j + 4 <= W; j++) {
                    string key;
                    key.reserve(64);
                    for (int di = 0; di < 4; di++)
                        for (int dj = 0; dj < 4; dj++) { key += char('a' + (grid[i + di][j + dj] & 31)); key += char('A' + ((grid[i + di][j + dj] >> 5) & 31)); }
                    seen.insert(key);
                }
            }
            distinctCount = (int)seen.size();
        }
    }
    double motifTerm = min(1.0, (double)distinctCount / (double)MOTIF_CAP);

    double Fd = FLOORC + W1 * periodicityTerm + W2 * motifTerm;

    double sc = min(1000.0, 100.0 * Fd / max(1e-9, Bd));
    quitp(sc / 1000.0, "OK F=%.4f B=%.4f maxRep=%.4f motif=%d Ratio: %.6f", Fd, Bd, maxRep, distinctCount, sc / 1000.0);
    return 0;
}
