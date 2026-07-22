#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static const int TS = 16; // fixed "naive" uniform tile side used only for the internal baseline B

// LRU byte-capacity replay. tileBytesFlat[id] is the fixed footprint of tile id (row-major over
// bands, id = r*cB + c). rowBand[i]/colBand[j] map a grid cell to its band index.
static long long simulate(const vector<int>& rowBand, const vector<int>& colBand, int cB,
                           const vector<long long>& tileBytesFlat, long long cacheBytes,
                           const vector<int>& ti, const vector<int>& tj) {
    unordered_map<int, list<int>::iterator> pos;
    list<int> lru;
    long long resident = 0;
    long long F = 0;
    int T = (int)ti.size();
    for (int t = 0; t < T; t++) {
        int id = rowBand[ti[t]] * cB + colBand[tj[t]];
        auto it = pos.find(id);
        if (it != pos.end()) {
            lru.splice(lru.end(), lru, it->second); // move to MRU, iterator stays valid
            continue;
        }
        long long tb = tileBytesFlat[id];
        F += tb;
        if (tb > cacheBytes) continue; // can never be cache-resident alone: pass-through miss every time
        while (resident + tb > cacheBytes && !lru.empty()) {
            int old = lru.front();
            lru.pop_front();
            pos.erase(old);
            resident -= tileBytesFlat[old];
        }
        lru.push_back(id);
        pos[id] = prev(lru.end());
        resident += tb;
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int R = inf.readInt();
    int C = inf.readInt();
    int K = inf.readInt();
    vector<long long> levelBytes(K);
    for (int k = 0; k < K; k++) levelBytes[k] = inf.readLong();
    long long cacheBytes = inf.readLong();

    vector<vector<int>> ent(R, vector<int>(C));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            ent[i][j] = inf.readInt(0, K - 1, "entropy");

    int T = inf.readInt();
    vector<int> ti(T), tj(T);
    for (int t = 0; t < T; t++) {
        ti[t] = inf.readInt(1, R, "trace_i") - 1;
        tj[t] = inf.readInt(1, C, "trace_j") - 1;
    }

    // ---- participant output: row/col cut boundaries + per-tile compression level ----
    auto readBoundaries = [&](int lim, const char* name) -> vector<int> {
        int nb = ouf.readInt(1, lim, name); // number of bands
        vector<int> b(nb + 1);
        b[0] = ouf.readInt(0, 0, "boundary0");
        int prev = 0;
        for (int k = 1; k <= nb; k++) {
            b[k] = ouf.readInt(prev + 1, lim, "boundary");
            prev = b[k];
        }
        if (b[nb] != lim) quitf(_wa, "%s boundaries must end exactly at %d, got %d", name, lim, b[nb]);
        return b;
    };
    vector<int> rb = readBoundaries(R, "rowBoundary");
    vector<int> cb = readBoundaries(C, "colBoundary");
    int rB = (int)rb.size() - 1, cB = (int)cb.size() - 1;

    if ((long long)rB * (long long)cB > 4000000LL)
        quitf(_wa, "too many tiles: %lld", (long long)rB * cB);

    vector<vector<int>> lvl(rB, vector<int>(cB));
    for (int r = 0; r < rB; r++)
        for (int c = 0; c < cB; c++)
            lvl[r][c] = ouf.readInt(0, K - 1, "level");

    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in participant output");

    // row/col -> band index
    vector<int> rowBand(R), colBand(C);
    {
        int r = 0;
        for (int i = 0; i < R; i++) { while (i >= rb[r + 1]) r++; rowBand[i] = r; }
        int c = 0;
        for (int j = 0; j < C; j++) { while (j >= cb[c + 1]) c++; colBand[j] = c; }
    }

    // per-tile entropy cap = min entropy class over the tile's exact cell block (weakest link).
    // Total work across all tiles = R*C exactly once, since bands partition the grid.
    vector<vector<int>> capMax(rB, vector<int>(cB, K - 1));
    for (int i = 0; i < R; i++) {
        int rband = rowBand[i];
        for (int j = 0; j < C; j++) {
            int cband = colBand[j];
            int& cm = capMax[rband][cband];
            if (ent[i][j] < cm) cm = ent[i][j];
        }
    }
    for (int r = 0; r < rB; r++)
        for (int c = 0; c < cB; c++)
            if (lvl[r][c] > capMax[r][c])
                quitf(_wa, "tile (%d,%d) picked level %d but entropy cap allows only %d",
                      r, c, lvl[r][c], capMax[r][c]);

    vector<long long> tileBytes((size_t)rB * cB);
    for (int r = 0; r < rB; r++)
        for (int c = 0; c < cB; c++) {
            long long area = (long long)(rb[r + 1] - rb[r]) * (long long)(cb[c + 1] - cb[c]);
            tileBytes[(size_t)r * cB + c] = area * levelBytes[lvl[r][c]];
        }

    long long F = simulate(rowBand, colBand, cB, tileBytes, cacheBytes, ti, tj);

    // ---- internal baseline B: uniform TS x TS grid, NO compression (level 0 everywhere) ----
    int rB0 = (R + TS - 1) / TS, cB0 = (C + TS - 1) / TS;
    vector<int> rowBand0(R), colBand0(C);
    for (int i = 0; i < R; i++) rowBand0[i] = i / TS;
    for (int j = 0; j < C; j++) colBand0[j] = j / TS;
    vector<long long> tileBytes0((size_t)rB0 * cB0, 0);
    {
        vector<int> rowCnt(rB0, 0), colCnt(cB0, 0);
        for (int i = 0; i < R; i++) rowCnt[rowBand0[i]]++;
        for (int j = 0; j < C; j++) colCnt[colBand0[j]]++;
        for (int r = 0; r < rB0; r++)
            for (int c = 0; c < cB0; c++)
                tileBytes0[(size_t)r * cB0 + c] = (long long)rowCnt[r] * colCnt[c] * levelBytes[0];
    }
    long long B = simulate(rowBand0, colBand0, cB0, tileBytes0, cacheBytes, ti, tj);
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
