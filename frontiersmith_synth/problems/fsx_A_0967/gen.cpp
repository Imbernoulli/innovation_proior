#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static const int TS = 16; // must match chk.cc's baseline tile side

struct Rect { int r0, r1, c0, c1; int level; bool hot; }; // [r0,r1) x [c0,c1)

static bool overlaps(const Rect& a, int r0, int r1, int c0, int c1) {
    return !(r1 <= a.r0 || a.r1 <= r0 || c1 <= a.c0 || a.c1 <= c0);
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int R, C, K, T;
    int numPatches;
    int patchMin, patchMax;
    bool latticeTrap = false;   // patches deliberately straddle the TS grid lines
    bool needle = false;        // one tiny extreme-hot patch amid a huge boring background
    double cacheFrac;           // fraction of the "optimal" footprint given as cache budget

    switch (testId) {
        case 1: R = 16; C = 16; K = 3; T = 220; numPatches = 1; patchMin = 4; patchMax = 4; cacheFrac = 2.0; break;
        case 2: R = 48; C = 48; K = 3; T = 3000; numPatches = 3; patchMin = 3; patchMax = 6; cacheFrac = 2.0; break;
        case 3: R = 80; C = 80; K = 4; T = 8000; numPatches = 5; patchMin = 4; patchMax = 8; cacheFrac = 2.5; break;
        case 4: R = 96; C = 96; K = 4; T = 12000; numPatches = 2; patchMin = 10; patchMax = 16; cacheFrac = 3.5; break;
        case 5: R = 128; C = 128; K = 4; T = 20000; numPatches = 36; patchMin = 4; patchMax = 5; cacheFrac = 2.2; latticeTrap = true; break;
        case 6: R = 128; C = 128; K = 4; T = 20000; numPatches = 7; patchMin = 5; patchMax = 5; cacheFrac = 2.5; latticeTrap = true; break;
        case 7: R = 150; C = 150; K = 4; T = 35000; numPatches = 40; patchMin = 4; patchMax = 6; cacheFrac = 2.2; latticeTrap = true; break;
        case 8: R = 150; C = 150; K = 5; T = 40000; numPatches = 1; patchMin = 3; patchMax = 3; cacheFrac = 3.5; latticeTrap = true; needle = true; break;
        case 9: R = 200; C = 200; K = 5; T = 90000; numPatches = 50; patchMin = 4; patchMax = 7; cacheFrac = 2.2; latticeTrap = true; break;
        case 10: R = 240; C = 220; K = 5; T = 160000; numPatches = 70; patchMin = 4; patchMax = 7; cacheFrac = 2.3; latticeTrap = true; needle = true; break;
        default: R = 64; C = 64; K = 3; T = 4000; numPatches = 3; patchMin = 4; patchMax = 6; cacheFrac = 2.2; break;
    }

    // ---- compression level byte table (nonincreasing, level 0 = raw / most expensive).
    // Cumulative compression across the WHOLE table is capped modestly (<=~2.6x) -- the
    // point of this problem is which cells get to use compression at all (entropy caps,
    // cross-region capacity), not a huge per-cell ratio.
    vector<long long> levelBytes(K);
    long long raw = rnd.next(16, 22);
    long long minLevel = max(8LL, raw * 48 / 100); // ~2.0-2.1x cumulative range
    levelBytes[0] = raw;
    for (int k = 1; k < K; k++)
        levelBytes[k] = max(1LL, raw - (raw - minLevel) * k / (K - 1));

    // ---- entropy grid: background = K-1 (max compressible); planted rectangles = low levels ----
    vector<vector<int>> ent(R, vector<int>(C, K - 1));
    vector<Rect> patches;

    auto tryPlace = [&](int r0, int r1, int c0, int c1, int level, bool hot) -> bool {
        r0 = max(0, r0); c0 = max(0, c0); r1 = min(R, r1); c1 = min(C, c1);
        if (r1 - r0 < 2 || c1 - c0 < 2) return false;
        for (auto& p : patches) if (overlaps(p, r0, r1, c0, c1)) return false;
        for (int i = r0; i < r1; i++) for (int j = c0; j < c1; j++) ent[i][j] = level;
        patches.push_back({r0, r1, c0, c1, level, hot});
        return true;
    };

    if (latticeTrap) {
        // Place small low-entropy patches CENTERED on the naive TS-grid intersections so the
        // fixed-uniform-tile reflex is forced to cap almost every neighboring tile to level 0.
        vector<pair<int,int>> corners;
        for (int i = TS; i < R; i += TS)
            for (int j = TS; j < C; j += TS)
                corners.push_back({i, j});
        shuffle(corners.begin(), corners.end(), std::mt19937(1000 + testId));
        int placed = 0;
        for (auto& pc : corners) {
            if (placed >= numPatches) break;
            int sz = rnd.next(patchMin, patchMax);
            int r0 = pc.first - sz / 2, c0 = pc.second - sz / 2;
            int level = (rnd.next(0, 3) == 0) ? 0 : rnd.next(0, min(1, K - 2)); // mostly incompressible
            if (tryPlace(r0, r0 + sz, c0, c0 + sz, level, true)) placed++;
        }
        // top up with a few random (non-lattice) patches for texture
        int extra = max(0, numPatches - placed) / 3;
        for (int t = 0; t < extra; t++) {
            int sz = rnd.next(patchMin, patchMax + 2);
            int r0 = rnd.next(0, max(0, R - sz)), c0 = rnd.next(0, max(0, C - sz));
            tryPlace(r0, r0 + sz, c0, c0 + sz, rnd.next(0, K - 2), false);
        }
    } else {
        for (int t = 0; t < numPatches; t++) {
            int sz = rnd.next(patchMin, patchMax);
            int r0 = rnd.next(0, max(0, R - sz)), c0 = rnd.next(0, max(0, C - sz));
            tryPlace(r0, r0 + sz, c0, c0 + sz, rnd.next(0, K - 2), true);
        }
    }

    Rect needlePatch{-1, -1, -1, -1, 0, false};
    if (needle) {
        // one tiny, maximally incompressible patch placed off any existing patch, hammered hard
        for (int attempt = 0; attempt < 200; attempt++) {
            int sz = 3;
            int r0 = rnd.next(TS / 2, R - TS), c0 = rnd.next(TS / 2, C - TS);
            // bias it to straddle a TS boundary: land it at q*TS + (TS-2) so its 3 cells
            // are [q*TS+TS-2, q*TS+TS+1) -- 2 cells in tile q, 1 cell in tile q+1, actually
            // crossing the fixed-16 baseline/greedy grid line (not just centered inside one tile).
            r0 -= r0 % TS; r0 += TS - 2;
            c0 -= c0 % TS; c0 += TS - 2;
            if (tryPlace(r0, r0 + sz, c0, c0 + sz, 0, true)) {
                needlePatch = patches.back();
                break;
            }
        }
    }

    // ---- fixed access trace: interleaved row sweeps, diagonal walks, and hot-patch hammers ----
    vector<int> ti_, tj_;
    ti_.reserve(T); tj_.reserve(T);
    vector<Rect> hotList;
    for (auto& p : patches) if (p.hot) hotList.push_back(p);
    if (needle && needlePatch.r0 >= 0) hotList.push_back(needlePatch);
    if (hotList.empty()) hotList.push_back({0, min(R,4), 0, min(C,4), 0, true});

    int remaining = T;
    int segIdx = 0;
    while (remaining > 0) {
        int kind = segIdx % 3;
        segIdx++;
        if (kind == 0) {
            // full row sweep across a band of rows
            int rows = min(R, 1 + rnd.next(0, 3));
            int r0 = rnd.next(0, max(0, R - rows));
            int len = min(remaining, rows * C);
            for (int k = 0; k < len; k++) {
                int i = r0 + (k / C) % rows, j = k % C;
                ti_.push_back(i + 1); tj_.push_back(j + 1);
            }
            remaining -= len;
        } else if (kind == 1) {
            // diagonal walk, bouncing at edges
            int len = min(remaining, 1 + rnd.next(0, min(R, C)));
            int i = rnd.next(0, R - 1), j = rnd.next(0, C - 1);
            int di = 1, dj = 1;
            for (int k = 0; k < len; k++) {
                ti_.push_back(i + 1); tj_.push_back(j + 1);
                i += di; j += dj;
                if (i < 0 || i >= R) { di = -di; i += 2 * di; }
                if (j < 0 || j >= C) { dj = -dj; j += 2 * dj; }
                i = max(0, min(R - 1, i)); j = max(0, min(C - 1, j));
            }
            remaining -= len;
        } else {
            // hammer a hot patch: repeated near-random accesses inside a randomly chosen hot rect
            const Rect& p = hotList[rnd.next(0, (int)hotList.size() - 1)];
            int len = min(remaining, 20 + rnd.next(0, 200));
            for (int k = 0; k < len; k++) {
                int i = rnd.next(p.r0, p.r1 - 1), j = rnd.next(p.c0, p.c1 - 1);
                ti_.push_back(i + 1); tj_.push_back(j + 1);
            }
            remaining -= len;
        }
    }
    T = (int)ti_.size();

    // ---- pick cacheBytes: a small multiple of ONE naive-baseline (TS x TS, uncompressed) tile.
    // This is deliberately scale-independent (not a fraction of the whole grid's footprint):
    // it keeps the baseline thrashing hard (only a couple of its raw tiles fit at once) while
    // still forcing a well-compressed, well-aligned layout to face genuine LRU eviction
    // pressure instead of trivially holding its entire footprint resident forever.
    long long rawTileUnit = (long long)TS * TS * levelBytes[0];
    long long cacheBytes = max(64LL, (long long)(rawTileUnit * cacheFrac));

    // ---- emit ----
    printf("%d %d %d\n", R, C, K);
    for (int k = 0; k < K; k++) printf("%lld ", levelBytes[k]);
    printf("\n%lld\n", cacheBytes);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) printf("%d ", ent[i][j]);
        printf("\n");
    }
    printf("%d\n", T);
    for (int t = 0; t < T; t++) printf("%d %d\n", ti_[t], tj_[t]);

    return 0;
}
