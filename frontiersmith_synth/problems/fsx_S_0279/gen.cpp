#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: square-ish volcano flank ----
    // testId 1 small (8x8); grows to 26x26 by testId 10.
    int s = 8 + 2 * (testId - 1);          // 8,10,12,...,26
    int H = s, W = s;

    // ---- hazard weight map: base 1 + a few decaying vents (hotspots) ----
    // Skewed weights make WHERE-you-place decisive, not just how much area.
    vector<vector<int>> w(H, vector<int>(W, 1));
    int nHot = 2 + rnd.next(0, 2);         // 2..4 vents
    for (int h = 0; h < nHot; h++) {
        int cr = rnd.next(0, H - 1);
        int cc = rnd.next(0, W - 1);
        int peak = rnd.next(24, 55);
        int decay = rnd.next(4, 8);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++) {
                int d = abs(r - cr) + abs(c - cc);
                int add = peak - decay * d;
                if (add > 0) w[r][c] += add;
            }
    }
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            w[r][c] = min(w[r][c], 60);

    // ---- fixed catalogue of awkward sensor footprints (normalized, min 0) ----
    // Type 1 (T-tetromino) is the baseline module; its 2x3 bbox wastes space so the
    // regular-grid baseline is deliberately mediocre.
    vector<vector<pair<int,int>>> cat = {
        {{0,0},{0,1},{0,2},{1,1}},          // T   a=4  bbox 2x3  (baseline)
        {{0,0},{1,0},{2,0},{2,1}},          // L   a=4  bbox 3x2
        {{0,1},{0,2},{1,0},{1,1}},          // S   a=4  bbox 2x3
        {{0,0},{0,1},{1,1},{1,2}},          // Z   a=4  bbox 2x3
        {{0,1},{1,0},{1,1},{1,2},{2,1}},    // +   a=5  bbox 3x3
        {{0,0},{0,1},{1,0},{1,1},{2,0}},    // P   a=5  bbox 3x2
        {{0,0},{0,1},{0,2},{0,3},{0,4}},    // I5  a=5  bbox 1x5
    };
    int T = (int)cat.size();

    // ---- scarce supply: total frame area far below H*W so packing/supply is binding ----
    // budget ~ 45% of the flank, split across types.
    ll cells = (ll)H * W;
    vector<int> sup(T, 0);
    // type 1 baseline: enough copies to tile ~25% of area on its grid
    int a1 = (int)cat[0].size();
    int bh = 2, bw = 3;
    ll slots = (ll)(H / bh) * (ll)(W / bw);
    ll c1 = (cells / 4) / a1;                       // ~25% area worth of type-1 frames
    c1 = max(1LL, min(c1, slots));
    sup[0] = (int)c1;
    // other types share the remaining ~20% budget
    ll rem = cells / 5;                             // ~20% of flank across T-1 types
    for (int t = 1; t < T; t++) {
        int a = (int)cat[t].size();
        ll per = rem / ((T - 1) * (ll)a);
        int base = (int)max(1LL, per);
        int jitter = rnd.next(0, 1);
        sup[t] = base + jitter;
    }

    // ---- emit ----
    printf("%d %d %d\n", H, W, T);
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) printf("%d%c", w[r][c], c + 1 < W ? ' ' : '\n');
    }
    for (int t = 0; t < T; t++) {
        printf("%d %d\n", (int)cat[t].size(), sup[t]);
        for (auto& p : cat[t]) printf("%d %d\n", p.first, p.second);
    }
    return 0;
}
