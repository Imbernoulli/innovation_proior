#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: square substation parcel ----
    // testId 1 tiny (example scale ~6x6); grows to ~33x33 by testId 10.
    int s = 6 + 3 * (testId - 1);   // 6, 9, 12, ..., 33
    int H = s, W = s;

    // ---- fixed catalogue of awkward polyomino footprints (normalized min 0) ----
    // Type 1 (T-tetromino) is the baseline module; its bbox wastes space so the
    // regular-grid baseline is deliberately mediocre.
    vector<vector<pair<int,int>>> cat = {
        {{0,0},{0,1},{0,2},{1,1}},          // T  a=4  bbox 2x3
        {{0,0},{1,0},{2,0},{2,1}},          // L  a=4  bbox 3x2
        {{0,1},{0,2},{1,0},{1,1}},          // S  a=4  bbox 2x3
        {{0,0},{0,1},{1,1},{1,2}},          // Z  a=4  bbox 2x3
        {{0,1},{1,0},{1,1},{1,2},{2,1}},    // X (plus) a=5 bbox 3x3
        {{0,0},{0,1},{1,0},{1,1},{2,0}},    // P  a=5  bbox 3x2
        {{0,0},{1,0},{1,1},{2,1},{2,2}},    // W  a=5  bbox 3x3
    };
    int T = (int)cat.size();

    // ---- supplies ----
    vector<int> sup(T, 0);
    int a1 = (int)cat[0].size();            // 4
    int bh = 2, bw = 3;                     // T-tetromino bounding box
    long long slots = (long long)(H / bh) * (long long)(W / bw);
    long long c1 = (3LL * H * W) / (10LL * a1);   // aim baseline ~30% of parcel
    c1 = max(1LL, min(c1, slots));
    sup[0] = (int)c1;
    for (int t = 1; t < T; t++) {
        int a = (int)cat[t].size();
        int base = (H * W) / (2 * a);       // generous: packing (not supply) is binding
        base = max(1, base);
        int jitter = rnd.next(0, base / 3 + 1);
        sup[t] = base + jitter;
    }

    // ---- emit ----
    printf("%d %d %d\n", H, W, T);
    for (int t = 0; t < T; t++) {
        printf("%d %d\n", (int)cat[t].size(), sup[t]);
        for (auto& p : cat[t]) printf("%d %d\n", p.first, p.second);
    }
    return 0;
}
