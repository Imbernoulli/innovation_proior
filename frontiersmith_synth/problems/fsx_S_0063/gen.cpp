#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- polyomino library (relative cells; each shape is connected) ----
    vector<vector<pair<int,int>>> lib = {
        {{0,0}},                                   // monomino  (area 1)
        {{0,0},{1,0}},                             // domino    (area 2)
        {{0,0},{1,0},{2,0}},                       // I-tromino (area 3)
        {{0,0},{1,0},{0,1}},                       // L-tromino (area 3)
        {{0,0},{1,0},{0,1},{1,1}},                 // O-tetromino (area 4)
        {{0,0},{1,0},{2,0},{1,1}},                 // T-tetromino (area 4)
        {{0,0},{0,1},{0,2},{1,0}},                 // L-tetromino (area 4)
        {{0,0},{1,0},{1,1},{2,1}},                 // S-tetromino (area 4)
        {{0,0},{1,0},{2,0},{3,0}},                 // I-tetromino (area 4)
        {{0,0},{1,0},{0,1},{1,1},{0,2}},           // P-pentomino (area 5)
        {{1,0},{0,1},{1,1},{2,1},{1,2}},           // X-pentomino (area 5)
        {{0,0},{0,1},{0,2},{0,3},{1,0}},           // L-pentomino (area 5)
    };
    int L = (int)lib.size();

    // ---- structure ladder: tray grows from tiny (example) to large ----
    int W = 6 + 6 * (testId - 1);       // 6, 12, ..., 60
    int H = W;
    int n = min(L, 2 + testId);         // 3, 4, ..., 12

    // choose n distinct types
    vector<int> idx(L);
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end());    // testlib shuffle uses rnd -> deterministic
    idx.resize(n);
    sort(idx.begin(), idx.end());

    ll board = (ll)W * H;
    printf("%d %d %d\n", W, H, n);

    for (int j = 0; j < n; j++) {
        auto& shp = lib[idx[j]];
        int area = (int)shp.size();
        // stock so that a SINGLE type covers only a fraction of the tray
        // (forces mixing), but total stock across types can exceed the tray.
        ll hi = max(1LL, (ll)floor(0.28 * board / area));
        ll lo = max(1LL, (ll)floor(0.08 * board / area));
        if (lo > hi) lo = hi;
        ll c = rnd.next(lo, hi);
        // higher testIds: skew a subset scarce to reward smart selection
        if (testId >= 5 && (j % 3 == 0)) c = max(1LL, c / 3);
        printf("%d %lld\n", area, c);
        for (auto& p : shp) printf("%d %d\n", p.first, p.second);
    }
    return 0;
}
