#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W;
    if (!(cin >> H >> W)) return 0;          // empty input -> nothing to do

    // Read the grid. Each cell is 0 or 1. We only need the previous row of the
    // DP table at any moment, so we keep two rolling rows of side-lengths.
    // side[j] = largest side length of an all-ones square whose BOTTOM-RIGHT
    // corner is the current cell (i, j).
    vector<int> prev(W + 1, 0), cur(W + 1, 0);
    int best = 0;                            // best side length seen so far

    for (int i = 0; i < H; i++) {
        cur[0] = 0;                          // column 0 sentinel (no left/diag)
        for (int j = 0; j < W; j++) {
            int v;
            cin >> v;
            if (v == 1) {
                // A square ending at (i,j) is limited by the three squares that
                // end just above, just left, and at the up-left diagonal.
                int up   = prev[j + 1];      // square ending at (i-1, j)
                int left = cur[j];           // square ending at (i,   j-1)
                int diag = prev[j];          // square ending at (i-1, j-1)
                int s = min(min(up, left), diag) + 1;
                cur[j + 1] = s;
                if (s > best) best = s;
            } else {
                cur[j + 1] = 0;              // a 0 cell ends no all-ones square
            }
        }
        swap(prev, cur);                     // current row becomes previous row
    }

    // Area is side * side. Use 64-bit: side can be up to 1500, area up to
    // 2.25e6 which fits in 32 bits, but we stay safe.
    long long side = best;
    cout << side * side << "\n";
    return 0;
}
