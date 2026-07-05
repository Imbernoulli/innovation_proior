// TIER: trivial
// Baseline: stamp only footprint 0 in its given orientation, row-major, wherever it
// fits on solid & unoccupied cells -- identical to the checker's internal baseline B,
// so F == B and the score is exactly 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int W, H;
    if (scanf("%d %d", &W, &H) != 2) return 0;
    int K; scanf("%d", &K);
    vector<vector<char>> blk(H, vector<char>(W, 0));
    for (int i = 0; i < K; i++) { int r, c; scanf("%d %d", &r, &c); blk[r][c] = 1; }
    int T; scanf("%d", &T);
    vector<vector<pair<int,int>>> shapes(T);
    for (int i = 0; i < T; i++) {
        int sz; scanf("%d", &sz); shapes[i].resize(sz);
        for (int j = 0; j < sz; j++) { int r, c; scanf("%d %d", &r, &c); shapes[i][j] = {r, c}; }
    }
    // normalize shape 0
    auto s0 = shapes[0];
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : s0) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : s0) { p.first -= mr; p.second -= mc; }
    int bh = 0, bw = 0;
    for (auto& p : s0) { bh = max(bh, p.first); bw = max(bw, p.second); }
    int SR = bh + 3, SC = bw + 3;   // identical coarse lattice as the checker baseline

    vector<vector<pair<int,int>>> placed;
    for (int r = 0; r < H; r += SR) for (int c = 0; c < W; c += SC) {
        bool fits = true;
        for (auto& p : s0) {
            int rr = r + p.first, cc = c + p.second;
            if (rr < 0 || rr >= H || cc < 0 || cc >= W) { fits = false; break; }
            if (blk[rr][cc]) { fits = false; break; }
        }
        if (!fits) continue;
        vector<pair<int,int>> cells;
        for (auto& p : s0) cells.push_back({r+p.first, c+p.second});
        placed.push_back(cells);
    }

    printf("%d\n", (int)placed.size());
    for (auto& cells : placed) {
        printf("%d", (int)cells.size());
        for (auto& p : cells) printf(" %d %d", p.first, p.second);
        printf("\n");
    }
    return 0;
}
