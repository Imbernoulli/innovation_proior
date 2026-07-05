// TIER: greedy
// Largest-first single-orientation greedy: consider footprints in decreasing size,
// and for each run one row-major stamping pass using only its base orientation.
// Uses the whole catalogue (unlike the baseline) so it covers noticeably more.
#include <bits/stdc++.h>
using namespace std;

int W, H, T;
vector<vector<char>> blk, occ;

static vector<pair<int,int>> norm(vector<pair<int,int>> s) {
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : s) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : s) { p.first -= mr; p.second -= mc; }
    return s;
}

int main() {
    if (scanf("%d %d", &W, &H) != 2) return 0;
    int K; scanf("%d", &K);
    blk.assign(H, vector<char>(W, 0));
    for (int i = 0; i < K; i++) { int r, c; scanf("%d %d", &r, &c); blk[r][c] = 1; }
    scanf("%d", &T);
    vector<vector<pair<int,int>>> shapes(T);
    for (int i = 0; i < T; i++) {
        int sz; scanf("%d", &sz); shapes[i].resize(sz);
        for (int j = 0; j < sz; j++) { int r, c; scanf("%d %d", &r, &c); shapes[i][j] = {r, c}; }
        shapes[i] = norm(shapes[i]);
    }
    // order shape indices by size descending
    vector<int> order(T); iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){ return shapes[a].size() > shapes[b].size(); });

    occ.assign(H, vector<char>(W, 0));
    vector<vector<pair<int,int>>> placed;

    for (int oi : order) {
        auto& s = shapes[oi];
        for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {
            bool fits = true;
            for (auto& p : s) {
                int rr = r + p.first, cc = c + p.second;
                if (rr < 0 || rr >= H || cc < 0 || cc >= W) { fits = false; break; }
                if (blk[rr][cc] || occ[rr][cc]) { fits = false; break; }
            }
            if (!fits) continue;
            vector<pair<int,int>> cells;
            for (auto& p : s) { occ[r+p.first][c+p.second] = 1; cells.push_back({r+p.first, c+p.second}); }
            placed.push_back(cells);
        }
    }

    printf("%d\n", (int)placed.size());
    for (auto& cells : placed) {
        printf("%d", (int)cells.size());
        for (auto& p : cells) printf(" %d %d", p.first, p.second);
        printf("\n");
    }
    return 0;
}
