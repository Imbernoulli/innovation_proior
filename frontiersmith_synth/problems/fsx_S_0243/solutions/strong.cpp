// TIER: strong
// Orientation-aware multi-pass packing: place footprints largest-first over ALL
// rotations/reflections, then repeat with smaller shapes to fill leftover pockets.
// Covers substantially more of the solid region than the single-orientation greedy.
#include <bits/stdc++.h>
using namespace std;

int W, H, T;
vector<vector<char>> blk, occ;

static vector<pair<int,int>> norm(vector<pair<int,int>> s) {
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : s) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : s) { p.first -= mr; p.second -= mc; }
    sort(s.begin(), s.end());
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
    }

    // all distinct orientations per shape
    vector<vector<vector<pair<int,int>>>> orients(T);
    vector<int> szOf(T);
    for (int i = 0; i < T; i++) {
        szOf[i] = (int)shapes[i].size();
        set<vector<pair<int,int>>> uniq;
        for (int refl = 0; refl < 2; refl++) for (int rot = 0; rot < 4; rot++) {
            vector<pair<int,int>> cur;
            for (auto& p : shapes[i]) {
                int r = p.first, c = p.second;
                if (refl) c = -c;
                for (int t = 0; t < rot; t++) { int nr = c, nc = -r; r = nr; c = nc; }
                cur.push_back({r, c});
            }
            uniq.insert(norm(cur));
        }
        orients[i].assign(uniq.begin(), uniq.end());
    }

    // shapes ordered by size descending
    vector<int> order(T); iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){ return szOf[a] > szOf[b]; });

    occ.assign(H, vector<char>(W, 0));
    vector<vector<pair<int,int>>> placed;

    auto tryStamp = [&](const vector<pair<int,int>>& s) -> bool {
        bool any = false;
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
            any = true;
        }
        return any;
    };

    // multiple rounds: each round stamps large->small over every orientation;
    // repeat while progress is made so freshly-shaped gaps get filled.
    for (int round = 0; round < 4; round++) {
        bool progress = false;
        for (int oi : order)
            for (auto& s : orients[oi])
                if (tryStamp(s)) progress = true;
        if (!progress) break;
    }

    printf("%d\n", (int)placed.size());
    for (auto& cells : placed) {
        printf("%d", (int)cells.size());
        for (auto& p : cells) printf(" %d %d", p.first, p.second);
        printf("\n");
    }
    return 0;
}
