// TIER: trivial
// Reproduces the checker's sparse reference baseline exactly: place shape-0 (2x2 square)
// on a coarse lattice with spacing 6 wherever it fits on free tiles. Scores ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int H, W, D;
vector<vector<pair<int,int>>> base;
vector<int> supply;
vector<vector<char>> blocked;
static const int LATTICE = 6;

int main() {
    scanf("%d %d %d", &H, &W, &D);
    base.resize(D); supply.assign(D, 0);
    for (int i = 0; i < D; i++) {
        int A; scanf("%d %d", &A, &supply[i]);
        for (int j = 0; j < A; j++) { int dr, dc; scanf("%d %d", &dr, &dc); base[i].push_back({dr, dc}); }
    }
    int Q; scanf("%d", &Q);
    blocked.assign(H, vector<char>(W, 0));
    for (int j = 0; j < Q; j++) { int r, c; scanf("%d %d", &r, &c); blocked[r][c] = 1; }

    // shape 0, orientation 0 is exactly its base cells (already normalized)
    vector<array<int,4>> placements; // i,o,r,c
    int usedS0 = 0;
    for (int r = 0; r + 1 < H; r += LATTICE) {
        for (int c = 0; c + 1 < W; c += LATTICE) {
            if (usedS0 >= supply[0]) break;
            bool ok = true;
            for (auto& p : base[0]) {
                int rr = r + p.first, cc = c + p.second;
                if (rr < 0 || rr >= H || cc < 0 || cc >= W || blocked[rr][cc]) { ok = false; break; }
            }
            if (ok) { placements.push_back({0, 0, r, c}); usedS0++; }
        }
    }
    printf("%d\n", (int)placements.size());
    for (auto& a : placements) printf("%d %d %d %d\n", a[0], a[1], a[2], a[3]);
    return 0;
}
