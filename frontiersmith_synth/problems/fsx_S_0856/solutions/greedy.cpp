// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The "obvious" first approach: scan the grid row by row, left to right,
// and at each cell try to raise the thread (1) unless that would complete
// a forbidden 2x2 window against the three already-fixed neighbor cells; only
// then fall back to 0. This never looks beyond the single window it is about
// to complete, so it has no notion of the best long-run repeating pattern --
// it just locks in whatever the local picture allows, one cell at a time.
bool forbidden[16];
static inline int patIdx(int w, int x, int y, int z) { return w * 8 + x * 4 + y * 2 + z; }

int main() {
    int R, C, K;
    scanf("%d %d %d", &R, &C, &K);
    for (int i = 0; i < K; i++) {
        int w, x, y, z;
        scanf("%d %d %d %d", &w, &x, &y, &z);
        forbidden[patIdx(w, x, y, z)] = true;
    }

    vector<vector<char>> g(R, vector<char>(C, 0));
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            if (r == 0 || c == 0) {
                g[r][c] = 1; // no window is fully determined yet: greedily raise it
                continue;
            }
            int w = g[r - 1][c - 1], x = g[r - 1][c], y = g[r][c - 1];
            if (!forbidden[patIdx(w, x, y, 1)]) g[r][c] = 1;
            else g[r][c] = 0; // z=0 is never forbidden by this problem's construction
        }
    }

    string row(C, '0');
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) row[c] = g[r][c] ? '1' : '0';
        printf("%s\n", row.c_str());
    }
    return 0;
}
