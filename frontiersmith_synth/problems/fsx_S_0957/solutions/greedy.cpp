// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The one-pass approach an average coder writes first: scan the grid row-major,
// and at each cell place the SMALLEST-INDEX catalog tile that is locally
// consistent with the already-placed left/top neighbours. No lookahead, no
// global planning. On this tile catalog (every tile's W-edge lives in a
// 2-colour alphabet, its N-edge in a 2-colour alphabet) that local rule is a
// deterministic function on a tiny state space, so it is forced to settle into
// a short repeating cycle almost immediately -- the "easy periodic groove".
int main() {
    int R, W, T;
    scanf("%d %d %d", &R, &W, &T);
    vector<array<int,4>> tiles(T); // N,E,S,W
    for (int i = 0; i < T; i++)
        scanf("%d %d %d %d", &tiles[i][0], &tiles[i][1], &tiles[i][2], &tiles[i][3]);

    vector<vector<int>> grid(R, vector<int>(W, -1));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < W; j++) {
            bool needW = (j > 0);
            bool needN = (i > 0);
            int reqW = needW ? tiles[grid[i][j - 1]][1] : -1;   // left tile's E
            int reqN = needN ? tiles[grid[i - 1][j]][2] : -1;   // top tile's S
            int chosen = -1;
            for (int t = 0; t < T; t++) {
                if (needW && tiles[t][3] != reqW) continue;
                if (needN && tiles[t][0] != reqN) continue;
                chosen = t;
                break; // smallest index that satisfies the local constraint
            }
            // Full 16-tile catalog covers every (N,W) combination, so a match
            // always exists; this fallback only guards a smaller custom catalog.
            if (chosen < 0) chosen = 0;
            grid[i][j] = chosen;
        }
    }

    for (int i = 0; i < R; i++) {
        for (int j = 0; j < W; j++) printf(j ? " %d" : "%d", grid[i][j] + 1);
        printf("\n");
    }
    return 0;
}
