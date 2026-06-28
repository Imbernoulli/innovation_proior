#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int h, w;
    long long p;
    if (!(cin >> h >> w >> p)) return 0;

    // Read the grid: h rows, each a string of length w over {'.', '#'}.
    // blocked[r][c] == true means cell (r,c) is a wall (#) and must stay uncovered.
    vector<string> grid(h);
    for (int r = 0; r < h; r++) cin >> grid[r];
    vector<vector<char>> blocked(h, vector<char>(w, 0));
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            blocked[r][c] = (grid[r][c] == '#');

    // Broken-profile DP.
    // We sweep cells in column-major order: c = 0..w-1, and within a column r = 0..h-1.
    // The DP state is an h-bit "profile" mask describing, at the moment we are about to
    // decide cell (r,c), which cells of the frontier are already filled.
    //
    // Concretely, when we stand at cell (r,c):
    //   - bit i for i <  r refers to cell (i, c)   in the CURRENT column,
    //   - bit i for i >= r refers to cell (i, c-1) in the PREVIOUS column.
    // A set bit = that cell is already covered (by a domino placed earlier, or it is a wall
    // we marked filled). The mask thus traces the broken contour between processed and
    // unprocessed cells, which is exactly the only information later placements depend on.
    //
    // dp[mask] = number of ways (mod p) to reach the current frontier with this profile.
    long long P = p;
    int full = 1 << h;
    vector<long long> dp(full, 0), ndp(full, 0);
    dp[0] = 1 % P;  // before processing anything, profile is empty (with p possibly 1)

    for (int c = 0; c < w; c++) {
        for (int r = 0; r < h; r++) {
            fill(ndp.begin(), ndp.end(), 0);
            int bit = 1 << r;
            for (int mask = 0; mask < full; mask++) {
                long long cur = dp[mask];
                if (cur == 0) continue;
                bool occupied = (mask & bit) != 0;  // is (r,c) already covered coming in?

                if (blocked[r][c]) {
                    // A wall must NOT be covered by a domino. If something already covered it
                    // (a horizontal domino reaching in from the left), this state is invalid.
                    if (occupied) continue;
                    // A wall never protrudes right, so its outgoing profile bit is 0. The bit
                    // is already 0 here (not occupied), so the profile is unchanged.
                    ndp[mask] = (ndp[mask] + cur) % P;
                    continue;
                }

                if (occupied) {
                    // Already covered (by a left horizontal domino, decided in column c-1):
                    // just clear the bit to mark this cell processed and uncovered-in-profile.
                    ndp[mask ^ bit] = (ndp[mask ^ bit] + cur) % P;
                } else {
                    // (r,c) is free and not yet covered. We must cover it now, two options:
                    // (1) vertical domino with (r+1,c): needs r+1<h, that cell free and not
                    //     already covered (its bit currently refers to (r+1,c-1) until we pass
                    //     it; but since we haven't reached row r+1 in THIS column yet, its bit
                    //     still describes the previous column's (r+1,c-1)). For a vertical
                    //     placement we require (r+1,c) to be available, i.e. not blocked, and
                    //     its profile bit must be 0 (nothing from the left covers it).
                    if (r + 1 < h && !blocked[r + 1][c] && !(mask & (1 << (r + 1)))) {
                        // Place vertical: cover (r,c) and (r+1,c). In the outgoing profile,
                        // (r,c)'s bit becomes 0 (processed, not protruding) and (r+1,c)'s bit
                        // becomes 1 (now covered, protruding into row r+1's decision).
                        int nm = (mask & ~bit) | (1 << (r + 1));
                        ndp[nm] = (ndp[nm] + cur) % P;
                    }
                    // (2) horizontal domino with (r,c+1): cover (r,c) now; mark (r,c)'s bit set
                    //     in the outgoing profile so that when we process column c+1's row r the
                    //     cell appears already covered. Needs c+1<w and (r,c+1) free.
                    if (c + 1 < w && !blocked[r][c + 1]) {
                        ndp[mask | bit] = (ndp[mask | bit] + cur) % P;
                    }
                }
            }
            swap(dp, ndp);
        }
    }

    // After processing every cell, the only consistent final profile is the empty one:
    // nothing protrudes past the last column.
    cout << (dp[0] % P) << "\n";
    return 0;
}
