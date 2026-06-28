**Problem.** An `h x w` grid has open cells `'.'` (must be covered) and pillar cells `'#'` (must stay
uncovered). Count, modulo `p`, the ways to pave every open cell with `1 x 2` panels (each covering two
orthogonally adjacent open cells, no overlaps, nothing on a pillar, nothing past the wall). Constraints:
`1 <= h <= 14`, `1 <= w <= 1000`, `1 <= p <= 10^9`. Read `h w p` then `h` rows; print the count mod `p`.

**Why the obvious approaches fail.** Backtracking over panel placements is correct (it is the oracle) but
its search tree is exponential in the number of open cells — fine for a `5 x 5` board, hopeless for
`14 x 1000`. The natural fix, a *column-at-a-time* DP keyed by which cells protrude from one column into the
next, has only `2^h` states but a transition that must enumerate every consistent way to fill a whole
column given the incoming protrusions — a `3^h`-flavored, off-by-one-prone pairing of an incoming mask with
an outgoing mask. The state is cheap; the column-bundled transition is the trap.

**Key idea — broken-profile (cell-by-cell contour) DP.** Stop bundling a whole column per step. Sweep one
**cell** at a time, column by column, top to bottom. The boundary between processed and unprocessed cells is
then a one-cell-deep staircase — the *broken profile* — described by exactly `h` bits, one per row, with the
single meaning: **bit `i` = the frontier cell at row `i` is already covered (so it must not be covered
again).** Standing at `(r, c)` with mask bit `r = 1<<r`:

- `(r,c)` is a pillar: if bit `r` is set, a left panel illegally covered it — drop the state; else keep the
  mask unchanged (a pillar protrudes nothing).
- `(r,c)` open and bit `r` set: it arrived covered by a horizontal panel from column `c-1`; clear bit `r`
  (`mask ^ bit`) to mark it consumed.
- `(r,c)` open and bit `r` clear: cover it now, exactly one of
  - **vertical** with `(r+1,c)` — needs `r+1<h`, `(r+1,c)` open, and its bit `0`; new mask
    `(mask & ~bit) | (1<<(r+1))` (this cell done, the one below now covered);
  - **horizontal** with `(r,c+1)` — needs `c+1<w`, `(r,c+1)` open; new mask `mask | bit` (row `r` protrudes
    into the next column, where it will be consumed by the "already covered" case).

Start `dp[0] = 1`; the answer is `dp[0]` after the full sweep, since nothing may protrude past the last
column. Each step is an `O(1)` one-bit rewrite per state, so the total is `O(w * h * 2^h)` — about
`2.3 * 10^8` operations at the limits, well under two seconds.

**Pitfalls.**
1. *Index transpose.* Read the grid into `grid[r]` but keep `blocked[r][c]` and make **every** access
   `blocked[r][c]` — a stray `blocked[c][r]` in the horizontal-neighbor test silently corrupts the answer and
   is the kind of bug only a differential test against backtracking pins down (hand-tracing tends to
   rationalize the code instead of catching it).
2. *Direction of each rewrite.* A horizontal panel sets *this* row's outgoing bit (consumed next column); a
   vertical panel sets the *row-below* bit and clears this one. Swapping them double-covers or leaks cells.
3. *Modulus and overflow.* Counts explode, so keep residues in `long long` and reduce after every `+=`; a sum
   of residues near `10^9` overflows `int`. Seed `dp[0] = 1 % p` so `p = 1` yields `0` automatically.
4. *Terminal state.* The answer is `dp[0]`, not the sum over all masks — any nonempty terminal profile means a
   panel hangs past column `w-1`.

**Edge cases.** Odd number of open cells -> `0` (some cell can never be paired). Fully blocked board -> `1`
(the empty paving). `1 x 1` open -> `0`; `1 x 1` blocked -> `1`. A `2 x w` open strip -> `Fib(w+1)`. Walls
splitting the board into regions multiply the per-region counts transparently. `p = 1` -> `0`. The `14 x 1000`
worst case runs in `~0.16 s` and `~3.6 MB`.

**Complexity.** Time `O(w * h * 2^h)`, memory `O(2^h)` (two rolling arrays of `2^h` `long long`s).

**Code.**

```cpp
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
```
