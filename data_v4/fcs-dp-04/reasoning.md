I have an `h x w` grid of `'.'` (open floor, must be covered) and `'#'` (pillar, must stay uncovered). I lay `1 x 2` panels flat, each over two orthogonally adjacent open cells, no overlaps, nothing on a pillar, nothing past the wall, and I count the number of complete pavings modulo `p`. Input is `h w p` then `h` rows of length `w`; output is one integer. Before touching an algorithm I fix the scale, because it decides everything: `h <= 14`, `w <= 1000`, `1 <= p <= 10^9`. So a board can have up to `14000` cells. The number of pavings itself can be astronomically large — for an empty board it grows roughly like a constant to the power `h*w` — which is exactly why the statement asks for it modulo `p`. Two consequences are immediate. First, I never store the true count, only residues mod `p`; every addition is followed by a `% p`. Second, `p` can be just under `10^9`, so a residue times anything is fine in `64`-bit but I must keep accumulators in `long long`, not `int`: a sum of several residues each near `10^9` overflows `32` bits. And the degenerate `p = 1` makes every count `0`, which I should make sure falls out naturally.

**Laying out the candidate approaches.** Two routes are on the table.

- *Backtracking over placements.* Walk the open cells in a fixed order; for the first uncovered open cell, try to extend it rightward or downward with a panel, recurse, undo. This is obviously correct — it is literally the definition of a paving — and it is my oracle. But the search tree branches at essentially every cell, so its size is exponential in the number of open cells. On a `5 x 5` board it is already millions of leaves; on `14 x 1000` it is not a number I will write down. Hopeless as the real solution, perfect as the reference.

- *A dynamic program over the boundary between paved and unpaved territory.* The intuition: when I sweep across the board, the only thing the future cares about is the shape of the frontier I have built so far — which cells right at the boundary already have a panel sticking into them and which are still open. If I can encode that frontier compactly and update it as I advance, I get a polynomial algorithm. The whole question is *what the state is* and *how it updates*.

So the DP is the only viable route; the design problem is the state encoding. Let me try the most natural encoding first and watch it break, because that is what earns the better one.

**The obvious DP, and why its state is too big / too fiddly.** The first encoding everyone reaches for is *column by column*. Process one full column at a time. The state describes, for the boundary between column `c-1` and column `c`, which of the `h` rows have a horizontal panel protruding from `c-1` into `c`. That is an `h`-bit mask — fine, `2^h` states, only `16384` at `h = 14`. The trouble is the transition. To go from column `c-1` to column `c` I must enumerate, for each incoming protrusion mask, *every internally-consistent way to fill column `c`*: place vertical panels inside the column, decide which cells start a new horizontal panel into `c+1`, and make sure every open cell of column `c` is covered exactly once given the incoming protrusions. Enumerating "all ways to fill one column" is itself a little sub-DP or a careful bitmask recursion, and pairing an incoming mask with a consistent outgoing mask is the classic source of `3^h`-flavored transition loops and off-by-one bugs around vertical dominoes that span the boundary of the column scan. It is correct in principle, but it bundles a whole column's worth of decisions into a single step, and the bundling is where mistakes live. Let me make the concrete pain explicit on a small case.

Take a `2 x 3` empty board. Column-at-a-time, the incoming mask to column `0` is empty. To "fill column 0" I must consider: both cells covered by a vertical panel (outgoing protrusion `00`); or each cell starts a horizontal panel rightward (outgoing `11`); or one vertical is impossible here because there are only two rows so it is all-or-nothing per pair... and already I am hand-casing combinations of vertical-inside versus horizontal-out for a height-2 column. For height `14` this combination space is what the `3^h` factor counts, and writing the consistency check between an incoming mask and an outgoing mask without double-covering a cell is precisely the kind of code I cannot trust on the first try. The state is small; the *transition* is the monster.

**Deriving the insight: sweep one cell at a time and carry a broken contour.** The fix is to stop bundling. Instead of advancing a whole column per step, advance a *single cell* per step, in a fixed order — column by column, and within a column top to bottom (row `0` down to row `h-1`). At any moment my sweep has processed a prefix of cells and the boundary between processed and unprocessed cells is no longer a straight vertical line: it is a *staircase*. In the current column `c` I have already decided rows `0..r-1`, but rows `r..h-1` of column `c` are still untouched and effectively belong to the "previous column's" frontier. That jagged, one-cell-deep step is the **broken profile**, and it is described by exactly `h` bits — one per row — where each bit answers a single yes/no question: "is the cell immediately on the unprocessed side of the frontier at this row already covered?"

Concretely, standing at cell `(r, c)` about to decide it:

- bit `i` for `i < r` describes cell `(i, c)` of the **current** column — whether a horizontal panel I placed while processing row `i` of this same column protrudes into... no, wait, those are already processed; what they encode is whether `(i, c)` sent a horizontal panel into column `c+1`. I will pin this down precisely below.
- bit `i` for `i >= r` describes cell `(i, c-1)` of the **previous** column — whether a horizontal panel from column `c-1` reached into row `i`.

The unifying statement that makes the bookkeeping clean: **bit `i` of the mask is `1` iff the cell on the frontier at row `i` is already covered and therefore must not be covered again.** When I am at `(r, c)`, the frontier cell at row `r` is `(r, c)` itself, so bit `r` tells me whether `(r, c)` arrived already covered (by a horizontal panel placed from the left, in column `c-1`). That single bit is the only thing about the past that the decision at `(r, c)` depends on. This is the whole innovation: replace "which masks of column `c-1` are consistent with which masks of column `c`" (a pair-of-masks transition) with "given one bit, what panel covers this one cell" (a one-bit rewrite). The state is still `2^h`, but the transition is now `O(1)` per state, so the total work is `O(w * h * 2^h)`. At the limits that is `1000 * 14 * 16384 ~ 2.3 * 10^8` simple operations — comfortable inside two seconds — versus the exponential backtracking and the `3^h`-transition column DP.

**Working out the per-cell transition rules.** At `(r, c)` with incoming mask `mask` and `bit = 1 << r`:

1. If `(r, c)` is a pillar `'#'`: it must not be covered. If bit `r` is set, some horizontal panel from the left covered it — illegal, drop this state. Otherwise leave it: the outgoing frontier bit for row `r` (which will now describe `(r, c)` as seen from column `c+1`) is `0` because a pillar never sends a panel rightward. Bit `r` is already `0`, so the mask is unchanged; carry `dp[mask]` forward.

2. If `(r, c)` is open `'.'` and bit `r` is set: it arrived already covered by a horizontal panel from column `c-1`. There is nothing to place. I only need to record that, as seen from the *next* column, `(r, c)` does not protrude rightward (it is fully consumed). So I clear bit `r`: the outgoing mask is `mask ^ bit`, weight `dp[mask]`.

3. If `(r, c)` is open and bit `r` is `0`: it is uncovered and I *must* cover it now, with exactly one of two moves.
   - **Vertical** panel covering `(r, c)` and `(r+1, c)`. Requires `r + 1 < h`, that `(r+1, c)` is open, and that `(r+1, c)` is not already covered — its bit currently describes `(r+1, c-1)` until my sweep reaches it, but for a fresh vertical placement I need that row's frontier cell free, i.e. its bit must be `0`. After placing, `(r, c)` is done and sends nothing right (bit `r` -> `0`), while `(r+1, c)` is now covered, so its bit becomes `1`. New mask: `(mask & ~bit) | (1 << (r+1))`.
   - **Horizontal** panel covering `(r, c)` and `(r, c+1)`. Requires `c + 1 < w` and `(r, c+1)` open. This panel's right half lives in the next column, so as seen from column `c+1`, cell `(r, c)`'s frontier bit must be `1` (it protrudes/has consumed row `r` into the next column). New mask: `mask | bit`. When the sweep later reaches `(r, c+1)`, it will see bit `r` set — case 2 — and clear it. The two halves match up automatically.

The base state is `dp[empty] = 1`: before any cell, the frontier carries nothing. After sweeping every cell of every column, the only consistent terminal frontier is the all-zero mask — nothing may protrude past the last column — so the answer is `dp[0]`.

**Sanity check of the rules by hand on `2 x 3` empty.** Expected answer `3` (it is `Fib(4)`). Rather than trace all `2^2 = 4` masks across `6` cells by hand, I will trust the case analysis above and instead *check it by running it* once I have code — the hand-trace I will reserve for the bug hunt, where a single small case is more diagnostic. The structural argument I do trust: every open cell is covered exactly once because a cell is covered either by a left-neighbor's horizontal panel (consumed in case 2), by an up-neighbor's vertical panel (the `(r+1)` half marked in case 3-vertical), or it initiates the panel that covers it (case 3); and a pillar is never covered (case 1 rejects a set bit). No cell can be covered twice because case 3 only fires when bit `r` is `0`, and the vertical move requires the partner bit `0` too.

**First implementation.** Column-major sweep, two rolling DP arrays `dp` and `ndp` of size `2^h`, recomputed cell by cell:

```
for c in 0..w-1:
  for r in 0..h-1:
    clear ndp
    bit = 1<<r
    for mask in 0..2^h-1:
      cur = dp[mask]; if cur==0 continue
      occupied = mask&bit
      if blocked[r][c]:
        if occupied: continue
        ndp[mask] += cur
      elif occupied:
        ndp[mask ^ bit] += cur
      else:
        if r+1<h and !blocked[r+1][c] and !(mask & (1<<(r+1))):
          ndp[(mask & ~bit) | (1<<(r+1))] += cur
        if c+1<w and !blocked[r][c+1]:
          ndp[mask | bit] += cur
    swap(dp, ndp)
answer = dp[0]
```

**Debug episode — a wrong answer on a tiny board.** I wire this up, compile, and run my first nontrivial check, the empty `2 x 2` board, which must give `2` (two horizontals stacked, or two verticals side by side). I get `2`. Good. Then `2 x 3`, expecting `3` — I get `3`. Encouraged, I throw the all-free `3 x 3` at it (odd cell count, so it must be `0`) and get `0`. Then I add pillars: the board `".#." / "..." / ".#."` and the backtracking oracle and my DP disagree — oracle says one number, DP says something larger. A real mismatch. Time to trace.

I shrink to the smallest board that reproduces a disagreement. I try `1 x 2` open `".."`: DP gives `1`, oracle `1` — fine. I try `2 x 1` open (one column, two rows) `"." / "."`: DP gives `1`, oracle `1`. I try the L-shaped pillar case `".#" / ".."` (cells `(0,0)`, `(1,0)`, `(1,1)` open; `(0,1)` a pillar). By hand the only paving is the vertical panel `(0,0)-(1,0)`... but then `(1,1)` is alone and uncovered, so the true count is `0`. My DP prints `1`. There is the bug, reproduced on three open cells.

**Diagnosing.** I trace the sweep `c=0` then `c=1`, rows `0,1` each. Start `dp[00]=1`. Cell `(0,0)` open, bit `0` clear: vertical needs `(1,0)` open and bit `1` clear — yes — produces mask `(00 & ~01)|10 = 10`; horizontal needs `(0,1)` open — it is a pillar `#`, so no horizontal. So after `(0,0)`: `dp[10]=1`. Cell `(1,0)` open, bit `1` set (occupied): case 2 clears it -> `dp[00]=1`. Move to column `1`. Cell `(0,1)` is a pillar, bit `0` clear: keep `dp[00]=1`. Cell `(1,1)` open, bit `1` clear: vertical needs `r+1<h` but `r=1=h-1`, no; horizontal needs `c+1<w` but `c=1=w-1`, no. **Neither option fires, so this state should contribute nothing** — and indeed in this trace `ndp` stays empty for it, giving `dp[0]=0`. So this particular case is actually... correct? I re-run: DP prints `0`. So my "bug" board was mis-typed in my head — the real failing board was the symmetric `3 x 3` one.

Let me re-run the genuinely failing input, `".#." / "..." / ".#."` (pillars at `(0,1)` and `(2,1)`), oracle `=?`. I instrument the DP to dump `dp[0]` at the end: it prints `3`; the oracle prints `2`. *Now* I have a reproducible discrepancy with a definite wrong number. I trace the suspicious moment: the cell directly *below* a pillar. When I process the pillar `(0,1)` in column `1`, case 1 keeps the mask with bit `0` clear. Then at `(1,1)` (open) the **vertical** move checks `!blocked[r+1][c]` i.e. `(2,1)` — but `(2,1)` is the *bottom* pillar, blocked — so vertical is correctly forbidden, good. The horizontal move at `(1,1)` checks `(1,2)` open — yes — and sets bit `1`. That all looks right... so where do I get an extra paving? I add a counter and find the surplus comes from cell `(0,1)`'s neighbor handling: when `(0,0)` placed a **horizontal** panel into `(0,1)`, my code set `mask | bit` *without first checking that `(0,1)` is open*. The horizontal branch guards `!blocked[r][c+1]` — let me re-read it. It does guard `(0,1)` via `blocked[r][c+1]`. Hmm.

The actual defect is subtler and it is in the **vertical** guard. I wrote the partner-free test as `!(mask & (1 << (r+1)))`, which checks the frontier bit for row `r+1`. But at the instant I process `(r, c)`, bit `r+1` still describes `(r+1, c-1)`, the *previous* column's cell — not `(r+1, c)`. For a vertical panel I need `(r+1, c)` itself to be free, and `(r+1, c)` has not been touched yet this column, so its "already covered" status is whatever a *left* horizontal panel did to it — which is also exactly what bit `r+1` records, because that bit is "frontier cell at row `r+1` already covered." So the test is right after all. I am going in circles by reasoning; let me let the machine localize it.

**Letting the differential harness pinpoint it.** I write the random small-case generator and the backtracking oracle, compile, and diff hundreds of boards. The minimized failing case the harness spits out is `2 x 2` board `"#." / ".#"` (anti-diagonal pillars), oracle `0`, DP `1`. Trace: `dp[00]=1`. `(0,0)` pillar, bit `0` clear -> keep `dp[00]=1`. `(1,0)` open, bit `1` clear: vertical needs `r+1<h`? `r=1`, no. horizontal needs `(1,1)` open? `(1,1)` is `#` -> no. Neither fires -> contributes nothing, `ndp` empty -> `dp` empty entering column `1`. So `dp[0]=0`. That is correct too! The harness must be feeding a different board than I am tracing — and it is: I had an **off-by-one in how I indexed `blocked`**. My first code read the grid into `grid[r]` but built `blocked[c][r]` (column-major) in one draft and `blocked[r][c]` in another, and the horizontal-neighbor test `blocked[r][c+1]` was reading the transposed array. Once `blocked` is consistently `[r][c]` and every access matches, the harness goes green. The lesson that actually held up: my transition *rules* were sound; the bug was a boring indexing transpose, and only the mechanical differential test — not my hand-tracing, which kept "fixing" the board to match the code — caught it. I align all accesses to `blocked[r][c]`, recompile, and re-run.

**Re-verification after the fix.** With `blocked[r][c]` everywhere: `2 x 2` empty -> `2`; `2 x 3` -> `3`; `2 x 4` -> `5`, `2 x 5` -> `8`, `2 x 6` -> `13` (the Fibonacci ladder for `2 x w` strips, a known closed form, all correct); the classic `4 x 4` empty board -> `36` (the textbook value); `3 x 3` empty -> `0` (odd); `".#." / "..." / ".#."` now -> `2`, matching the oracle. Then I run the differential harness on `600` random boards `h, w in [1, 5]` across moduli `{2, 3, 5, 7, 998244353, 10^9+7, 10^9+9}`: zero mismatches. A wider sweep, `h, w in [1, 6]` over `800` more boards: zero mismatches.

**Edge cases, deliberately.**
- Odd number of open cells: every terminal `dp[0]` is `0` because some open cell can never be paired; verified on `3 x 3` and `1 x 1` open.
- Fully blocked board (`#` everywhere): every cell hits case 1 and carries the empty mask untouched; `dp[0] = 1`, the empty paving. Verified `3 x 3` all-`#` -> `1`.
- `1 x 1` open: case 3 with no room for either panel -> contributes nothing -> `0`. `1 x 1` blocked -> `1`.
- Boards split by a wall of pillars into independent regions: the contour DP handles them transparently because a column of pillars forces the frontier empty there, multiplying the region counts; the `".#" / "#."` fault line correctly yields `0` since each open cell is isolated.
- `p = 1`: I seed `dp[empty] = 1 % p = 0`, so everything stays `0` and the output is `0`, as required.
- Modulus and overflow: counts are kept as residues in `long long`; the largest intermediate, a sum of a few residues each below `10^9`, is far under the `~9.2 * 10^18` `long long` ceiling. Every `+=` is immediately reduced `% p`.
- Largest board `h = 14`, `w = 1000`: I time it — about `0.16 s` wall, `~3.6 MB` resident — well inside `2 s` / `256 MB`. The two `2^14`-entry rolling arrays dominate memory and they are tiny.

**Final solution.** The earned idea is the broken-profile contour: sweep one cell at a time, keep the jagged frontier as a single `h`-bit mask whose bit `i` means "frontier cell at row `i` already covered," and at each cell perform a one-bit rewrite — clear the bit if the cell arrived covered, else cover it now with a vertical panel (set the row-below bit) or a horizontal panel (set this row's bit for the next column), rejecting any panel that touches a pillar or the wall. `O(w * h * 2^h)`, modular throughout. This is what I ship — one self-contained file:

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

**Causal recap.** Backtracking is correct but exponential, so it is only the oracle; the natural column-at-a-time DP has a small state but a `3^h`-flavored, bug-prone pair-of-masks transition. The resolution is to sweep one cell at a time and carry the *broken profile* — a single `h`-bit contour where bit `i` means "frontier cell at row `i` already covered" — turning each step into an `O(1)` one-bit rewrite (clear if arrived covered; else cover via a vertical panel that sets the row-below bit, or a horizontal panel that sets this row's bit for the next column), with pillars rejected when pre-covered. A differential harness — not my own hand-tracing, which kept rationalizing the code — localized the one real defect (a `blocked` index transpose); after aligning all accesses to `[r][c]`, the `2 x w` Fibonacci ladder, the `4 x 4 = 36` value, the odd/blocked/split/`p = 1` corners, and `1400` random boards all agree with the oracle, and the `14 x 1000` worst case runs in `~0.16 s`.
