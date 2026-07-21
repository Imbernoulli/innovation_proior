An `n x n` grid with values in `[-10^9, 10^9]`, and a falling path that starts anywhere in the top row and from `(i, j)` drops to one of `(i+1, j-1)`, `(i+1, j)`, `(i+1, j+1)` until it reaches the bottom row — one cell per row, `n` cells, minimize the total. The first thing that fixes my data types is the scale: `n` up to 1000 and each cell up to `10^9` in magnitude, so a single path can sum to `n * 10^9 = 10^12`, well past the `~2.1*10^9` a 32-bit `int` holds. Every accumulator and dp cell has to be 64-bit `long long` — an `int` here is a silent wrong answer on the large tests. Input is `n` then the `n*n` values in row-major order on stdin; I print one integer, and `n = 0` (no path) prints `0`.

Two ways to attack it. Greedy descent — start at the cheapest top cell and always step to the cheapest of the three reachable cells below — is `O(n^2)` and a few lines, but a falling path is a *global* object: the column I sit in at row `i` decides which columns I can reach further down, and greedy picks a column from only the next row's three values, blind to what that column fences off later. That is exactly where greedy tends to fail, so I will not trust it without trying to break it. The alternative is a layered DP: for each column keep the minimum cost of a path ending there, and relax each cell from the (at most) three cells above. Same `O(n^2)`, and its correctness is the textbook DAG-relaxation argument — the only real risk is transcription at the edge columns.

Let me actually break greedy rather than argue about it. I want a grid where the locally cheapest early step forces an expensive region later; `n = 3`, uniform top row so the trap is purely in the step decision:

```
5 5 5
1 9 9
50 50 1
```

Greedy starts anywhere, say column 0 (cost 5). Row 1 from column 0 reaches columns 0 (value 1) and 1 (value 9); it grabs the 1. Row 2 from column 0 reaches columns 0 and 1, both 50 — forced. Greedy's path `0 -> 0 -> 0` costs `5 + 1 + 50 = 56`. But the path `0 -> 1 -> 2` takes the worse-looking 9 at row 1 to keep column 2 reachable, landing on the lone `1`: `5 + 9 + 1 = 15`. The cheap `1` at row 1 bought a fence around column 0. Greedy is out.

So I want `dp[i][j]` = the minimum cost of a falling path that *ends* at cell `(i, j)`. The future of a partial path depends only on its current cell, which gives clean optimal substructure. The top row is the base case, `dp[0][j] = a[0][j]` (a one-cell path), and a path ending at `(i, j)` for `i >= 1` arrived from one of `(i-1, j-1)`, `(i-1, j)`, `(i-1, j+1)` inside `[0, n-1]`:

```
dp[i][j] = a[i][j] + min( dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1] )   // skip out-of-range predecessors
```

The answer is `min over j of dp[n-1][j]`. Every dp value is the cost of some real path, and the `min` ranges over every legal predecessor, so it can neither invent an unachievable sum nor miss the optimum — the guarantee greedy lacked. On the given sample the recurrence reproduces the stated `13`: row 0 `[2, 1, 3]`, row 1 `[7, 6, 5]` (e.g. `dp[2] = 4 + min(1, 3) = 5`), row 2 `dp[0] = 7 + min(7, 6) = 13`, so the minimum is `13` via columns `1 -> 2 -> 1`.

The edge columns are where this kind of code dies: columns `0` and `n-1` have only two predecessors. I guard the reads instead of padding with sentinels — start `best = dp[j]` (straight-down, always valid), fold in `dp[j-1]` only when `j > 0` and `dp[j+1]` only when `j+1 < n`. No index escapes `[0, n-1]` and no fabricated infinity can leak into a `min`. For `n = 1` both guards are false and each row's dp is just the running column sum — the only path.

My first instinct for the row update was to overwrite `dp` in place to save a second vector:

```
long long best = dp[j];
if (j > 0)     best = min(best, dp[j - 1]);
if (j + 1 < n) best = min(best, dp[j + 1]);
dp[j] = best + a[i][j];          // overwrite dp[j] in place
```

Overwriting `dp[j]` while a later `j+1` will read `dp[j]` as its left predecessor is dangerous, so I trace the smallest grid that could expose it:

```
2
1 100
100 1
```

The true minimum is `2` (columns `0 -> 1`: `1 + 1`). Trace the in-place loop. Row 0: `dp = [1, 100]`. Row 1, `j = 0`: `best = min(dp[0]=1, dp[1]=100) = 1`, `dp[0] = 1 + 100 = 101`, so `dp = [101, 100]`. Row 1, `j = 1`: `best = min(dp[1]=100, dp[0]) = min(100, 101) = 100` — but `dp[0]` is now `101`, the row-1 value I just wrote, not the row-0 value `1` I needed. Result `dp = [101, 101]`, answer `101`. Wrong.

The defect is exact: computing row `i` in place shreds row `i-1` left to right, so `dp[1]`'s left predecessor becomes the best path ending at `(1, 0)` — a same-row cell, not a legal predecessor of `(1, 1)`. The transition needs the whole previous row intact. Fix: compute row `i` into a fresh `ndp` from the untouched `dp`, then swap. Re-tracing the same grid, `ndp[1] = a[1][1] + min(dp[0], dp[1]) = 1 + 1 = 2` — correct, and it fails-then-passes for exactly the reason I diagnosed. I could hold `O(n)` memory in place with a saved-previous temporary, but the two-vector version is `O(n)` anyway and obviously correct, so I keep it.

A few corners. `n = 0`: print `0` before allocating anything. `n = 1` with a negative single cell, say `-7`: the loop never runs and the answer is that cell — a falling path must cross every row, so there is no "empty path" floor and I do not clamp at 0 (a real distinction from no-adjacent-selection problems). All-negative and uniform grids fall straight out of the same min-over-real-paths construction. Overflow is covered by `long long`, and since I guard rather than pad there is no sentinel to underflow.

For confidence I wrote an independent brute oracle that enumerates every falling path by DFS over all starts and three-way moves — a structurally different program from the DP — plus a generator for tiny/small/all-negative/wide-range/mixed grids kept small enough to enumerate. Differential testing several hundred random cases against the oracle, together with the fixed corners (`n = 0`, single negatives, the greedy counterexample where the DP returns `15`, uniform grids), produced zero mismatches. And the `O(n^2)` DP on a worst-case `n = 1000` grid with values near `10^9` finishes in a fraction of the 2-second limit, on a sum large enough to overflow 32-bit — confirming the `long long` choice was load-bearing, not cosmetic.

So I ship the layered DP I can prove rather than the greedy I broke: read the grid as `long long`, seed `dp` with the top row, sweep each subsequent row into a fresh vector using the two guarded neighbour reads, and print `min` over the final row. The full module is in the answer.
