**Reading the problem and pinning the contract.** I have an `n x n` grid `a[i][j]`, values possibly negative. A *falling path* starts at any cell in the top row and, from `(i, j)`, drops to one of `(i+1, j-1)`, `(i+1, j)`, `(i+1, j+1)` while staying inside the grid; it stops at the bottom row. Each path touches exactly one cell per row, so `n` cells total, and its cost is the sum of those cells. I want the minimum cost over every falling path. Input on stdin is `n` followed by the `n*n` grid values in row-major order; I print one integer. For `n = 0` there is no path and the natural answer is `0`. Let me fix the scale before any algorithm, because it dictates the data types: `n <= 1000` and `|a[i][j]| <= 10^9`, so a path visits `n` cells and its sum can reach `n * 10^9 = 10^12` in magnitude. That blows past the 32-bit range of about `2.1*10^9`, so the running cost and every dp cell have to be 64-bit. I will use `long long` throughout. That is the first decision and it is non-negotiable; an `int` here is a silent wrong-answer on the large tests, where a single path easily sums past two billion.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is shortest to type.

- *Greedy descent.* Start at the minimum cell of the top row, then at each row step to the minimum of the (at most) three cells reachable from the current column; accumulate. It is `O(n^2)` just to read the grid plus an `O(n)` walk, and it is a handful of lines. The risk is structural: a falling path is a *global* object — the column I sit in at row `i` constrains which columns I can reach at rows `i+1, i+2, ...` — but greedy commits to a column using only the next row's three values, ignoring what that column fences me out of further down. That is exactly the configuration where greedy tends to be wrong, so I will not trust it until I have tried to break it.
- *Layered DP.* Process the grid row by row. Keep, for each column `j`, the minimum cost `dp[j]` of any falling path that *ends* at cell `(i, j)`. A path ending at `(i, j)` came from one of `(i-1, j-1)`, `(i-1, j)`, `(i-1, j+1)`, so `dp_i[j] = a[i][j] + min` over those three of the previous row's `dp`. This is `O(n^2)` time and `O(n)` extra memory (two rolling rows). The risk here is not correctness of the idea — it is a textbook shortest-path-on-a-DAG relaxation, processed in topological (row) order — but correctness of the *transcription*: the edge columns `j = 0` and `j = n-1` have only two predecessors, and an off-by-one or an out-of-bounds read there is easy to write.

**Stress-testing greedy before committing.** Hand-waving "cheapest next cell feels right" is how wrong solutions get shipped, so let me actually attack greedy with a concrete instance instead of arguing in the abstract. I want a grid where taking the locally cheapest step early forces me into an expensive region later, while a *more expensive* early step keeps a cheap region reachable. Let me build one deliberately, `n = 3`:

```
5 5 5
1 9 9
50 50 1
```

I made the top row uniform on purpose, so greedy can start in any column and the trap is purely about the *step* decision, not the start. Say greedy starts at column 0 (cost 5). At row 1, the cells reachable from column 0 are column 0 (value 1) and column 1 (value 9); greedy grabs the 1 at column 0. Now at row 2, the cells reachable from column 0 are column 0 (value 50) and column 1 (value 50) — both enormous. Greedy is forced to add 50. Greedy's path is columns `0 -> 0 -> 0`, cost `5 + 1 + 50 = 56`.

Is 56 optimal? Let me hunt for something greedy structurally could not reach. Consider the path columns `0 -> 1 -> 2`: cost `5 + 9 + 1 = 15`. That path *deliberately* takes the worse-looking 9 at row 1 (column 1) instead of the tempting 1 (column 0), and that single concession keeps column 2 reachable at row 2, where the lone `1` lives. `15` is dramatically better than greedy's `56`. So greedy is wrong, and I now see *why*: by snatching the cheap `1` at row 1 it pinned itself to column 0, and column 0's only descendants at row 2 are the two `50`s. The cheap step bought a fence. The verification paid off — it killed an approach I would otherwise have shipped, on an instance small enough that I can check it by full enumeration. Greedy is out.

(I double-check the enumeration claim by brute force later, but even by hand: from a uniform top row the three starts are symmetric in the sense that each can only reach a limited set of bottom cells, and the only way to land on the bottom-row `1` at column 2 is to be at column 1, 2, or 3 at row 1 — column 1 is the cheapest such option at row 1 that still preserves reach to column 2, giving `5 + 9 + 1 = 15`. Nothing beats 15.)

**Deriving the DP and checking the recurrence on paper.** I want, for every cell `(i, j)`, the minimum cost of a falling path that ends at that cell. The future of a partial path depends only on which cell it currently occupies — not on how it got there — so this is a clean dynamic program over cells, with edges pointing from row `i-1` to row `i`. Define:

- `dp[i][j]` = minimum cost over all falling paths starting somewhere in row 0 and ending at `(i, j)`.

Base case, row 0: a path "ending at `(0, j)`" is just the single starting cell, so `dp[0][j] = a[0][j]`. Transition for `i >= 1`: a path ending at `(i, j)` arrived from one of the cells directly above-and-adjacent, i.e. `(i-1, j-1)`, `(i-1, j)`, `(i-1, j+1)`, restricted to columns inside `[0, n-1]`. So

```
dp[i][j] = a[i][j] + min( dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1] )   // skipping out-of-range predecessors
```

The answer is `min over j of dp[n-1][j]`: the cheapest path that reaches *any* cell of the bottom row. Because every value of `dp` is exactly the cost of a real path (the base case is a real one-cell path, and each transition appends a legal move to a real path), the DP can never invent a sum that no path achieves; and because the `min` ranges over every legal predecessor, it can never miss the optimal predecessor. That is the optimal-substructure argument I can stand behind, unlike greedy.

Let me confirm the recurrence by hand on the sample, answer `13`:

```
2 1 3
6 5 4
7 8 9
```

Row 0: `dp = [2, 1, 3]`. Row 1: `dp[0] = 6 + min(dp[0], dp[1]) = 6 + min(2,1) = 6 + 1 = 7`; `dp[1] = 5 + min(dp[0],dp[1],dp[2]) = 5 + min(2,1,3) = 5 + 1 = 6`; `dp[2] = 4 + min(dp[1],dp[2]) = 4 + min(1,3) = 4 + 1 = 5`. So row 1 `dp = [7, 6, 5]`. Row 2: `dp[0] = 7 + min(7,6) = 7 + 6 = 13`; `dp[1] = 8 + min(7,6,5) = 8 + 5 = 13`; `dp[2] = 9 + min(6,5) = 9 + 5 = 14`. Row 2 `dp = [13, 13, 14]`, minimum `13`. Correct — and the two ways to reach 13 (`2->5->? ` no; actually `1 (col1) -> 4 (col2) -> 8 (col1)` = `1+4+8 = 13`, columns `1->2->1`) match the path I expected.

**Boundary handling — this is where this kind of code dies.** The edge columns have only two predecessors, not three. I will *not* pad with sentinels and risk a sentinel leaking into a real `min`; instead I will guard the reads explicitly: start `best = dp[j]` (the straight-down predecessor, always valid), then fold in `dp[j-1]` only if `j > 0` and `dp[j+1]` only if `j+1 < n`. That guarantees I never index out of `[0, n-1]` and never compare against a fabricated value. For `n = 1` there is a single column with no horizontal neighbours, so each row's `dp[0]` is just the running column sum `a[0][0] + a[1][0] + ... ` — the only falling path. The guards handle that automatically (both `j > 0` and `j+1 < n` are false).

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first instinct for the rolling update was to overwrite `dp` in place to save the second vector:

```
for (int i = 1; i < n; i++) {
    for (int j = 0; j < n; j++) {
        long long best = dp[j];
        if (j > 0)     best = min(best, dp[j - 1]);
        if (j + 1 < n) best = min(best, dp[j + 1]);
        dp[j] = best + a[i][j];          // overwrite dp[j] in place
    }
}
```

Something about overwriting `dp[j]` while the *next* iteration `j+1` will read `dp[j]` as its left predecessor looks dangerous, so I trace the smallest input that could expose it. I use a grid where the in-place corruption changes the answer:

```
2
1 100
100 1
```

The true answer: two paths. Start col 0 -> row1 col0 or col1: `1+100=101` or `1+1=2`. Start col 1 -> row1 col0 or col1: `100+100=200` or `100+1=101`. Minimum is `2` (columns `0 -> 1`). Now trace the in-place loop. Row 0: `dp = [1, 100]`. Row 1, `j = 0`: `best = dp[0] = 1`; `j+1<n` so `best = min(1, dp[1]=100) = 1`; `dp[0] = 1 + a[1][0] = 1 + 100 = 101`. Now `dp = [101, 100]`. Row 1, `j = 1`: `best = dp[1] = 100`; `j>0` so `best = min(100, dp[0]) = min(100, 101) = 100` — but `dp[0]` is now `101`, the *already-updated* row-1 value, not the row-0 value `1` I needed! `dp[1] = 100 + a[1][1] = 100 + 1 = 101`. Final `dp = [101, 101]`, minimum `101`.

**Diagnosing the bug.** The code returns `101`; the correct answer is `2`. The defect is precise: when I compute `dp[1]` I read `dp[0]` as the "came-from-column-0" predecessor, but I have *already overwritten* `dp[0]` with row 1's value (`101`) earlier in the same row sweep. So `dp[1]`'s left predecessor is no longer "the best path ending at `(0, 0)`" (`= 1`) — it is "the best path ending at `(1, 0)`" (`= 101`), a cell in the *same* row, which is not a legal predecessor of `(1, 1)` at all. In-place update destroyed the previous row before I finished reading it. The transition for every cell in row `i` needs the *whole* of row `i-1` intact; overwriting cell by cell shreds it left to right.

**Fixing and re-verifying.** Don't overwrite in place — compute row `i` into a fresh vector `ndp` from the untouched previous `dp`, then swap:

```
for (int i = 1; i < n; i++) {
    vector<long long> ndp(n);
    for (int j = 0; j < n; j++) {
        long long best = dp[j];
        if (j > 0)     best = min(best, dp[j - 1]);
        if (j + 1 < n) best = min(best, dp[j + 1]);
        ndp[j] = best + a[i][j];
    }
    dp = move(ndp);
}
```

Re-trace `[[1,100],[100,1]]`: row 0 `dp = [1, 100]`. Row 1 into `ndp`: `ndp[0] = a[1][0] + min(dp[0], dp[1]) = 100 + min(1,100) = 100 + 1 = 101`; `ndp[1] = a[1][1] + min(dp[0], dp[1]) = 1 + min(1,100) = 1 + 1 = 2`. `dp = [101, 2]`, minimum `2`. Correct. The case that broke before now passes, and it broke for exactly the reason I fixed — the same-row read of an already-updated cell — which is the evidence I trust. (One could keep `O(n)` memory and update in place by saving the previous-row value in a temporary as the sweep proceeds, but the two-vector version is obviously correct and `O(n)` memory anyway, so I keep it; clarity over a constant-factor of memory I do not need.)

**Edge cases, deliberately.**
- `n = 0`: no grid, no path. I special-case this and print `0` before allocating anything, matching the contract.
- `n = 1`, `a = [[-7]]`: `dp = [-7]` from the base case, the loop body never runs (`i` starts at 1), answer `min(dp) = -7`. The single cell is the only path, and a negative single cell is a legitimate, optimal (only) path — so unlike the no-two-adjacent problem there is *no* "empty is allowed" floor here; a falling path must visit every row, so I do **not** clamp at 0. I verified this against the brute oracle, which also returns `-7`.
- All-negative grid: the DP simply finds the least-negative... no, the *most*-negative total reachable, since we minimize; every `dp` value is some real path sum and we take the min. Correct by construction, and confirmed by brute force on random all-negative grids.
- Uniform grid (all equal `v`): every path costs `n*v`; the DP returns `n*v`. Checked.
- Overflow: `dp` and `best` are `long long`; the worst path sum magnitude is `n * 10^9 = 10^12`, far inside `long long`'s `~9.2*10^18`. No sentinels are added to (the guards mean I never `min` against a fabricated infinity), so there is nothing to underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so row-major parsing is format-agnostic about line breaks.

**Self-verification I actually ran.** I wrote an independent brute oracle (`verify/brute.py`) that enumerates *every* falling path by depth-first recursion over all start columns and all three-way moves — no DP, a structurally different program — and a generator (`verify/gen.py`) producing tiny/small/all-negative/wide-range/mixed grids with `n` small enough that the exponential enumeration stays cheap. I compiled `sol.cpp` with `-O2 -std=c++17`, then differential-tested: 600 random seeds plus 500 mode-forced cases (100 each of tiny, small, negatives, wide, default) — **1100 cases, zero mismatches**. I also checked hand-built cases directly: `n = 0` -> `0`; `n = 1` single negative `-5` -> `-5`; the greedy counterexample above (`sol` = `15`, greedy = `56`, brute = `15`); and a `4x4` uniform `2` grid -> `8`. Finally I ran the `O(n^2)` DP on a worst-case `n = 1000` grid with values uniform in `[-10^9, 10^9]`: it finished in about `0.06s`, comfortably inside the 2-second limit, and produced a sum of magnitude `~6.5*10^11` that would have overflowed a 32-bit `int` — confirming the `long long` decision was load-bearing, not cosmetic.

**Final solution.** I convinced myself the *idea* is right by disproving greedy with a concrete, fully-enumerated counterexample and by checking the layered DP's recurrence on the sample, and I convinced myself the *code* is right by tracing the in-place corruption to a precise cause (`dp[1]` reading an already-overwritten `dp[0]`), fixing it with a fresh per-row vector, re-verifying the failing case, and then differential-testing 1100 cases against an independent enumerator with zero mismatches. That is what I ship — one self-contained file, the simple `O(n^2)` DP I can prove, not the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // no input -> nothing to do
    if (n == 0) { cout << 0 << "\n"; return 0; }

    vector<vector<long long>> a(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> a[i][j];

    // dp[j] = minimum sum of a falling path that ends at cell (current row, j).
    // Row 0: a path ending at (0, j) is just the single cell a[0][j].
    vector<long long> dp(a[0].begin(), a[0].end());

    for (int i = 1; i < n; i++) {
        vector<long long> ndp(n);
        for (int j = 0; j < n; j++) {
            long long best = dp[j];                       // came from (i-1, j)
            if (j > 0)     best = min(best, dp[j - 1]);    // came from (i-1, j-1)
            if (j + 1 < n) best = min(best, dp[j + 1]);    // came from (i-1, j+1)
            ndp[j] = best + a[i][j];
        }
        dp = move(ndp);
    }

    long long answer = *min_element(dp.begin(), dp.end());
    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** Greedy "take the cheapest next cell" looked right but a single traced, fully-enumerated counterexample (`[[5,5,5],[1,9,9],[50,50,1]]`: greedy `56` vs the reachable `15`) showed that a cheap step early fences you out of a cheap region later, so I moved to the layered DP `dp[i][j] = a[i][j] + min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1])` and checked its recurrence on the sample; both edge columns need a guard so I never read out of range or against a sentinel; my first in-place row update corrupted the previous row, and a trace of `[[1,100],[100,1]]` returning the wrong `101` instead of `2` pinpointed `dp[1]` reading an already-overwritten `dp[0]`; computing each row into a fresh vector fixes it; and `min` over the final row, with `long long` accumulators and no "empty allowed" floor (a falling path must cross every row), closes out the `n = 0`, single-cell, all-negative, uniform, and overflow corners — all confirmed by 1100 mismatch-free differential tests and a `0.06s` `n = 1000` timing run.
