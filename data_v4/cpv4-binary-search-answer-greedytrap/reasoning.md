**Reading the problem and pinning the contract.** I have `n` plant beds in a row, bed `i` wants `a[i]` liters (`a[i] >= 0`), and I must cut the row into `k` contiguous blocks, one per robot. A robot's busy time is the sum of its block; all robots run in parallel, so the greenhouse finishes at the *largest* block sum. I want to place the cuts so that largest block sum is as small as possible, and I print that minimum. Input is `n k` then the `n` values; output is one integer. Before any algorithm I fix the scale, because it decides the types: `n <= 2*10^5`, `a[i] <= 10^9`, so the total `S = sum(a)` can reach `2*10^5 * 10^9 = 2*10^14`. That is far above the 32-bit ceiling of `~2.1*10^9`, so every sum, every accumulator, and the answer must be 64-bit `long long`. Also `k` can be as large as `10^9` while `n` is only `2*10^5`; `k` itself fits in 32 bits but I will carry it as `long long` so a comparison like `blocks <= k` never has to worry about mixed signedness or a stray promotion. That is decision one and it is non-negotiable — an `int` total is a silent wrong answer on the big tests.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is fastest to type.

- *Balance-by-average greedy.* The fair share per robot is `S/k`. Walk left to right and cut a block as soon as its running sum reaches about `S/k`; or, equivalently, drop the `k-1` cut points at the prefix sums nearest to `S/k, 2S/k, ...`. It is `O(n)` and a dozen lines. The danger is that "make every block near the average" and "make the maximum block small" are different objectives, and contiguity removes my freedom to move beds around. This is exactly the configuration where a greedy that optimizes the wrong quantity locally produces a globally worse maximum. I will not trust it until I have tried to break it.
- *Binary search on the answer.* Guess a finish time `T` and ask the decision question: can I cut the row into at most `k` contiguous blocks each with sum `<= T`? If yes for `T`, then yes for every `T' > T` (I can only get more slack), so feasibility is **monotone** in `T` and the minimum feasible `T` is found by bisection. The feasibility check itself has an easy *correct* greedy: keep extending the current block while it still fits under `T`, and cut the instant the next bed would overflow; that uses the fewest blocks possible for that `T`. The open questions are the search bounds and getting the block count exactly right.

**Stress-testing the average greedy before committing.** Hand-waving "balance feels right" is how wrong solutions get shipped, so let me actually attack it with a concrete instance. Take `a = [7, 1, 2, 9, 2, 6]`, `k = 3`. The total is `S = 27`, so the target share is `S/k = 9`, and the greedy wants cut boundaries where the prefix sums are nearest to `9` and `18`. Prefix sums are `7, 8, 10, 19, 21, 27`. Nearest to `9`: both `8` (after bed index 1) and `10` (after bed index 2) are distance `1`; the greedy takes the first it meets, cutting after index 1 — block `[7, 1]` summing `8`. Nearest to `18` among the rest: `19` (after index 3) at distance `1` wins, cutting after index 3 — block `[2, 9]` summing `11`. The last block is `[2, 6]` summing `8`. Greedy's blocks are `[7,1] | [2,9] | [2,6]` with sums `8, 11, 8`, so its maximum is **11**.

Is 11 optimal? Let me hunt for a partition the average-greedy structurally could not reach. Try `[7, 1, 2] | [9] | [2, 6]`: sums are `10, 9, 8`, maximum **10**. That is strictly better than 11. So the greedy is wrong, and I see *why*: by trying to land each block near `9` it never wanted to isolate the lone `9` into its own block — isolating it pushes one block *below* average (`9` itself) and a neighbour slightly *above* the cut it preferred, which the "stay near the average" rule actively avoids. Minimizing the max sometimes wants deliberately *unbalanced* blocks, and the average greedy refuses that. The verification paid off — it killed an approach I would otherwise have shipped.

**A second, sharper way the average greedy dies.** There is an even cleaner failure: the average can be *infeasible*. Take the sample `a = [2, 3, 5, 8, 1, 1, 4]`, `k = 3`, total `S = 24`, average `8`. If I greedily fill each block up to `8` and cut on overflow: `[2,3]` (5, next `5` would make 10 > 8, cut) then `[5]` (next `8` makes 13 > 8, cut) then `[8]` (next `1` makes 9 > 8, cut) then `[1,1,4]` (sum 6). That is **four** blocks for `k = 3`. The greedy literally cannot finish inside the budget it chose, because the bed `8` plus *anything* exceeds the average, so the average is below the largest unavoidable block. The real answer here is `10` with `[2,3,5] | [8] | [1,1,4]`. This nails the lower-bound subtlety I have to respect: the answer can never drop below the single largest bed, and it can never drop below `ceil(S/k)` either, but neither bound alone is the answer — only the smallest *feasible* `T` is. Greedy on the average is out; binary search on `T` it is.

**Deriving the feasibility check and its bounds, then a paper sanity check.** For a fixed `T`, the minimum number of contiguous blocks with each sum `<= T` is produced by the obvious greedy: start a block, keep adding beds while the running sum stays `<= T`, and the moment the next bed would push the sum over `T`, close the block and start a new one at that bed. This is optimal for the *count* because closing a block early can never let a later block hold more — extending greedily is always at least as good (a standard exchange argument). One guard: if any single bed has `a[i] > T`, no block can ever hold it, so `T` is infeasible outright. The decision is then "is the resulting block count `<= k`?".

The bounds. `feasible(T)` is `false` for small `T` and `true` for large `T`, monotone, so I bisect on `[lo, hi]`. The smallest `T` that could *possibly* work is `lo = max(a[i])` — some block must contain the biggest bed, so the maximum block is at least that. The largest `T` I ever need is `hi = S` — one block holding everything is always feasible (one block `<= k` since `k >= 1`). I search for the least feasible `T` in `[lo, hi]` with the standard lower-bound bisection: while `lo < hi`, test `mid`; if feasible move `hi = mid`, else `lo = mid + 1`; the loop ends with `lo` the minimum feasible value.

Let me confirm the whole thing by hand on the sample `a = [2,3,5,8,1,1,4]`, `k = 3`, expected `10`. `lo = max = 8`, `hi = S = 24`. Test `T = 10` (a value I expect feasible): blocks `[2,3,5]` (10, next `8` makes 18 > 10, cut), `[8,1,1]`? `8+1=9 <= 10`, `+1=10 <= 10`, next `4` makes 14 > 10, cut → `[8,1,1]`, then `[4]`. That is `[2,3,5] | [8,1,1] | [4]`, three blocks, `<= 3`, feasible. Test `T = 9`: `[2,3]` (5, `+5=10>9` cut), `[5]` (`+8=13>9` cut), `[8,1]` (9, `+1=10>9` cut), `[1,4]`... that is already four blocks → infeasible. So the least feasible `T` is `10`. The derivation lands on the right answer.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the feasibility lambda counts blocks like this:

```
auto feasible = [&](long long T) -> bool {
    long long cur = 0, blocks = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] > T) return false;
        if (cur + a[i] > T) { blocks++; cur = a[i]; }
        else cur += a[i];
    }
    return blocks <= k;
};
```

Something about starting `blocks = 0` and only ever incrementing it *when I overflow* looks suspicious — what counts the very first block, or a row that never overflows? Let me trace the smallest input that exposes it: `a = [1, 1]`, `k = 2`, `T = 5`. Obviously feasible (everything fits in one block, well within `k = 2`). Run it: start `cur = 0, blocks = 0`. i=0: `1 <= 5`, and `0 + 1 = 1 <= 5`, so `cur = 1`. i=1: `1 <= 5`, `1 + 1 = 2 <= 5`, so `cur = 2`. Loop ends, `blocks = 0`, return `0 <= 2` → `true`. It returned the right verdict, but with `blocks = 0` — for an input that uses one block! Now trace a case where that miscount flips the answer: `a = [5, 5, 5]`, `k = 1`, `T = 5`. The true answer is "need 3 blocks, but `k = 1`" → infeasible. Run it: `cur=0, blocks=0`. i=0: `0+5=5 <= 5`, `cur=5`. i=1: `5+5=10 > 5`, overflow → `blocks=1, cur=5`. i=2: `5+5=10 > 5`, overflow → `blocks=2, cur=5`. End, return `2 <= 1` → `false`. By luck the verdict is right here, but the count is `2` when the partition `[5]|[5]|[5]` plainly uses **3** blocks.

**Diagnosing the bug.** The defect is precise: I count an *increment per overflow*, but the number of blocks is "one initial block plus one per overflow." Starting `blocks = 0` undercounts every partition by exactly one. It happened to keep the right verdict on my two traces only because the off-by-one shifted both sides of `<= k` the same direction in those particular numbers; it will not stay lucky. Construct the case that breaks the verdict: `a = [3, 3, 3]`, `k = 2`, `T = 3`. The truth is `[3]|[3]|[3]` = 3 blocks > 2 → **infeasible**. Buggy run: `cur=0, blocks=0`. i=0: `0+3=3<=3`, `cur=3`. i=1: `3+3=6>3`, `blocks=1, cur=3`. i=2: `3+3=6>3`, `blocks=2, cur=3`. Return `2 <= 2` → **`true`**. Wrong — it declares an infeasible `T` feasible, which makes the binary search return a `T` that is too small, i.e. an answer below the true optimum. The undercount is a genuine correctness bug, not cosmetics.

**Fixing and re-verifying.** Start the count at `1` — every non-empty row uses at least one block — and bail early the instant the count exceeds `k`:

```
auto feasible = [&](long long T) -> bool {
    long long cur = 0, blocks = 1;
    for (int i = 0; i < n; i++) {
        if (a[i] > T) return false;
        if (cur + a[i] > T) { blocks++; cur = a[i]; if (blocks > k) return false; }
        else cur += a[i];
    }
    return blocks <= k;
};
```

Re-trace `[3,3,3]`, `k=2`, `T=3`: `cur=0, blocks=1`. i=0: `0+3=3<=3`, `cur=3`. i=1: `6>3`, `blocks=2, cur=3`, `2 <= 2` ok. i=2: `6>3`, `blocks=3`, `3 > 2` → return `false`. Correct now. Re-trace `[5,5,5]`, `k=1`, `T=5`: `blocks` becomes `2` at i=1 and `2 > 1` → `false` immediately. Correct, and now with the honest count of "would be 3". Re-trace `[1,1]`, `k=2`, `T=5`: never overflows, ends `blocks=1`, `1 <= 2` → `true`. Correct. The cases that broke (or that I distrusted) now pass, and they broke for the reason I fixed, which is the evidence I trust. Note the early `if (blocks > k) return false;` is not just an optimization: when `k` is up to `10^9` it stops `blocks` from running far past `k`, though as a `long long` capped at `n` it could never actually overflow — it is correctness-neutral but cheap insurance and keeps the intent obvious.

**Second debug episode — the search bounds and `n = 0`.** With feasibility correct I wire up the bisection. My first bounds were `lo = 0, hi = S`. Trace a case where `lo = 0` matters: it does not, because `feasible` is monotone and the search converges regardless of starting `lo` *as long as the true answer lies in `[lo, hi]`* — and `0` is a valid lower endpoint. But there is a real corner hiding in the *upper* end and the empty input. Consider `n = 0` (no beds): then `hi = S = 0` and `lo = 0`, the `while (lo < hi)` body never runs, and I would print `lo = 0`. That is actually correct — no beds means finish time `0` — but only by accident, and I want it explicit, so I short-circuit `if (n == 0) { print 0; }` before the loop to make the intent unambiguous and to avoid relying on `lo` happening to be `0`.

Now trace the more dangerous corner: all-zero beds, `a = [0, 0, 0]`, `k = 2`. Then `lo = max = 0`, `hi = S = 0`, loop never runs, print `0`. Correct: every block sums `0`, the max is `0`. And `k >= n`, say `a = [5, 1, 9, 2]`, `k = 1000000000`: `lo = 9`, `hi = 17`. Test `T = 13`: greedy makes `[5,1]`(6), then `9` overflows `→ [9,2]`(11)? `9+2=11<=13` so `[9,2]`, that is 2 blocks, `2 <= 1e9` feasible. Search drives down. Test `T = 9`: `[5,1]`(6), `+9=15>9 → [9]`, `+2=11>9 → [2]`, three blocks, `3 <= 1e9` feasible. Test `T = 8`: `[5,1]`(6), `+9` over → `[9]`? but `9 > 8`! the `a[i] > T` guard fires → infeasible. So the least feasible `T` is `9` = the largest bed, exactly what `k >= n` should give (every big bed alone). The `lo = max(a[i])` lower bound is what makes that converge cleanly; had I left `lo = 0` the answer is still correct (monotonicity), but starting at `max` saves useless iterations and documents the floor. I keep `lo` initialized to `max(a[i])`.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: handled explicitly, prints `0`. The greenhouse with no beds finishes instantly.
- `k = 1`: feasibility forces one block, so the only feasible `T >= S`; the search returns `hi = S`, the whole sum. Traced `a=[3,1,4,1,5]`, `k=1` → `14`. Correct.
- `k >= n`: every bed can be its own block, answer = `max(a[i])`; traced above, returns `9`. Correct.
- All zeros: returns `0`, traced above.
- A single bed dwarfing the average, `a=[1,1,1,10]`, `k=4`: `lo=10`, and `T=10` is feasible (`[1][1][1][10]`... actually greedy `[1,1,1]`(3), `+10>10 → [10]`, two blocks `<= 4`), so answer `10`. The lone big bed sets the floor — exactly the average-greedy trap, now handled. Correct.
- Overflow: `S` and every accumulator are `long long`; the max total `~2*10^14` fits with three decimal digits to spare. `mid = lo + (hi - lo)/2` avoids `lo + hi` overflowing. Verified on `n = 2*10^5`, all `10^9`, `k = 1`: prints `200000000000000`. Correct.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace so the values may span any number of lines.

**Final solution.** I convinced myself the *idea* is right by disproving the average greedy two different ways (`[7,1,2,9,2,6]`: greedy 11 vs the reachable 10; and the sample where the average is outright infeasible) and by hand-checking the bisection on the sample; I convinced myself the *code* is right by tracing the block-count off-by-one to a precise cause (`blocks = 0` undercounts by one, flipping the verdict on `[3,3,3]`, `k=2`) and re-verifying the fix and every corner. That is what I ship — one self-contained file, the `O(n log S)` binary search I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Feasibility: can we partition a[0..n-1] into at most k contiguous blocks,
    // each with block-sum <= T?  Greedy: extend the current block while the running
    // sum stays <= T; when it would exceed, cut (start a new block at this bed).
    auto feasible = [&](long long T) -> bool {
        long long cur = 0;
        long long blocks = 1;            // we always use at least one block
        for (int i = 0; i < n; i++) {
            if (a[i] > T) return false;  // a single bed already exceeds T
            if (cur + a[i] > T) {        // start a new block at bed i
                blocks++;
                cur = a[i];
                if (blocks > k) return false;
            } else {
                cur += a[i];
            }
        }
        return blocks <= k;
    };

    // Lower bound: the largest single bed (some block must hold it).
    // Upper bound: the total (one block holds everything).
    long long lo = 0, hi = 0;
    for (long long x : a) { lo = max(lo, x); hi += x; }

    // Empty greenhouse: no beds, finishing time is 0.
    if (n == 0) { cout << 0 << "\n"; return 0; }
    // If k >= n every bed can be its own block, so the answer is the max bed.
    // (The binary search handles this too, but it is a clean sanity anchor.)

    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
```

**Causal recap.** The average greedy looked right but two traced counterexamples (`[7,1,2,9,2,6]`, `k=3`: balanced 11 vs the unbalanced-but-better 10; and `[2,3,5,8,1,1,4]`, `k=3`: the average `8` is *below* an unavoidable block so it cannot even hit `k=3`) showed minimizing the max sometimes demands deliberately unbalanced blocks — so I switched to binary search on the finish time `T`, where feasibility is monotone and decided by the fewest-blocks greedy; my first feasibility lambda started the block count at `0`, undercounting by one and flipping the verdict on `[3,3,3]`, `k=2` (claimed feasible when 3 blocks are needed for `k=2`), which a trace pinned to "first block is never counted," fixed by `blocks = 1`; and `long long` everywhere with `lo = max(a[i])`, `hi = S`, the `mid = lo + (hi-lo)/2` form, and an explicit `n = 0` short-circuit close out the empty, all-zero, `k=1`, `k>=n`, lone-giant-bed, and overflow corners.
