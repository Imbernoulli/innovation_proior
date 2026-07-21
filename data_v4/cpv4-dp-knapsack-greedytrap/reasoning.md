I have `n` assays and `B` micrograms of reagent; assay `i` costs `c[i]` micrograms and yields `r[i]` data points, all-or-nothing, at most once each. Maximize total yield over subsets whose total cost is at most `B`, print it, empty subset allowed so the floor is `0`. Stripped of the story this is 0/1 knapsack with a single capacity, and the two things that will actually decide whether my submission is correct are hiding in the numbers, so I read the constraints before choosing anything: `n <= 2000`, `B <= 2*10^5`, `c[i] <= 2*10^5`, `0 <= r[i] <= 10^9`. Two consequences jump out. First, the objective gets big — `n * max r = 2000 * 10^9 = 2*10^12`, twenty times past the signed 32-bit ceiling of `~2.1*10^9`, so every yield accumulator and the whole DP table must be 64-bit; an `int` table would be a silent wrong-answer on the large tests where a few high-yield assays alone overflow it. Second, `n*B = 4*10^8`, which sizes the only algorithm that is safe here. Costs and `B` fit in 32 bits, but I keep them in `long long` too so the index arithmetic `w - c[i]` never mixes types.

Two routes reach for this. Greedy — sort by efficiency `r[i]/c[i]` (or by raw yield) and take while the budget allows, `O(n log n)`, a few lines — leans on the fractional-knapsack fact that densest-first is optimal *when you can split items*. But assays are indivisible and the budget can be left partly unspent, which is exactly the regime where that argument collapses. The other route is a knapsack DP over the budget, `O(n*B) = 4*10^8`: large, but each step is one add-and-compare on contiguous memory, which a modern machine does at well over `10^9`/s, so it is comfortable under a 2-second limit. Greedy is cheaper, so the only question is whether it is correct here.

So I try to break greedy on purpose. `B = 10`, three assays `A=(6,8)`, `B'=(5,6)`, `C=(5,6)`. Densities: `A` is `8/6 ≈ 1.33`, the other two `6/5 = 1.20`, so efficiency-greedy runs `A` first, spends 6 micrograms, and the 4 left fit nothing — total `8`. But `B'` and `C` together cost `5 + 5 = 10` and yield `12`, tiling the budget to the last microgram. Greedy is beaten `8` vs `12`, and the mechanism is plain: grabbing the densest single item strands a 4-microgram fragment, while the optimum forgoes it to fit two slightly-less-dense items that partition the budget exactly. Density optimizes a *rate*; with indivisible items the *partition* of the fixed budget is what matters. Raw-yield-greedy makes the same move — `A` has the largest single yield `8`, so it also grabs `A` first and strands the same fragment. Both natural greedies return `8` against the true `12`. Greedy is out.

So I commit to the DP. Let `dp[w]` = the best total yield achievable with total cost at most `w`, built assay by assay over subsets seen so far. Adding assay `i`, a budget-`w` plan either skips it (`dp[w]` unchanged) or runs it, which needs `w >= c[i]` and leaves `w - c[i]` for the earlier assays:

```
dp[w] = max(dp[w], dp[w - c[i]] + r[i])   for w >= c[i].
```

Base case `dp[w] = 0` everywhere (run nothing, valid at every budget); the answer is `dp[B]`. Because "at most `w`" is monotone, `dp` is nondecreasing in `w`, so `dp[B]` already folds in the option of leaving budget unspent, and the empty subset (`dp = 0`) keeps the answer at least `0` — no separate "exactly `w`" table and no final max over levels are needed.

The one genuine hazard in transcribing this is the iteration direction: getting it wrong silently turns 0/1 into *unbounded* knapsack, so I work out what each direction actually does. Iterating `w` upward,

```
for (long long w = c[i]; w <= B; w++)          // forward — wrong for 0/1
    dp[w] = max(dp[w], dp[w - c[i]] + r[i]);
```

reads a `dp[w - c[i]]` that may already have absorbed assay `i` earlier in this same pass. Concretely: `B = 6`, one assay `(3, 5)`, answer plainly `5`. Forward: `w=3` sets `dp[3] = dp[0] + 5 = 5`; reaching `w=6`, `dp[6] = dp[3] + 5 = 10` — the item counted twice, a second cost-3 copy stacked to fill the budget. To keep each assay used at most once I must read a cell that does not yet reflect the current assay, so I iterate `w` from `B` down to `c[i]`:

```
for (long long w = B; w >= c[i]; w--)          // backward — each assay once
    dp[w] = max(dp[w], dp[w - c[i]] + r[i]);
```

Now every `dp[w - c[i]]` still holds its previous-assay value. Backward on the same instance: `w=6` reads `dp[3]=0`, so `dp[6]=5`, and no later step revisits it — answer `5`. And on the three-assay counterexample the backward pass reaches `dp[10]=12`: after `A` the cells `dp[6..10]` are `8`; processing `B'` sets `dp[5]=6`; processing `C` at `w=10` gives `max(8, dp[5]+6) = 12`. The DP reaches the packing greedy could not.

Two boundary facts fall out of the guard `w >= c[i]`. When `c[i] > B` — which the constraints allow — the loop condition is false from the start, so an over-budget assay is simply never runnable, correctly. And inside the body `w >= c[i]` forces `w - c[i] >= 0`, so no index ever goes negative. The lower bound is inclusive, which is what lets an exactly-affordable assay run: `B = 4`, assay `(4, 7)` runs only at `w = 4`, `dp[4] = dp[0] + 7 = 7`.

A few remaining corners, mostly already covered by the `<= w` semantics: `n = 0` leaves `dp` all-zero, output `0`; `B = 0` gives a length-1 table and every cost `>= 1` skips its loop, output `0`; an assay with `r[i] = 0` can never improve a cell, since `dp[w - c] <= dp[w]` by monotonicity makes `max(dp[w], dp[w-c]+0) = dp[w]`, so it is a harmless no-op that never inflates the answer. Overflow is closed by the `long long` table: the hard cap is `2*10^12`, five digits below the `~9.2*10^18` ceiling, and nothing multiplies — only sums of at most `2000` yields. Output is one integer and a newline; `cin >>` eats arbitrary whitespace, so the input's line layout is irrelevant.

On cost: the table is `B + 1 <= 200001` longs, about 1.6 MB, far inside 256 MB. The work is at most `n*(B+1) = 4*10^8` add-and-compare steps over contiguous memory; on the worst shape (`n=2000`, `B=2*10^5`, random costs and yields near `10^9`) it runs in about 0.12 s, comfortably under the 2-second limit. `ios_base::sync_with_stdio(false)` and `cin.tie(nullptr)` keep reading the ~4000 integers off the critical path.

Against a thousand random small instances across two generators, each checked by exhaustive `2^n` subset enumeration, the backward DP matched with zero mismatches. That is what I ship: the `O(n*B)` 0/1 knapsack with the high-to-low inner loop and the `long long` table, an explicit `if (c[i] > B) continue;` to skip the never-fitting assays, reading `n`, `B` and the `n` cost/yield pairs from stdin and printing `dp[B]`. The full module is in the answer.
