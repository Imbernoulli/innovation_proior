A chain of `n` matrices multiplied left-to-right, `A[i]` of shape `p[i-1] x p[i]`, and I choose the parenthesization minimizing total scalar multiplications, where an `a x b` matrix times a `b x c` costs `a*b*c`. Two features of this instance settle the solution before I even pick an algorithm. First the arithmetic scale: `n <= 300` and each `p[i] <= 1000`, so a single multiplication costs up to `10^9` and a full evaluation performs `n-1` of them — the total reaches `~300*10^9 = 3*10^11`, an order of magnitude past the 32-bit ceiling of `~2.1*10^9`. Every cost accumulator and the dimension array have to be 64-bit `long long`; an `int` here is a silent wrong-answer on the large tests, not a crash. Second, `n <= 300` is *small*: an `O(n^3)` method is only `~2.7*10^7` operations, trivially inside the 2-second limit. So if a cubic exact method exists there is no reason to reach for anything cleverer, and the only real question is whether the cheap greedy is exact or whether I need the interval DP.

The tempting `O(n^2)` greedy — repeatedly multiply the adjacent pair whose immediate cost `dims[m]*dims[m+1]*dims[m+2]` is smallest, collapse it into one matrix, repeat — has a structural failure mode: each merge eliminates one interior dimension, and eliminating a small one early can strand two large dimensions adjacent for every later multiply. Take `p = [10, 1, 100, 10]`. The two adjacent merges both cost `1000` (`10*1*100` and `1*100*10`), so greedy ties and picks one — say `A[1]*A[2]`, producing a `10x100` matrix; the remaining multiply against `A[3]` then costs `10*100*10 = 10000`, total `11000`. But `A[1]*(A[2]*A[3])` does `A[2]*A[3]` first (`1000`, a `1x10` matrix) and then `10*1*10 = 100`, total `1100` — ten times cheaper. The cheap first merge exposed the interior `100` as a boundary and made everything after it expensive. Greedy is out, and not marginally: a random sweep of small chains beats it routinely (`[20,100,100,1,50]`: greedy `305000` vs optimal `13000`).

So the interval DP, which is exhaustive rather than heuristic. Let `dp[i][j]` be the minimum cost to reduce `A[i..j]` to one matrix. Whatever the parenthesization, some multiplication is performed last, and it combines a fully-evaluated `A[i..k]` (a `p[i-1] x p[k]` matrix) with a fully-evaluated `A[k+1..j]` (a `p[k] x p[j]` matrix) for exactly one split `k` with `i <= k < j`, at cost `p[i-1]*p[k]*p[j]`. Hence `dp[i][i] = 0` and, for `i < j`,

- `dp[i][j] = min over k in [i, j-1] of ( dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j] )`.

This ranges over every possible outermost split, hence over every full binary tree on `A[i..j]`; nothing is missed. Filling by increasing chain length `len = j - i + 1` from `2` to `n` guarantees every sub-chain a length-`len` interval needs is shorter and already computed, and the answer is `dp[1][n]`.

Transcribing it into code, the loop bounds and the base seed are where this kind of DP actually dies. My first table fill:

```
vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, 0));
for (int len = 2; len <= n; len++) {
    for (int i = 1; i + len - 1 <= n; i++) {
        int j = i + len - 1;
        long long best = 0;                       // <-- wrong
        for (int k = i; k < j; k++) {
            long long cost = dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j];
            if (cost < best) best = cost;
        }
        dp[i][j] = best;
    }
}
```

The `best = 0` seed, for a quantity I am *minimizing* over strictly-positive candidates, is wrong, and the smallest input exposes it: two matrices `p = [2, 3, 4]`, answer obviously `24`. With `len=2, i=1, j=2, best=0`, the single `k=1` gives `cost = 0 + 0 + 24 = 24`, but `24 < 0` is false, so `best` stays `0` and I print `0`. Seeding a minimum at `0` when every candidate is positive floors the result at `0` — it reports every multi-matrix chain as free, and only the degenerate `len <= 1` cases (which never enter this loop) hide it. The fix is to seed `best` with a sentinel above any achievable cost. The max total is `~3*10^11`, so `INF = LLONG_MAX/4 ~ 2.3*10^18` sits enormously above it; and because that sentinel is only ever read inside the comparison `cost < best` and never has a `~10^9` term added to it, it cannot itself overflow. With `best = INF`, re-running `[2,3,4]` now returns `24`, and the greedy counterexample `[10,1,100,10]` returns `dp[1][3] = min(0+1000+10*1*10, 1000+0+10*100*10) = min(1100, 11000) = 1100` — the optimal order greedy missed.

The degenerate sizes I peel off before the table: `n = 0` (empty product; I still read the lone `p[0]` so the parse is well-formed) and `n = 1` (one matrix, nothing to multiply) both return `0`. The `len` loop starts at `2`, so it would never populate `dp[1][n]` meaningfully for those anyway; an explicit early `0` avoids reasoning about the ragged `dp[1][0]` cell. `n = 2` is just the single product `p[0]*p[1]*p[2]`. And the `n = 300` all-`1000` chain returns `299 * 10^9`, which doubles as the concrete demonstration that `int` would overflow — that value does not fit in 32 bits.

For what hand-traces can't reach I lean on a differential test: an independent oracle that enumerates every parenthesization straight from the definition — no memoization, a genuinely different mechanism from the bottom-up fill — run against the compiled solution over hundreds of cases spanning the fixed edge library, greedy-trap chains (alternating tiny and huge dimensions), and random small chains, with zero mismatches. Separately the `n = 300` worst case finishes in a few milliseconds, confirming the cubic work is nowhere near the time limit.

That is what I ship: the `O(n^3)` interval DP over `long long`, degenerate sizes peeled off, the running minimum seeded at `INF`, not the greedy I broke. The full program is in the answer.
