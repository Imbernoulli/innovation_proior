The whole hallway-covering turns on one word in the statement: the rug interval `[l, r]` is
**inclusive**, so its length is `r - l + 1` and its cost-`max` runs over both endpoints `a[l]` and
`a[r]`. Every boundary I write — the length cap `r - l + 1 <= L` and the running max — has to carry
that `+1`, and interval-partition code dies precisely on those bounds. The scale settles the data
types before anything else: `n <= 5000`, but `K` and each `a[i]` reach `10^9`, and a covering can use
up to `n` rugs, so the worst total is about `5000 * 2*10^9 = 10^13`, well past the 32-bit ceiling.
Every dp accumulator must be `long long`; an `int` dp table is a silent wrong answer on the big tests.

Two routes are on the table. Greedy — march left to right laying the longest (or locally cheapest)
legal rug, `O(n)` — is tempting, but a rug is charged only for its single roughest panel, which makes
the marginal value of extending a rug wildly non-local: a rug that already covers a rough panel can
swallow more panels almost for free, while one sitting on smooth panels should perhaps stop before it
reaches a rough one. Partition DP — `dp[i]` = min cost to cover panels `1..i`, the last rug an
inclusive suffix `[j+1, i]` — is `O(n*L) <= 2.5*10^7`, comfortable under a second. Before trusting
greedy I try to break it on the sample. `a = [1,5,5,1,5]`, `K=2`, `L=2`: longest-first lays
`[1,2],[3,4],[5,5]` for `7+7+7 = 21`. But isolating the smooth first panel — `[1,1],[2,3],[4,5]` —
costs `3+7+7 = 17`, strictly better. Greedy extended the first rug onto panel 2 and forced panel 1
(roughness 1) to share a rug with a rough panel, paying 5 for its coverage instead of 1. Charging only
for the max makes "use the whole length" a globally wrong local rule; greedy is out, and I commit to
the DP.

Setting it up: `dp[0] = 0`, and the last rug covering `1..i` ends exactly at `i` and starts at some
`j+1`, covering the inclusive interval `[j+1, i]` of length `i - (j+1) + 1 = i - j`. Legality
`1 <= i - j <= L` gives `j` in `[max(0, i-L), i-1]`. Sweeping `j` down from `i-1` grows the rug one
panel to the left each step, so a running `curMax = max(curMax, a[j+1])` tracks its roughest panel and

  `dp[i] = min over j of  dp[j] + K + curMax`,

Let me confirm the recurrence by hand on the sample `a = [1,5,5,1,5]`, `K = 2`, `L = 2`, answer `17`.
`dp[0]=0`.
- `dp[1]`: `j=0`, rug `[1,1]`, max `1`, cand `0+2+1=3`. So `dp[1]=3`.
- `dp[2]`: `j=1` rug `[2,2]` max `5` cand `3+2+5=10`; `j=0` rug `[1,2]` max `5` cand `0+2+5=7`. `dp[2]=7`.
- `dp[3]`: `j=2` rug `[3,3]` max `5` cand `7+2+5=14`; `j=1` rug `[2,3]` max `5` cand `3+2+5=10`. `dp[3]=10`.
- `dp[4]`: `j=3` rug `[4,4]` max `1` cand `10+2+1=13`; `j=2` rug `[3,4]` max `5` cand `7+2+5=14`. `dp[4]=13`.
- `dp[5]`: `j=4` rug `[5,5]` max `5` cand `13+2+5=20`; `j=3` rug `[4,5]` max `5` cand `10+2+5=17`. `dp[5]=17`.

with answer `dp[n]` and no separate inner max loop.

Both boundaries here are inclusive-interval traps, and each sits one index away from the natural wrong
guess:

- Length cap. The rug `[j+1, i]` has length `i - j`, so the smallest legal `j` is `i - L`. Looping one
  step further, to `j = i - L - 1`, admits a rug of length `L + 1`. On `a = [1,2,2]`, `K=1`, `L=2`
  (legal optimum 5: `[1,1]` for 2, then `[2,3]` for 3), that extra step would let a single rug cover
  `[1,3]`, length 3, at cost `0 + 1 + 2 = 3` — an illegal partition beating the legal answer. So the
  loop stops at `j >= max(0, i-L)`, never `lo-1`.
- Max endpoint. The running max must fold `a[j+1]`, the rug's inclusive left panel, not `a[j]`. With
  `a[j]`, the length-1 rug `[1,1]` at `j=0` reads the unused slot `a[0] = 0` and gets priced at
  `K + 0` instead of `K + a[1]`, undercounting that same `[1,2,2]` to 4. The `+1` is load-bearing on
  both bounds.

**A second trace, because the running max has its own boundary.** With the length bound fixed I
re-examine the max update `curMax = max(curMax, a[j + 1])`. The rug is `[j+1, i]`, so its leftmost
panel is `a[j+1]` — the `+1` matters. A very natural slip is to write `curMax = max(curMax, a[j])`,
reasoning "I'm at split index `j`, fold in `a[j]`." Let me trace *that* variant on `a = [1,2,2]`,
`K=1`, `L=2`, where I now know the answer is `5`. With the buggy `a[j]`:
- `dp[1]`: `j=0` rug `[1,1]`, update `curMax = max(0, a[0])`. But `a[0]` is the unused index-0 slot,
  value `0`. cand `0 + 1 + 0 = 1`. `dp[1]=1`.
- `dp[2]`: `j=1` rug `[2,2]` `curMax=max(0,a[1])=1` cand `1+1+1=3`; `j=0` rug `[1,2]`
  `curMax=max(1,a[0])=1` cand `0+1+1=2`. `dp[2]=2`.
- `dp[3]`: `j=2` rug `[3,3]` `curMax=max(0,a[2])=2` cand `2+1+2=5`; `j=1` rug `[2,3]`
  `curMax=max(2,a[1])=2` cand `1+1+2=4`. `dp[3]=4`.

On the sample the recurrence sweeps `dp = 0, 3, 7, 10, 13, 17`, so `dp[5] = 17`, realized by
`[1,1],[2,3],[4,5]` — the partition I reached by hand against greedy.

The corners the hidden tests probe all fall out of these bounds. `n=1`: only `j=0`, rug `[1,1]`,
answer `K + a[1]`. `L=1`: `lo = i-1`, the sole `j` is `i-1`, forcing singletons that total
`nK + sum a[i]`. `L=n`: `lo=0` throughout, so the single rug `[1,n]` is offered, and when `K` is large
enough that every extra split costs another `K` the DP takes it. `L=n-1`: the full-hallway rug needs
`j=0` at `i=n`, but `n - 0 = n > L`, so it is never offered and at least two rugs are forced — exactly
the length off-by-one, now landing on the correct side of the cap. `K=0`: the recurrence is unchanged.
Overflow: `dp` is `long long`, worst total near `10^13`; `INF = LLONG_MAX/4` is only read inside
comparisons and behind a `dp[j] != INF` guard, so `K + curMax` is never added to a poisoned state.
The `O(n*L)` program is in the answer.
