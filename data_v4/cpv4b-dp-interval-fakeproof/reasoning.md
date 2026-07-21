`n` stones in a row, each `0 <= a[i] < 2^20`; a merge takes two *currently adjacent* stones `x` (left)
and `y` (right), replaces them by one stone valued `x XOR y`, and costs `x OR y` on the two values at
the instant of the merge. After `n - 1` merges one stone remains and I want the minimum total cost. The
sample pins the convention: on `[6, 5, 3]`, merging the left pair first costs
`(6|5) + ((6^5)|3) = 7 + (3|3) = 10`, and merging the right pair first costs
`(5|3) + (6|(5^3)) = 7 + (6|6) = 13`, so the minimum is `10`.

Two sizes matter before any algorithm. Each merge costs at most `2^20 - 1`, and there are at most
`1499` of them, so the total is below `1.6*10^9` — under the signed-32-bit ceiling of `2.1*10^9`, but
close enough that I would rather not audit every intermediate `+` for headroom. `long long`
accumulators throughout. And `n` can be `0` or `1`, where nothing merges and the cost is `0`.

**What kind of problem this is.** A merge only ever joins two stones adjacent in the current row, and
adjacency in the row is adjacency of the underlying index ranges, so at every moment the surviving
stones partition `[0, n-1]` into contiguous segments. A stone covering `[l..r]` has value
`XOR of a[l..r]` — every internal merge XORed its parts and XOR is associative. So a full merge sequence
is a binary parenthesization of the row, and the last merge inside `[l..r]` joins a left block `[l..k]`
and a right block `[k+1..r]`, paying `(XOR a[l..k]) OR (XOR a[k+1..r])`. That is interval / matrix-chain
DP in shape. `n <= 1500` sizes the intended solution: an `O(n^3)` interval DP is `~3.4*10^9` body-ops in
the absolute worst case — tight but plausible in 2 s over a couple of additions, an OR, and a `min` —
which tells me the setters expect exactly this and not a closed form.

**But the problem dangles a closed form.** XOR is associative and commutative, so the final stone equals
`a[0] ^ ... ^ a[n-1]` regardless of order — the final value does not depend on how I merge. That makes
it tempting to guess the *cost* is order-independent too. Two natural one-liners: with `n - 1` merges
and the global OR `O = a[0] | ... | a[n-1]`, maybe the minimum is `(n-1)*O`; or maybe it is the sum of
`a[i] | a[i+1]` over original adjacent pairs, each adjacency crossed once. Both are wrong on the sample:
on `a = [6,5,3]`, `O = 6|5|3 = 7`, so `(n-1)*O = 2*7 = 14`, and the adjacent-OR sum is
`(6|5)+(5|3) = 14`, against the true `10`. Both overcount, and the reason is visible — a merge's cost is
the OR of two *partial* XORs, and partial XORs cancel bits the global OR keeps: the cheap order merges
`6` and `5` into `6^5 = 3`, killing the high bit so the next merge is `3|3 = 3`, while the expensive
order keeps a `6` around and pays `6|6 = 6`. No order-independent quantity can see that cancellation, so
order matters and the interval DP is genuinely needed.

**The DP.** Let `seg(l,r)` = XOR of `a[l..r]`, computed in `O(1)` from a prefix-XOR array `px`
(`px[0]=0`, `px[i+1]=px[i]^a[i]`, `seg(l,r)=px[r+1]^px[l]`). Let `dp[l][r]` be the minimum cost to
reduce `[l..r]` to one stone, base `dp[l][l]=0`. The last merge splits at `k` into two non-empty blocks:

  `dp[l][r] = min over k in [l, r-1] of ( dp[l][k] + dp[k+1][r] + ( seg(l,k) | seg(k+1,r) ) )`,

with answer `dp[0][n-1]`. The split bound is the one place this recurrence bites back: `k` must stop at
`r-1`. Running `k` up to `r` takes the right block `[r+1..r]` empty, reads `dp[r+1][r]` and a reversed
`seg(r+1, r)` outside the DP's meaning, and — because I fill `dp[l][r]` in increasing segment length —
reads `dp[l][r]` itself while it is still the `INF` seed. So the loop is `for (k = l; k < r; k++)`,
non-empty blocks on both sides.

On `[6,5,3]` this reproduces the two real orders: `dp[0][1]=6|5=7`, `dp[1][2]=5|3=7`, then for `[0..2]`
split `k=0` gives `dp[0][0]+dp[1][2]+(6|(5^3)) = 7+6 = 13` and split `k=1` gives
`dp[0][1]+dp[2][2]+((6^5)|3) = 7+3 = 10`, so `dp[0][2]=10` — the two splits are exactly the left-first
and right-first orders.

Two things the code leans on. I allocate `dp` zero-initialized and never touch the diagonal, so
`dp[l][l]=0` is the base case for free. The `best` seed `INF = LLONG_MAX/4` is always overwritten — any
segment of length `>= 2` offers at least the split `k=l` — and only ever seeds a `min`, never an operand
of `+`, so it cannot overflow.

**Cross-checking against a brute force.** Hand traces only cover cases I thought to look at, so I run the
DP against an independent brute force that simulates every legal sequence of adjacent merges on the live
row (every adjacent position at every step, memoized on the row tuple) with no interval decomposition
reused, so the two are unlikely to share a bug. Across hundreds of random small rows (`n` up to 7) with
narrow bit-widths so XOR/OR collisions are dense, plus an exhaustive sweep of all rows with `n <= 5`
over values `{0,1,2,3}`, zero mismatches. On `[6,5,3]` the brute confirms `10` is the global minimum
over all orders, not just the two I checked.

Edge cases: `n=0` guarded explicitly to print `0`; `n=1` prints `dp[0][0]=0` because the length loop
never runs; `n=2` gives the single forced `a[0]|a[1]`; an all-zero row gives `0` since every OR of
XOR-blocks is `0`; and a left-to-right or "cheapest adjacent pair now" greedy is not safe (the sample's
`10` vs `13` already splits on order), which is why every split is tried — the brute agreed on rows
built to punish greedy. At `n=1500` the `n x n` `long long` table is `~18 MB` and the run finishes well inside 2 s /
256 MB.

The full program is in the answer: the prefix-XOR helper, the length-ordered `dp` fill with the `k < r`
split loop, and `dp[0][n-1]` printed.
