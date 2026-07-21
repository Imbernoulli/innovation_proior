I have `n` slabs in a row with widths `w[0..n-1]`, and a robot fuses the whole row into one piece by
welding *currently adjacent* pieces, each weld costing the combined width of the two pieces joined. I
want the minimum total cost over all weld orders. The two numbers in the constraints already tell me
most of what the solution has to look like. `n <= 400` and each `w[i] <= 10^6`: the total width is at
most `4*10^8`, the final weld over the whole row pays that much on its own, and since earlier welds
re-pay overlapping sub-widths the grand total scales like `(n-1)` times the total width — order
`1.6*10^11`. That blows past the 32-bit ceiling of `2.1*10^9`, so every accumulator and the whole DP
table has to be 64-bit; an `int` here is a silent wrong answer on the large tests, not a crash. And
`400^3 = 6.4*10^7` fits comfortably in 2 seconds while `400^4` does not, so I am being pointed at an
`O(n^3)` interval DP.

Before I build it, the cost structure smells Huffman-like — always weld the cheapest adjacent pair —
and I want to know whether that shortcut is safe. It is not, and the reason is structural rather than
something I need a counterexample to see: Huffman may combine *any* two items, but here I may only
fuse *neighbours*. Fusing two cheap neighbours buries a slab in the interior of a piece that is then
re-paid by every later weld, whereas a different order can keep an expensive boundary slab on the
outside and weld it last. That is a global trade-off a local smallest-first rule cannot see, so I
discard greedy and commit to the DP; an exhaustive order-enumeration brute force over small rows will
confirm it.

The recurrence follows from one fact about any full fusion of a closed range `[i, j]`: its *last*
weld joins two adjacent sub-pieces that together span all of `[i, j]`, so it pays the full range
width `W(i, j) = w[i] + ... + w[j]` no matter how the two halves were built, and each half is itself
a fully-fused sub-range. So with `dp[i][j]` the minimum cost to fuse `[i, j]`:

```
dp[i][i] = 0
dp[i][j] = min over k in [i, j-1] of ( dp[i][k] + dp[k+1][j] ) + W(i, j)
```

`k` is the last index of the left half `[i, k]`, leaving the right half `[k+1, j]`; both halves must
be non-empty, which forces `i <= k <= j-1`. Every parenthesization of the line is captured by some
split at each level, and once the split is fixed each half's cost is independent of the other, so the
recurrence is exhaustive and has optimal substructure.

Now the part this problem is actually built around: the boundaries. I store widths in a half-open
prefix sum `prefix[t] = w[0] + ... + w[t-1]`, so `prefix[0] = 0` and `prefix[n] = total`. The width
term is where the trap lives, because the range is *closed* `[i, j]` while the prefix array is
half-open, and the two conventions sit one index apart. The width of `[i, j]` is `prefix[j+1] -
prefix[i]` — with `j+1`, not `j`. The naive `prefix[j] - prefix[i]` sums the half-open `[i, j)`,
dropping slab `j` entirely: on the two-slab row `[2, 3]`, whose only weld costs `2+3 = 5`, it would
give `prefix[1] - prefix[0] = 2`, while `prefix[2] - prefix[0] = 5` is right. That is exactly the
inclusive/exclusive slip the constraints invite, so I fix the index before writing the loop.

The split bound carries the same hazard from the other side. `k` must leave both halves non-empty,
so it runs `i .. j-1` and the loop is `k < j`; `k = j` would mean an empty right half `[j+1, j]` and,
concretely, read `dp[i][j]` (the cell being computed) and the inverted range `dp[j+1][j]`, both
garbage. The outer interval `[i, j]` with `j = i + len - 1` needs `j <= n-1`, i.e. the loop condition
`i + len - 1 < n`. One subtlety in the length loop that is not cosmetic: `len` must start at `2`.
Length-1 intervals are the base case already sitting at `0` in the table; if `len` started at `1` the
body would re-enter each singleton with an empty `k`-loop, leave `best = INF`, and poison `dp[i][i]`
to `INF + width`, corrupting everything above it.

The sample `w = [3, 1, 4, 1]` runs the recurrence end to end. `W(0,3) = 9`. Splitting `dp[0][3]` at
`k=1`: `dp[0][1] + dp[2][3] + 9 = (3+1) + (4+1) + 9 = 4 + 5 + 9 = 18`. The other splits lose — `k=0`
gives `0 + dp[1][3] + 9 = 0 + 11 + 9 = 20`, `k=2` gives `dp[0][2] + 0 + 9 = 12 + 9 = 21` — so the
minimum is `18`, matching the stated answer. The weld order behind it is `(3,1)`, then `(4,1)`, then
the final join, paying `4 + 5 + 9`.

That leaves the edges where interval DP tends to die. `n = 0` and `n = 1` need no welds, and I
short-circuit both to `0` before building any table, which also dodges a zero-size vector; `n = 2`
falls out as `w[0] + w[1]`. For overflow safety the table and accumulators are `long long`, `INF =
LLONG_MAX/4 ≈ 2.3*10^18`, and the largest real `dp[i][k] + dp[k+1][j]` is around `3*10^11`, nowhere
near `INF`; `INF` is only ever read inside the `min` and never has a width added to a cell that
stayed `INF`, because every `dp[i][j]` with `i < j` gets a real value from a non-empty `k`-loop. The
table is `400*400*8 ≈ 1.3 MB` and the `6.4*10^7` operations run in hundredths of a second.

The full program is in the answer: a half-open prefix sum, the `O(n^3)` interval DP with `k` over
`i .. j-1` and `len` from `2`, width `prefix[j+1] - prefix[i]`, and `dp[0][n-1]` printed at the end.
Run against the order-enumeration brute force on hundreds of random small rows, it matches with no
disagreement.
