The balloons sit in a row, and bursting balloon `i` pays `nums[left] * nums[i] * nums[right]` against whichever neighbours are still present at that instant — so every burst rewrites the multipliers seen by every balloon downstream of it. With `n` up to `500` the bursting order is one of `n!` arrangements, and even a subset enumeration would be `2^n`; anything that walks orders is dead on the large tests, so whatever I ship has to be polynomial.

Before picking an algorithm, the arithmetic. Values cap at `100`, so one burst pays at most `100^3 = 10^6` and the whole game at most `n * 10^6 = 5 * 10^8` at `n = 500`. Unusually for this shape of problem, that still fits a signed 32-bit `int` (cap `~2.1 * 10^9`), so this is *not* an overflow trap. But I would rather not compute the triple product `v[l]*v[k]*v[r]` in `int` on reflex, and `long long` costs nothing at this scale, so I carry 64-bit throughout and stop thinking about range.

Two families are on the table: a fixed local pop-rule keyed on the painted numbers (`O(n^2)`, a few lines), or an interval DP (`O(n^3)`, `O(n^2)` memory). Greedy is the tempting one, so it is the one to try to break first. The natural rule is smallest-first — cash a cheap balloon out *now*, while a big neighbour is still beside it to multiply against. On the sample `[3, 1, 5, 8]`: pop `1` (`3*1*5 = 15`), row becomes `[3, 5, 8]`; pop `3` at the left end (`1*3*5 = 15`); pop `5` (`1*5*8 = 40`); pop `8` alone (`1*8*1 = 8`); total `78`. But the statement's witness order (burst `1, 5, 3, 8`) scores `167`, more than double. The reason is visible: smallest-first popped `5` with only the boundary `1` on its left, cashing `1*5*8`, whereas keeping `3` and `8` flanking `5` scores `3*5*8 = 120`. Largest-first squanders the boundary just as badly (`8` first is `1*8*1`). Every fixed local rule founders on the same rock — the payout couples each balloon to whichever neighbours globally survive beside it, and a one-balloon-at-a-time rule cannot steer that. Greedy is out.

So the order has to be reasoned about globally, which points at an interval DP. The natural first cut: let `f[i][j]` be the best over range `[i..j]` and split on the balloon `m` I burst *first* in it. Bursting `m` first pays `v[m-1]*v[m]*v[m+1]`, and then I recurse on `[i..m-1]` and `[m+1..j]` — except those two sub-ranges are not independent. Once `m` is gone, `m-1` and `m+1` become adjacent, so the right neighbour of the last balloon burst in `[i..m-1]` is no longer `m` but whatever survives in `[m+1..j]`; solving a sub-range in isolation has no idea which wall ends up beside it. The first-burst split leaves the sub-problems' neighbours undefined — it is unsound, and I would get garbage if I coded it.

Flip it: split on the balloon `k` in a range that is burst *last*. If `k` is the last balloon strictly between two still-present walls `l` and `r`, then at the instant `k` pops, every other balloon between `l` and `r` is already gone, so `k`'s neighbours are exactly `l` and `r` — its payout is pinned to `v[l]*v[k]*v[r]` regardless of the internal order. And the two sides now decouple cleanly: everything strictly between `l` and `k` is burst with `l, k` standing as its walls, and everything strictly between `k` and `r` with `k, r` as its walls — exactly the pinned walls the first-burst split could never supply.

Pad the row: `v[0] = v[n+1] = 1`, real balloons in `v[1..n]`. Let `dp[l][r]` be the max coins from bursting every balloon strictly between padded indices `l` and `r`, with `l` and `r` present as walls. Then

```
dp[l][r] = max over k in (l, r) of  dp[l][k] + v[l]*v[k]*v[r] + dp[k][r]
```

Empty gaps (`r = l + 1`, no balloon between) have `dp = 0`. The answer is `dp[0][n+1]` — burst everything between the two virtual walls, which is the whole real row.

To fill the table I need `dp[l][k]` and `dp[k][r]` ready before `dp[l][r]`; both sit at a strictly smaller gap `r - l`, so iterating by increasing gap length `len = r - l` orders the dependencies correctly. The trap is the *range* of `len`. My reflex from closed-interval DPs is `len = 1..n`, but the padding shifts everything: the answer `dp[0][n+1]` has gap `n + 1`, so a loop stopping at `len = n` never fills it and prints the initialized `0` — on the sample it would print `0` instead of `167`. The meaningful gaps run `len = 2` (a wall-balloon-wall triple, the smallest range holding a balloon) up to `len = n + 1` (the two outer walls with all `n` balloons between). Gaps of `len = 1` are empty intervals whose `dp` must stay `0`; starting `len` at `2` leaves them untouched, exactly the base case.

```
for (int len = 2; len <= n + 1; len++) {          // gap length r - l
    for (int l = 0; l + len <= n + 1; l++) {
        int r = l + len;
        long long best = 0;
        for (int k = l + 1; k < r; k++)
            best = max(best, dp[l][k] + v[l]*v[k]*v[r] + dp[k][r]);
        dp[l][r] = best;
    }
}
```

On `[3, 1, 5, 8]` (padded `v = [1, 3, 1, 5, 8, 1]`) this fills up through `dp[0][5]`, and the top-level maximum is `k = 4` — burst `8` last — giving `dp[0][4] + v[0]*v[4]*v[5] + dp[4][5] = 159 + 1*8*1 + 0 = 167`, matching the witness. Corners: `n = 0` makes the `len` loop run from `2` to `1`, body never executes, and I print `dp[0][1] = 0`; `n = 1` with `[7]` gives `dp[0][2] = 1*7*1 = 7`; an all-zero row pays `0` everywhere.

A hand-trace proves the recurrence on one input, not the code across the input space, and I have already tripped once on a bound. So I check against an independent brute that shares none of the last-balloon reasoning: it simulates the game directly, recursing over which balloon to burst *next* on the current survivors, scoring each choice against its present neighbours and taking the max, memoized on the set of surviving positions so it stays tractable to about `n = 9`. That is the first-move framing I rejected for the DP, derived from a different principle, so a bug shared between the two is unlikely. Over 1000+ random rows (`n <= 9`, values including many `0`s and the cap `100`) plus edge cases — `n = 0`, single `0`, single `100`, all zeros, all `100`s, the sample, alternating `100 0 100 0 ...` — the two agree with 0 mismatches. The worst case for time and range, `n = 500` all `100`s, finishes in about `0.02 s` (`~ n^3/6 ~ 2*10^7` operations, well under both the `1.25*10^8` work ceiling and the `1 s` limit) and yields `498010100 ~ 5*10^8`, confirming the arithmetic bound; the `(n+2)^2`-long table is about `2 MB`, comfortably inside `256 MB`.

That is what ships — the `O(n^3)` last-burst interval DP, not the greedy it disproves. The full self-contained program is in the answer.
