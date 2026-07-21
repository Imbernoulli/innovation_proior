The count I need is the number of integer tuples `(x_1, ..., x_n)` with `0 <= x_i <= c` and
`sum x_i = k`, reduced mod `p = 1 000 000 007`. Input is the three integers `n k c`, output is one
integer. What sets the shape of the solution before any algorithm is the scale: `n, k` up to `2*10^6`
with `n + k <= 4*10^6`. Any factorial-style precompute has to be linear in `n + k` and live inside
256 MB — two `long long` arrays of length `~4*10^6` is about 64 MB, comfortable. And I will be
multiplying values near `10^9`, so every modular product has to pass through `__int128`; a plain
64-bit multiply of two near-`p` values overflows silently into a wrong answer.

Two ways to count. The direct DP carries `dp[s]` = ways to fill the flavours seen so far to a running
sum `s`, convolving each new flavour against the window `{0..c}`. It is obviously correct, but each
flavour spreads over up to `k` sums, so it is `O(n*k) ~ 4*10^12` at the limits — unusable except as a
brute-force oracle. The other route is closed-form inclusion-exclusion: the uncapped count is
stars-and-bars `C(n + k - 1, n - 1)`, and the cap is enforced by subtracting the tuples where some
flavour overflows. That is `O(n + k)` after the factorial precompute. I take the inclusion-exclusion
route and keep the DP to check it against.

Deriving the identity. Let `U = C(n + k - 1, n - 1)` be the uncapped count. For a fixed set `S` of
flavours, let `A_S` count the tuples where every flavour in `S` is forced to at least `c + 1` (the
others free). Substituting `x_i = y_i + (c + 1)` for `i in S` with `y_i >= 0` drops the budget by
`|S|*(c+1)`, so `A_S = C(n + (k - |S|*(c+1)) - 1, n - 1)`, depending only on `|S| = j`.
Inclusion-exclusion over the overflowing set gives

  `answer = sum_j (-1)^j * C(n, j) * C(n + (k - j*(c+1)) - 1, n - 1)`,

with `C(n, j)` counting the size-`j` sets and `(-1)^j` the standard sign.

Where the sum stops. The term at index `j` carries residual budget `r_j = k - j*(c+1)`. Once
`r_j < 0` no tuple can force those `j` flavours over the cap, so the term is `0`; and `r_j` strictly
decreases in `j`, so every later term vanishes too. Also `j` cannot exceed `n`. So I loop `j` upward
and break the first time `r_j < 0` or `j > n`. The largest binomial top I ever ask for is at `j = 0`,
namely `n + k - 1`, so a factorial table up to `n + k` suffices.

Sample check, `n = 3, k = 4, c = 2`, `c + 1 = 3`: `j = 0` gives `C(3,0)*C(6,2) = 15`; `j = 1` has
`r = 1`, `C(3,1)*C(3,2) = 9`; `j = 2` has `r = -2`, stop. `15 - 9 = 6`, matching the six listed
tuples.

The pitfall this problem actually invites. The factor `C(n + r - 1, n - 1)` is the stars-and-bars
count of nonnegative length-`n` tuples summing to `r`, but that formula is valid only for `n >= 1`.
The constraints allow `n = 0`, and there the correct count is `1` when `r = 0` (the empty tuple) and
`0` otherwise — whereas the binomial `C(r - 1, -1)` is `0` in either case. So at `n = 0, k = 0` the
naive formula returns `0` where the true answer is `1` (the empty box). It bites only when `n = 0`
and `k = 0`: at `n = 0, k = 3` the true answer is genuinely `0`, and the binomial also gives `0`
because `r != 0`. I route the stars-and-bars factor through an explicit `bars(m, r)` helper that
returns `(r == 0)` when `m == 0` and `C(m + r - 1, m - 1)` otherwise. With the fix, `n = k = c = 0`
runs only `j = 0` with `term = C(0,0) * bars(0,0) = 1` before `j = 1 > 0` breaks, giving `1`.

Overflow. Both factors of a term are reduced mod `p < 2^30`, so their product is `~2^60`, which does
fit in signed 64-bit — but only just, and the three-way product inside `C`
(`fact[a] * inv_fact[b] * inv_fact[a-b]`) stays in range only because of the intermediate `% MOD`.
That margin is thin enough that I cast every modular multiply to `__int128` rather than rely on it.

Edge cases the hidden tests probe fall out of the same loop: `k = 0` gives `1` everywhere (only
`j = 0` survives, `bars(n, 0) = 1`); `c = 0` makes `r_j = k - j` and the alternating sum collapses to
`1` iff `k = 0` else `0`; `k > n*c` is infeasible so the sum is `0`, while the loop still runs only
`j <= n` and stays fast even at `k = 4*10^6`; `c >= k` kills every `j >= 1` term (its residual is
already negative), leaving the plain `C(n + k - 1, n - 1)`; and `k` an exact multiple of `c + 1`
lands some `r_j` on `0`, exercising `bars(n, 0)`.

Verification. I compared against the `O(n*k)` DP on over a thousand random small instances (`n` up to
~12, `k` up to ~20, `c` up to ~10) plus every corner above. Before the `bars` fix the only mismatches
were the `n = 0, k = 0` family; after it, none. The full module — the `O(n + k)` factorial precompute
up to `n + k`, the `C` and `bars` helpers, and the early-break inclusion-exclusion loop — is in the
answer.
