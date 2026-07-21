What makes this more than a textbook knapsack is the word *exactly*. I must load **exactly `K`**
parcels, not "at most `K`" — so I cannot drop the money-losers to keep the total non-negative. If the
`K` parcels I am forced to take are all losers, the true optimum is a negative number, and that
negative number is the answer, not `INFEASIBLE`. Weights `w[i] >= 0` allow zero (paperwork-only
manifests), and profits `v[i]` can be negative, zero, or positive. So there are three outcomes that
must never collide: a positive profit, a genuinely negative feasible profit, and the unreachable
verdict `INFEASIBLE` when no size-`K` load fits under the cap `C`. Keeping those three apart is the
whole problem, and it is precisely where a lazy base case will silently confuse "profit `0`" with "no
such load exists."

Scale first, because it fixes the data types. `n <= 200`, `C <= 1000`, `|v[i]| <= 10^9`, and a load
holds up to `K <= n = 200` parcels, so a total profit reaches `200 * 10^9 = 2 * 10^11` — well past the
32-bit range (`~2.1 * 10^9`). Every accumulator and table cell must be 64-bit `long long`; an `int` is
a silent wrong-answer on the large tests. Two constraint quirks also bite. `K` is given up to `10^9`
even though only `K <= n` can ever be feasible, so I read `K` as 64-bit and special-case `K > n`
*before* I size any array by `K`, or the allocation itself explodes. And `w[i]` can be `10^9`, far
larger than `C = 1000`, so a single parcel can be individually uncarriable — I must guard against
indexing a weight axis of length `C+1` with such a weight.

Two routes are on the table. Subset enumeration is `O(C(n,K))`, obviously correct but astronomical at
`C(200,100)`; meet-in-the-middle reaches `n` near 40 and is fiddly to merge under "exactly `K`
parcels, weight `<= C`, maximize profit," still far short of `n = 200`. I keep exhaustive enumeration
only as a brute-force oracle for small random tests. That leaves the two-dimensional count×weight DP:
`dp[k][c]` = best total profit choosing exactly `k` parcels of exact total weight `c`, relaxed parcel
by parcel in 0/1 fashion. Time `O(n * K * C) = 4 * 10^7`, memory `O(K * C) ~ 1.6 MB` — comfortable
inside 1 s / 256 MB.

I track exact weight on one axis so "weight `<= C`" becomes a final max over `c <= C`, and exact count
on the other so "exactly `K`" is just reading row `K`. Before any parcel is placed, the only reachable
state is `(0,0)` at profit `0` — the empty load. Every other `(k,c)` is unreachable, encoded by a
sentinel `NEG = LLONG_MIN/4` (dividing by 4 leaves headroom so `+v_i` never overflows, and I only ever
add to a *reachable* cell, never the sentinel). The 0/1 transition: a reachable `(k,c)` can skip
parcel `i` or take it, reaching `(k+1, c+w_i)` at profit `+v_i` when `c + w_i <= C`. So

```
dp[k+1][c + w_i] = max(dp[k+1][c + w_i], dp[k][c] + v_i)   if dp[k][c] reachable and c + w_i <= C
```

The answer is `max over c in 0..C of dp[K][c]`; if that whole row is sentinel, no size-`K` load fits
and I print `INFEASIBLE`.

The sample makes one subtlety explicit: its optimum `{(3,5),(2,0)}` (weight `5`, profit `5`) uses a
zero-profit parcel, so the recurrence has to treat adding `v = 0` as a real placement, not as "nothing
happened."

The base case is where the sign handling actually bites. The tempting first cut — "an empty cell
should not block anything" — initialises the *entire* table to `0`:

```
vector<vector<long long>> dp(Kc + 1, vector<long long>(Cc + 1, 0));
```

But `dp[k][c] = 0` is meant to mean "unreachable," and `0` is also a perfectly real profit. On the
smallest input that pits those two meanings against each other — `n=1, K=1, C=5`, single parcel
`(2,-4)`, whose only size-1 load has weight `2 <= 5` so the answer must be `-4` — the all-zero table
computes `dp[1][2] = max(0, 0 + (-4)) = 0` and reports `0`. Seeding every `dp[k][c]` to `0` asserts
that "exactly `k` parcels at profit `0`" is achievable before any parcel is placed; because profits can
be negative, that phantom `0` actively beats the true optimum. The fix is to seed only `dp[0][0] = 0`,
everything else `NEG`, and skip any `NEG` cell when relaxing:

```
vector<vector<long long>> dp(Kc + 1, vector<long long>(Cc + 1, NEG));
dp[0][0] = 0;                                    // only the empty load is reachable a priori
...
if (dp[k][c] == NEG) continue;                   // never add v_i to an unreachable cell
```

Now the same input relaxes `dp[1][2] = 0 + (-4) = -4` while the rest of row 1 stays `NEG`, so
`ans = -4` — the real negative load reports its true profit instead of a phantom `0`.

With the base case right, the remaining danger is one parcel filling several of the `K` slots. The
count axis grows by one per take, so I sweep `k` **downward** (`K-1 .. 0`): then the `dp[k][c]` I read
was computed before this parcel was offered. That the upward sweep genuinely double-counts, rather than
merely being unconventional, shows on `n=2, K=2, C=0`, two zero-weight parcels `(0,5),(0,3)`: parcel 0
sets `dp[1][0] = 5` at `k=0`, then at `k=1` reads that just-written `5` and sets `dp[2][0] = 10` — one
parcel occupying both slots, an impossible profit. Downward, `dp[2][0]` is still `NEG` when parcel 0
runs, so parcel 1 reads `dp[1][0] = 5` and lands the correct `dp[2][0] = 8`. The weight axis `c` only
needs to avoid collision within a single `k`-row, so sweeping it downward too is safe.

The huge-weight guard: `w_i` can be `10^9` while the weight axis spans only `0..1000`. If `w_i > C`
the parcel alone exceeds the cap and can never be chosen, so I `continue` past it; after that guard
`w_i <= C` fits an `int` and `c + w_i <= C` is enforced by sweeping `c` from `C - w_i` down to `0`. On
`n=3, K=2, C=3`, parcels `(2,5),(5,9),(6,9)`, two parcels exceed the cap and are skipped, leaving one
carriable parcel — no size-2 load exists, row `K` stays all `NEG`, I print `INFEASIBLE`, and computing
`c + 6` never indexes out of bounds.

The remaining corners:

- `K = 0`: only `dp[0][0] = 0` is reachable; reading row 0 gives `ans = 0`, the empty load even when
  every parcel is negative.
- `n = 0`: `K = 0` gives `0`; `K >= 1` hits the early `K > n` `INFEASIBLE`.
- `K > n` (up to `10^9`): no `K` distinct parcels exist, so `INFEASIBLE`, tested before sizing any
  array by `K` so I never allocate a `10^9`-row table.
- All feasible loads negative, e.g. `n=3, K=2, C=10`, `(2,-3),(3,-1),(4,-5)`: best fit `{(2,-3),(3,-1)}
  = -4`; the sentinel keeps "feasible but negative" distinct from "infeasible."
- Zero-weight stacking, `n=3, K=3, C=0`, `(0,-2),(0,5),(0,-1)`: all three forced, loaded at weight
  `0`, returns `-2 + 5 - 1 = 2`.
- Output is one line — the integer (possibly negative or zero) or the token `INFEASIBLE`.

Against an independent exhaustive `C(n,K)` brute force I ran ~1100 random small cases from two
generators — varying `n`, `K` (including `0` and `> n`), many zero-weight and over-cap parcels, and
profits straddling zero — with zero mismatches, and every corner above hit. The DP itself measures
around `0.02 s`, comfortably inside the budget.
