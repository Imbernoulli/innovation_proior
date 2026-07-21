This is the unbounded make-change objective wearing a postage-meter costume: unlimited supply of each of `n <= 100` denominations `d[0..n-1]`, and I want the fewest stamps summing to *exactly* `A <= 10^5`, else `-1`. The scale fixes the toolkit at a glance — `n * A` is at most `10^7`, so an `O(n*A)` table over the amounts `0..A` is essentially free, and every stamp is worth at least `1`, so the count is bounded by `A` and never troubles even 32 bits (I carry `long long` anyway, since a wider accumulator costs nothing here). Two corners are already visible from the statement: `A = 0`, where the empty multiset sums to `0` and the answer is `0`, not `-1`; and outright impossibility — say every denomination even but `A` odd — which must print `-1`.

The tempting solution is the trap the problem is built around, so I won't ship it without trying to break it: sort denominations descending, repeatedly grab the largest stamp `<= remaining`, subtract, repeat. On `d = [1, 3, 4]`, `A = 6` greedy grabs `4`, then can only fill the remaining `2` with `1 + 1` — total `3` stamps — but `3 + 3` hits `6` in `2`. The large stamp was a local lure that stranded a remainder the set fills poorly. Not a fluke of one instance: `d = [1, 5, 8]`, `A = 10` does the same, greedy `8 + 1 + 1 = 3` versus `5 + 5 = 2`. Greedy is optimal only on canonical denomination systems, and the statement promises no canonicity, so it is out.

So I want `dp[v]`, the minimum number of stamps summing to exactly `v`. Any optimal multiset for `v` ends with some stamp `d[i] <= v`; strip it, and what remains is an optimal multiset for `v - d[i]` — otherwise I could improve the remainder and beat the whole. Hence

    dp[v] = 1 + min over i with d[i] <= v of dp[v - d[i]],

with base case `dp[0] = 0`, the empty multiset — which is also exactly what hands `A = 0` its answer for free. Filling `v` upward keeps every predecessor `v - d[i] < v` final before it is used. Unreachable amounts keep a sentinel `INF`, and `dp[A] == INF` prints `-1`. Filling the sample by hand, `dp[1..6]` climbs `1, 2, 1, 1, 2, 2`, so `dp[6] = 2` via `3 + 3` — the optimum greedy's `3` overshot.

The transition is meaningful only through a *legal, reachable* predecessor, and both halves of that bite at this scale. If I relax unconditionally — `dp[v] = min(dp[v], dp[v - d[i]] + 1)` for every `i` — then on `d = [4]`, `A = 2` the first iteration reads `dp[1 - 4] = dp[-3]`, an out-of-bounds negative index: undefined behavior, silent corruption or a crash. And even with the index guarded, adding `1` to an `INF` predecessor fabricates a finite count for a genuinely unreachable amount — and with a sentinel near the top of `long long`, `INF + 1` overflows. Both die if I guard the relaxation before touching the predecessor:

```
if (d[i] <= v && dp[v - d[i]] != INF) {
    dp[v] = min(dp[v], dp[v - d[i]] + 1);
}
```

The short-circuit `&&` guarantees `dp[v - d[i]]` is never even read when `d[i] > v`. Re-running `d = [4]`, `A = 2`: both `v = 1, 2` skip (`4 > v`), `dp[2]` stays `INF`, output `-1` — correct, `2` is not a multiple of `4`.

The remaining corners fall out of the same structure. `A = 0`: the table is a single cell `dp[0] = 0`, the fill loop never runs, output `0` — cleanly distinct from `-1`. Sizing `dp` as `A + 1` keeps `dp[A]` in range (for `A = 0` it is that lone cell). Parity impossibility (`d = [2, 4, 6]`, `A = 5`) leaves `dp[5] = INF` -> `-1`; a denomination larger than `A` never passes `d[i] <= v` and is silently ignored, no special case needed. At the max scale `A = 10^5`, `n = 100`, the table is ~0.8 MB and the fill is `<= 10^7` relaxations, milliseconds under the 1 s limit; counts never exceed `A`, so `cand` stays far inside `long long`.

That is what I ship — the `O(n*A)` DP I can defend, not the greedy I broke. `cin >>` skips whitespace, so the two-line input parses format-agnostically, and `if (!(cin >> n >> A)) return 0;` covers an empty stream. The full program is in the answer.
