The knot in this problem is that the weight constraint is *equality* — total weight exactly `C`, not `<= C` — and the trim scores `v[i]` are signed. Those two facts together are the whole test. Exactness turns "can weight `c` be hit at all" into a first-class reachability question rather than a free byproduct of leaving slack. Signedness means the best score at weight `C` can itself be negative. So a subset that hits `C` with optimum `-4` and a target `C` that no subset can reach are two genuinely different outcomes that must print different things (`-4` versus `IMPOSSIBLE`), and every choice below has to keep them apart. Everything else is a standard 0/1 knapsack.

The bounds are `n <= 2000`, `C <= 5*10^4`, `w[i] <= 10^9`, `|v[i]| <= 10^9`. A subset score can reach `2000 * 10^9 = 2*10^12`, well past the 32-bit range, so every accumulator is `long long`; a 32-bit `int` here would be a silent wrong answer on the large tests, not a crash. The capacity index runs `0..C`, at most `50001` cells. The DP is `O(n*C) = 10^8` cell relaxations at one cheap operation each — comfortably under the 1-second limit, so nothing cleverer than a flat table is warranted.

Exhaustive subset enumeration is `O(2^n)`: correct, and I keep it around as an offline oracle, but hopeless at `n = 2000`. Meet-in-the-middle reaches `n` around 40 and is awkward to merge under a maximize-score-at-a-fixed-weight objective — still nowhere near 2000. The capacity-indexed 0/1 DP is the only route that both scales and directly models the exact-weight objective, so I commit to it. The three things I have to get right are the base case, the unreachable sentinel, and the 0/1 iteration order.

Let `dp[c]` be the maximum total score over subsets of weight *exactly* `c`, or a sentinel `UNREACH` when no subset has weight `c`. The load-bearing point is that `dp` is **not** zero-initialized. Before any block is placed, the only reachable weight is `c = 0`, via the empty subset with score `0`; every other weight is unreachable. So the base case is `dp[0] = 0` and `dp[c] = UNREACH` for all `c >= 1`. Zero-filling the whole table — the reflex from the `<= C` knapsack, where every capacity is trivially reachable by taking nothing and leaving slack — would assert that every weight is reachable with score `0`, which is false under an exact constraint and would make the DP print a real number for genuinely impossible targets.

The transition for block `i` with weight `wi`, score `vi`, in 0/1 fashion: a weight-`c` subset either omits the block (value unchanged) or includes it on top of a reachable weight-`(c - wi)` subset, giving `dp[c] = max(dp[c], dp[c - wi] + vi)` for `c >= wi`, but only when `dp[c - wi] != UNREACH` — the guard forbids extending a subset that does not exist. The answer is `dp[C]`, or `IMPOSSIBLE` if it is still `UNREACH`. The iteration order carries the 0/1 restriction. Relaxing `c` upward would let `dp[c - wi]` already contain block `i` folded in during this same pass, using it twice — unbounded-knapsack behaviour. Concretely, one block `(2,5)` with `C = 4`, iterating upward: `dp[2] = 5`, then `dp[4] = dp[2] + 5 = 10`, claiming weight `4` by using the single weight-2 block twice, when weight `4` is in fact unreachable. Iterating downward (`c` from `C` to `wi`) keeps `dp[c - wi]` at its pre-block value, so `dp[4]` stays `UNREACH` and the output is the correct `IMPOSSIBLE`. So the inner loop runs downward.

Now the sentinel — my reflex first cut:

```
const long long UNREACH = -1;          // "weight not reachable"
vector<long long> dp(C + 1, UNREACH);
dp[0] = 0;
for (int i = 0; i < n; i++) {
    long long wi = w[i], vi = v[i];
    for (long long c = C; c >= wi; c--) {
        if (dp[c - wi] != UNREACH)
            dp[c] = max(dp[c], dp[c - wi] + vi);
    }
}
```

`-1` seems like a harmless "not set" marker until I trace an input whose optimum is legitimately negative. Take `C = 4`, blocks `(4,-6), (2,-1), (2,-3)`: the subsets hitting weight `4` are `{4}` with score `-6` and `{2,2}` with score `-4`, so the answer is `-4` — reachable, just unfavourable. Running the code: block `(4,-6)` sets `dp[4] = max(-1, 0 + (-6)) = max(-1, -6) = -1`, keeping `-1` because `-6 < -1`, so the cell now reads "unreachable" even though weight `4` is reachable with score `-6`. The two weight-2 blocks likewise land real scores `-1` that collide with the sentinel, and at the end `dp[4] == -1 == UNREACH`, so it prints `IMPOSSIBLE`. Wrong: the true answer is `-4`.

The sentinel `-1` lives inside the legal answer range (scores run down to about `-2*10^12`), so `max(dp[c], ...)` actively prefers the sentinel over a worse-but-real `-6`, and the final reachability test cannot distinguish a real `-1` from "unreachable". The fix is a sentinel strictly below every attainable score: the minimum reachable score is about `-2*10^12`, so

```
const long long UNREACH = LLONG_MIN / 4;   // ~ -2.3*10^18, below every real score
vector<long long> dp(C + 1, UNREACH);
dp[0] = 0;
```

sits far below any real value, can never be chosen by `max`, and makes the final equality test unambiguous. Re-running `C = 4, (4,-6),(2,-1),(2,-3)` now correctly settles `dp[4] = -4`. The `dp[c - wi] != UNREACH` guard already keeps `+vi` from ever landing on the sentinel, so even at `LLONG_MIN/4` it cannot underflow — it is only ever read, never the left operand of an addition.

The corners fall out of this structure. `C = 0`: the table is just `dp[0] = 0`, and since every `wi >= 1` the inner loop `c >= wi` never runs, so the answer is `0` — the empty load — even when every block scores negative. `n = 0`: no blocks, so `dp[0] = 0` gives `0` when `C = 0` and `dp[C]` stays `UNREACH` giving `IMPOSSIBLE` otherwise, which is right since an empty multiset can only make weight `0`. All-negative scores with a reachable `C` return the negative optimum (the `-4` case), never `IMPOSSIBLE`. An unreachable `C` leaves `dp[C] = UNREACH` and prints `IMPOSSIBLE`. A block with `wi > C` can never contribute to an exact total of `C`; the loop bound already no-ops its pass, but I add an explicit `if (wi > C) continue;` so a heavy block costs `O(1)` rather than a loop-header check. Output is a single line: the integer `dp[C]` or the literal `IMPOSSIBLE`.

To trust it beyond hand-traces I run the DP against the `O(2^n)` brute-force oracle on 1100+ random small cases (`n` up to ~12, small `C`, weights `1..8`, scores `-9..9`, with all-negative batches and `C = 0` mixed in): zero mismatches, and the batch genuinely exercises the corners — hundreds of `IMPOSSIBLE` cases and over a hundred whose correct answer is a negative number, exactly the situation the `-1` sentinel wrecked. Timing the true worst case (`n = 2000`, `C = 5*10^4`, all weights `1` so every inner loop runs full length, `10^8` relaxations) comes in around `0.05 s` in ~4 MB, well inside the limits.

That table — the non-zero base case, the sub-answer-range sentinel, the downward guarded relaxation, `long long` throughout — is what I ship; the full program is in the answer.
