**Reading the problem and pinning the contract.** I have `n` items, each with an integer price `p[i] >= 1` and an integer joy `v[i]` that may be negative. I buy each item at most once, and I must choose a subset whose **total price** `s` lands in the band `[L, R]` with **both ends inclusive** (`L <= s <= R`); among all such subsets I maximize the total joy, and if no subset's price falls in the band I print `IMPOSSIBLE`. The empty subset has price `0` and joy `0`, so it is a legal purchase exactly when `0` is inside `[L, R]`, i.e. when `L = 0`. Before touching an algorithm I fix the scale because it forces the data types: `n <= 2000`, `R <= 10^5`, and `|v[i]| <= 10^9`. A subset can contain up to `2000` items each with joy near `10^9`, so the total joy can reach `2*10^3 * 10^9 = 2*10^12`. That is far outside 32-bit range, so every joy accumulator must be `long long`. Prices stay small (`s <= R <= 10^5`), but I will carry them as `long long` too just to keep the comparisons uniform. That is decision one and it is non-negotiable: an `int` joy is a silent wrong answer on the large tests.

**Laying out the candidate approaches.** This is a 0/1 knapsack, but the feasible region is a *band* of total prices, not a single cap. I want the formulation I can read the band off cleanly.

- *"At most s" states.* Define `dp[s]` = best joy using a subset of total price `<= s`. This is the textbook capacity knapsack. But it bakes the upper bound into the state and throws away the exact price, so I cannot ask "is the price at least `L`?" afterwards — the lower edge of the band is lost. I would have to do inclusion–exclusion between two capacities, and with negative joys "best at price `<= s`" is not even monotone in a way that lets me subtract cleanly. Too fragile.
- *"Exactly s" states.* Define `dp[s]` = best joy using a subset whose total price is **exactly** `s`, for `s` in `0..R`. Items with `p[i] > R` can never fit and are dropped. Then the answer is simply `max(dp[s])` over `s` in the band `[L, R]`, and `IMPOSSIBLE` is "every `dp[s]` in that range is unreachable." This keeps the price visible, so the band query is a plain scan over the eligible indices. I commit to exact-price states.

**Deriving the DP.** With exact-price states, `dp[s]` is the best joy of a subset summing to price exactly `s`. Base: before any item, the only reachable price is `0` (the empty subset), with joy `0`; every other price is unreachable. I model "unreachable" with a sentinel `NEG` far below any real joy. Transition for item `i` with price `pi`, joy `vi`: a subset reaching price `s` either excludes `i` (joy stays `dp[s]`) or includes `i`, which means the rest reached price `s - pi`, giving `dp[s - pi] + vi`. So `dp_new[s] = max(dp_old[s], dp_old[s - pi] + vi)` for `s >= pi`. This is exactly 0/1 knapsack, and the standard space-saving trick is to keep one array and iterate `s` **downward** from `R` to `pi`, so that when I read `dp[s - pi]` it still holds the *old* (item-`i`-excluded) value and item `i` is used at most once.

The window endpoints `L` and `R` do **not** appear in this recurrence. They appear in two other places, and both are inclusive-boundary decisions:
- The array spans prices `0` through `R` inclusive, which is `R + 1` cells, indices `0..R`.
- The answer scans `s = L, L+1, ..., R` inclusive on both ends.

**Numeric self-check of the recurrence on the sample.** Sample: items (1-indexed) prices/joys `(4,3),(5,4),(6,5),(7,-1),(3,2)`, band `[10, 12]`; claimed answer `9`. Let me verify the recurrence reaches `9` and that `9` is genuinely optimal in the band, rather than trusting the claim. I will track only the prices that matter. Reachable prices in `[10,12]` and their best joys:
- price `10`: `{4,6}` -> joy `3+5=8`; `{7,3}` -> joy `-1+2=1`. Best `8`.
- price `11`: `{5,6}` -> joy `4+5=9`; `{4,7}` -> joy `3-1=2`. Best `9`.
- price `12`: `{4,5,3}` -> joy `3+4+2=9`; `{5,7}` -> joy `4-1=3`. Best `9`.
So `max over band = max(8,9,9) = 9`. The recurrence's exact-price states reproduce the hand answer, and I have now actually exhibited the optimal subset (`{5,6}` at price `11`, joy `9`) rather than asserting it. Good — the model is right; now I have to transcribe it without an off-by-one, which is where this kind of problem dies.

**First implementation — and immediately a trace, because the window edges are exactly where I will slip.** My first cut:

```
const long long NEG = LLONG_MIN / 4;
vector<long long> dp(R, NEG);          // (A) prices 0..R
dp[0] = 0;
for (int i = 0; i < n; i++) {
    long long pi = p[i], vi = v[i];
    if (pi > R) continue;
    for (long long s = R; s >= pi; s--)
        if (dp[s - pi] != NEG) dp[s] = max(dp[s], dp[s - pi] + vi);
}
long long best = NEG;
for (long long s = L; s < R; s++)      // (B) scan band
    best = max(best, dp[s]);
if (best == NEG) cout << "IMPOSSIBLE\n"; else cout << best << "\n";
```

Two boundary spots smell wrong: line (A) sized the array `R`, and line (B) scanned `s < R`. I trace the smallest input that puts the optimum *at the top edge* of the band, because that is what an exclusive upper bound would drop. Take a single item priced exactly at `R`: input `n=1, L=0, R=4`, item `(4, 7)`. The only nonempty in-band subset is `{item}` at price `4 = R`, joy `7`; the empty subset is price `0`, joy `0`, also in-band since `L=0`. So the true answer is `max(0, 7) = 7`.

Run my code on it. Line (A): `dp` has size `R = 4`, so valid indices are `0,1,2,3` — there **is no `dp[4]`**. The item loop runs `s` from `R = 4` down to `pi = 4`, so it executes `s = 4` and writes `dp[4 - 4] -> dp[4] = max(dp[4], dp[0] + 7)`. But `dp[4]` is **out of bounds** (size 4). That is undefined behaviour — a heap overwrite. Even setting that aside, the answer scan (B) runs `s = L=0 .. < R=4`, i.e. `s = 0,1,2,3`, and never looks at `s = 4`. So the price-`4` subset is invisible regardless.

**Diagnosing the first bug (array size, the `R` vs `R+1` off-by-one).** Prices range over `0..R` **inclusive**, which is `R + 1` distinct values, so the array must have `R + 1` cells, not `R`. Sizing it `R` drops the single most important index — the cap itself — and turns the descending knapsack loop's `s = R` write into out-of-bounds UB. Fix: `vector<long long> dp(R + 1, NEG)`.

**Diagnosing the second bug (window upper edge, inclusive vs exclusive).** The answer scan `for (s = L; s < R; s++)` stops at `R - 1`, excluding `s = R`. But the band is `[L, R]` with `R` **included**. So any optimum whose total price is exactly `R` is silently discarded. The two bugs conspire on my trace: even after I fix the array size, `s < R` would still miss the price-`4` answer and print `0` instead of `7`. Fix: scan `s <= R`.

**Fixing and re-verifying on the trace.** Corrected loops: `dp(R + 1, NEG)` and `for (s = L; s <= R; s++)`. Re-run `n=1, L=0, R=4`, item `(4,7)`. `dp` size `5`, `dp[0]=0`, rest `NEG`. Item: `pi=4 <= R`; loop `s=4`: `dp[0] != NEG`, so `dp[4] = max(NEG, 0 + 7) = 7`. Answer scan `s = 0..4`: `dp[0]=0`, `dp[4]=7`, others `NEG`; `best = 7`. Prints `7`. Correct, and it now reads `dp[4]` legally because the array has index `4`. The case that broke fails no longer, and it failed for exactly the two reasons I fixed — that is the evidence I trust.

**A second trace, this time on the *lower* edge `L`, because inclusive/exclusive can bite there too.** Consider whether I should have written the scan `s > L` (a different off-by-one). Construct an input whose only in-band subset sits exactly at `s = L`: `n=2, L=5, R=9`, items `(5, 4)` and `(3, -2)`. Subsets and prices: `{}` -> `0` (out, since `L=5`); `{a}` -> `5` (in band, exactly `L`), joy `4`; `{b}` -> `3` (out); `{a,b}` -> `8` (in band), joy `4 + (-2) = 2`. True answer `max(4, 2) = 4`, achieved at price `5 = L`. Run the corrected code: `dp` size `10`, `dp[0]=0`. Item `a` (5,4): `s=5` -> `dp[5]=0+4=4`. Item `b` (3,-2): descending `s=9..3`: at `s=8`, `dp[5] != NEG` so `dp[8] = max(NEG, dp[5] + (-2)) = 2`; at `s=3`, `dp[0] != NEG` so `dp[3] = 0 + (-2) = -2`. Now scan `s = L=5 .. R=9` inclusive: `dp[5]=4`, `dp[8]=2`, rest in range `NEG`; `best = 4`. Prints `4`. Correct. Had I written `s > L` (i.e. started the scan at `L + 1 = 6`), I would have skipped `dp[5]=4` and reported `2` — wrong. So the lower bound must be **inclusive** (start at `L`, not `L + 1`), confirmed by a concrete case where the optimum lives precisely at `s = L`. This also exercised a negative-joy item (`b`) being deliberately added: here it only ever lowered joy, so the optimizer correctly preferred the price-`5` subset over the price-`8` one.

**Edge cases, deliberately, because this is where boundary code dies.**
- `n = 0`: no items. `dp` size `R + 1`, only `dp[0] = 0` reachable. If `L = 0`, the scan hits `dp[0] = 0` and prints `0` (empty purchase legal). If `L > 0`, the scan over `[L, R]` sees only `NEG`, so it prints `IMPOSSIBLE` — correct, since the empty subset (price `0`) is the only option and it is below the band.
- `L = 0, R = 0`: array size `1`, `dp[0] = 0`, scan `s = 0..0` gives `0`. The empty subset, joy `0` — correct.
- Single item with `p > R`: e.g. `n=1, L=0, R=3`, item `(5, 100)`. The `if (pi > R) continue;` drops it; only `dp[0]=0` survives; with `L=0` prints `0`, with `L>0` prints `IMPOSSIBLE`. Correct — that item can never be part of an in-band purchase. (Without the guard, the descending loop `s = R..pi` would not execute since `R < pi`, so it is harmless either way, but the guard makes intent explicit and avoids a needless pass.)
- All-negative joys, in-band subset forced: e.g. `n=2, L=4, R=10`, items `(4,-3),(6,-5)`. Reachable in band: price `4` joy `-3`, price `6` joy `-5`, price `10` joy `-8`. Best is `-3` (the least-bad in-band subset). My code: `dp[4]=-3`, `dp[6]=-5`, `dp[10]=-8`; scan `[4,10]` gives `max(-3,-5,-8) = -3`. Correct — the answer can legitimately be negative, and because `NEG = LLONG_MIN/4` is far below any real joy near `-2*10^12`, no reachable negative joy is ever mistaken for "unreachable."
- No in-band subset at all: e.g. `n=1, L=2, R=2`, item `(5, 9)`. Reachable prices are `0` and `5`; neither is in `[2,2]`. Scan sees only `NEG`; prints `IMPOSSIBLE`. Correct.
- Overflow: joy accumulators are `long long`; max magnitude `~2*10^12` fits with room. The sentinel `NEG = LLONG_MIN/4` is only ever *read* after the `dp[s-pi] != NEG` guard, so I never compute `NEG + vi` and never underflow. Safe.
- Sentinel separation: the largest possible true (negative) joy magnitude is about `2*10^12`, while `|NEG| = LLONG_MIN/4 ~ 2.3*10^18`. The gap is six orders of magnitude, so a real joy can never collide with the sentinel — the `!= NEG` test is unambiguous.

**Numeric self-check that the array bound is exactly right, not off by one in the other direction.** I claimed `R + 1` cells indexed `0..R`. Check the largest write: the descending loop writes `dp[s]` for `s` up to `R`, and reads `dp[s - pi]` for `s - pi` down to `0`. Highest index touched is `R`, which exists in a size-`R+1` array (indices `0..R`); lowest is `0`. No index `R + 1` or `-1` is ever touched. And the answer scan touches `dp[L..R]`, all within `0..R` since `0 <= L <= R`. So `R + 1` is the exact size — one more would waste a cell, one fewer is the UB I already hit. Confirmed by counting, not by feel.

**Final solution.** I convinced myself the *idea* is right by ruling out "at most s" states and numerically reproducing the sample's optimum at exact prices; I convinced myself the *code* is right by tracing two boundary cases — an optimum sitting exactly on the upper edge `R` (which exposed both the size-`R` array UB and the `s < R` exclusive scan) and an optimum sitting exactly on the lower edge `L` (which confirmed the scan must start at `L`, not `L + 1`) — and re-verifying every corner. What I ship is one self-contained file: the exact-price 0/1 knapsack with an array of `R + 1` cells and an inclusive `[L, R]` answer scan.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;

    vector<long long> p(n), v(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> v[i];

    const long long NEG = LLONG_MIN / 4;
    // dp[s] = best total joy using a subset whose total price is EXACTLY s,
    // for s in [0, R] (indices 0..R inclusive, so size R+1).
    vector<long long> dp(R + 1, NEG);
    dp[0] = 0;                                   // empty subset: price 0, joy 0

    for (int i = 0; i < n; i++) {
        long long pi = p[i], vi = v[i];
        if (pi > R) continue;                    // cannot ever fit within the cap
        for (long long s = R; s >= pi; s--) {    // 0/1 knapsack: descend so each item used once
            if (dp[s - pi] != NEG)
                dp[s] = max(dp[s], dp[s - pi] + vi);
        }
    }

    long long best = NEG;
    for (long long s = L; s <= R; s++)           // window [L, R] INCLUSIVE on both ends
        best = max(best, dp[s]);

    if (best == NEG) cout << "IMPOSSIBLE\n";
    else cout << best << "\n";
    return 0;
}
```

**Causal recap.** Modeling the feasible region as a band rather than a single cap pushed me to *exact-price* states `dp[s]` so the band could be read off as a plain scan, which is correct but concentrates all the risk at the two window endpoints; my first cut made both classic off-by-one slips — it sized `dp` as `R` (dropping index `R`, the cap, into out-of-bounds UB) and scanned the answer as `s < R` (excluding the inclusive upper edge), so a traced single item priced exactly at `R` exposed both at once and is fixed by `R + 1` cells and `s <= R`; a second trace with the optimum sitting exactly at the lower edge `L` confirmed the scan must begin at `L` (not `L + 1`); and a `NEG` sentinel that is only ever read behind a reachability guard, with `long long` joys, closes the `IMPOSSIBLE`, empty-subset, all-negative, `p > R`, and overflow corners.
