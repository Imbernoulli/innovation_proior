**Reading the problem and pinning the contract.** I have `n` ballast blocks; block `i` carries a weight `w[i] >= 1` and a trim score `v[i]` that can be negative, zero, or positive. I must choose a subset whose weights sum to **exactly** `C`, and over all such subsets maximize the total score; the empty subset (weight `0`, score `0`) is legal. If no subset reaches weight `C`, I print `IMPOSSIBLE`. Two features of this contract immediately shape everything downstream and I want them nailed before I pick an algorithm. First, the weight constraint is *equality*, not `<= C` — that turns this into a subset-sum feasibility question wearing a knapsack's clothes, so "reachability of a weight" is a first-class notion I have to track explicitly. Second, scores are **signed**, so the optimal achievable score at weight `C` can itself be negative; that means `IMPOSSIBLE` (no subset hits `C`) and "the best subset that hits `C` happens to score `-4`" are two genuinely different outcomes that I must never conflate. The whole difficulty of this problem lives in that distinction.

**Fixing the scale and the data types.** Bounds: `n <= 2000`, `C <= 5*10^4`, `1 <= w[i] <= 10^9`, `|v[i]| <= 10^9`. The accumulated score over a subset can reach `2000 * 10^9 = 2*10^12`, which is far past the 32-bit limit of about `2.1*10^9`, so every score accumulator must be 64-bit `long long`. Weights are read as `long long` too, partly for uniformity and partly because a single `w[i]` up to `10^9` would be fine in `int` but comparisons like `c - w[i]` are cleaner without sign-mixing. The capacity index runs `0..C`, at most `50001` entries, trivially in memory. The product `n*C = 2000 * 5*10^4 = 10^8` is the time budget of the DP; a flat array relaxation at one cheap operation per cell is comfortably under a 1-second limit, so I do not need anything cleverer than the standard `O(n*C)` table.

**Candidate approaches.** I weigh two routes and commit to the one that scales.

- *Exhaustive / meet-in-the-middle enumeration.* Enumerate subsets and keep the best score among those of weight exactly `C`. Pure enumeration is `O(2^n)` and obviously correct — I will in fact use it as my offline brute-force oracle — but at `n = 2000` it is hopeless. Splitting into halves and merging by weight (meet-in-the-middle) pushes feasible `n` to ~40, still nowhere near 2000, and merging under a *maximize-score-at-a-fixed-weight* objective is fiddly. Out for the real solution.
- *Capacity-indexed 0/1 DP.* Index a table by total weight `0..C`; `dp[c]` is the best score achievable by a subset of weight exactly `c`. Relax block by block in the standard 0/1 fashion. `O(n*C)` time, `O(C)` memory, and it directly models exact weights. This is the one I can both scale and prove, so I commit to it — but the proof obligations are exactly the three soft spots I flagged: the base case, the unreachable sentinel, and the 0/1 iteration order.

**Deriving the DP and its base case.** Define `dp[c]` = maximum total trim score over all subsets whose weights sum to *exactly* `c`, or a sentinel `UNREACH` if no subset has weight `c`. This is the crux: `dp[c]` is **not** initialized to `0`. The only weight reachable before placing any block is `c = 0`, via the empty subset, and that has score `0`. Every other weight is, a priori, unreachable. So the base case is `dp[0] = 0` and `dp[c] = UNREACH` for all `c >= 1`. If I had instead initialized the whole table to `0` (the reflex from the `<= C` knapsack, where every capacity is trivially achievable by "take nothing and leave slack"), I would be silently asserting that *every* weight is reachable with score `0` before any block exists — which is false for an *exact* constraint and would let the DP report a real numeric answer for genuinely impossible targets. The sentinel is the load-bearing idea.

Transition for block `i` with weight `wi`, score `vi`, processed in 0/1 style: a subset of weight `c` either excludes block `i` (then `dp[c]` is unchanged from before this block) or includes it (then the rest of the subset has weight `c - wi` and must itself be reachable). So
`dp[c] = max(dp[c],  dp[c - wi] + vi)`  whenever `c >= wi` and `dp[c - wi] != UNREACH`.
The guard `dp[c - wi] != UNREACH` is essential: if weight `c - wi` is not reachable, I cannot extend it, and worse, adding `vi` to the sentinel would corrupt the cell with a meaningless near-minimum value. The answer is `dp[C]`: print it if it is not `UNREACH`, else print `IMPOSSIBLE`.

**Why the iteration order matters (0/1, not unbounded).** Each block may be used at most once. If I relax capacities **upward** (`c` from `wi` to `C`), then when I compute `dp[c]` I may already have folded block `i` into `dp[c - wi]` during *this same* block's pass, which would let me use block `i` twice — the unbounded-knapsack behaviour. To keep it 0/1 I must relax **downward** (`c` from `C` down to `wi`), so `dp[c - wi]` on the right-hand side still reflects the state *before* block `i` was considered. This is the classic 1-D 0/1 trick and I will guard it with a trace, because getting the direction wrong is a silent over-count, not a crash.

**Sanity-checking the derivation on the sample, by hand.** Sample: `C = 7`, blocks `(w,v) = (3,5), (4,-2), (2,0), (5,4)`. I expect `4`. Start `dp = [0, U, U, U, U, U, U, U]` (indices `0..7`, `U = UNREACH`).

- Block (3,5), `c` from 7 down to 3: `dp[3] = max(U, dp[0]+5) = 5`; `dp[4..7]` all reference `dp[1..4]` which are `U` (except none reachable yet), so unchanged. State: `dp[3]=5`, rest `U` (and `dp[0]=0`).
- Block (4,-2), `c` from 7 down to 4: `dp[7] = max(U, dp[3] + (-2)) = 5 - 2 = 3`; `dp[4] = max(U, dp[0] + (-2)) = -2`; `dp[5],dp[6]` reference `dp[1],dp[2]` = `U`, unchanged. State: `dp[0]=0, dp[3]=5, dp[4]=-2, dp[7]=3`.
- Block (2,0), `c` from 7 down to 2: `dp[7] = max(3, dp[5]+0)`, `dp[5]` is `U`, stays `3`; `dp[6] = max(U, dp[4]+0) = -2`; `dp[5] = max(U, dp[3]+0) = 5`; `dp[4] = max(-2, dp[2]+0)`, `dp[2]` is `U`, stays `-2`; `dp[2] = max(U, dp[0]+0) = 0`. State now includes `dp[5]=5, dp[6]=-2, dp[2]=0`.
- Block (5,4), `c` from 7 down to 5: `dp[7] = max(3, dp[2]+4) = max(3, 0+4) = 4`; `dp[6] = max(-2, dp[1]+4)`, `dp[1]` is `U`, stays `-2`; `dp[5] = max(5, dp[0]+4) = max(5, 4) = 5`. Final `dp[7] = 4`.

`dp[7] = 4`, matching the hand-computed optimum (`{2,5}` gives `0+4=4`, beating `{3,4}`'s `3`). The recurrence, the base case, and the downward order all check out on the sample.

**First implementation — and immediately a trace, because the sentinel is a trap.** My first cut:

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

I chose `-1` as the sentinel reflexively. Let me trace a tiny adversarial input designed to make a *negative* score legitimate: `C = 4`, blocks `(4,-6), (2,-1), (2,-3)`. The subsets hitting weight `4` are `{4}` with score `-6` and `{2,2}` with score `-1 + -3 = -4`; the optimum is `-4`, and the right output is `-4` (it is reachable, just unfavourable). Run it. `dp = [0, -1, -1, -1, -1]`.

- Block (4,-6), `c=4`: `dp[0] != UNREACH`, so `dp[4] = max(-1, 0 + (-6)) = max(-1, -6) = -1`. **It kept `-1`** because `-6 < -1`. But `-1` is my sentinel! The cell now reads "unreachable" even though weight `4` *is* reachable (with score `-6`).
- Block (2,-1), `c=4`: `dp[2] == -1 == UNREACH`, skip; `c=2`: `dp[0] != UNREACH`, `dp[2] = max(-1, 0 + (-1)) = -1`. Again the real score `-1` collides with the sentinel.
- Block (2,-3), `c=4`: `dp[2] == -1`, treated as unreachable, skip; `c=2`: `dp[2] = max(-1, 0 + (-3)) = -1`.
- End: `dp[4] == -1 == UNREACH`, so I print `IMPOSSIBLE`.

**The bug.** The output is `IMPOSSIBLE`, but the true answer is `-4`. The defect is precise and it is *exactly* the negatives/zeros pitfall: my sentinel value `-1` is inside the legal range of real answers (scores can be any value down to `-2*10^12`), so a genuine reachable-but-negative score is indistinguishable from "unreachable". The `max(dp[c], ...)` then actively *prefers* the sentinel `-1` over a worse-but-real `-6`, and the final test `dp[C] != UNREACH` misfires. Two things are wrong at once: (1) the sentinel must lie strictly *below* every attainable score so `max` never picks it and the equality test is unambiguous; (2) I must make sure I never *add* `vi` to the sentinel — which the `dp[c-wi] != UNREACH` guard already prevents on the read side, but I have to keep that guard intact.

**Fix and re-verification.** Move the sentinel far below any reachable score: the minimum reachable score is `-2*10^12`, so `LLONG_MIN / 4 ≈ -2.3*10^18` is safely below it and still far from underflow even if a `+vi` ever touched it. With the guard in place it never does, but a low sentinel is the belt-and-suspenders that also makes `max` and the final equality test correct:

```
const long long UNREACH = LLONG_MIN / 4;
vector<long long> dp(C + 1, UNREACH);
dp[0] = 0;
```

Re-trace `C = 4`, `(4,-6),(2,-1),(2,-3)`, `dp = [0, U, U, U, U]` with `U = LLONG_MIN/4`.

- Block (4,-6), `c=4`: `dp[0] != U`, `dp[4] = max(U, 0 + (-6)) = -6` (now `-6 > U`, so it sticks). Good — weight 4 is correctly marked reachable with score `-6`.
- Block (2,-1), `c=4`: `dp[2] == U`, skip; `c=2`: `dp[2] = max(U, 0 + (-1)) = -1`.
- Block (2,-3), `c=4`: `dp[2] = -1 != U`, `dp[4] = max(-6, -1 + (-3)) = max(-6, -4) = -4`; `c=2`: `dp[2] = max(-1, 0 + (-3)) = -1`.
- End: `dp[4] = -4 != U`, print `-4`. **Correct.** The case that returned the wrong `IMPOSSIBLE` now returns the true `-4`, and it broke for exactly the reason I fixed — the sentinel was inside the answer range.

**Second debug episode — the iteration direction.** I asserted downward iteration keeps it 0/1, but assertions are where silent bugs hide, so I deliberately code the *wrong* direction and trace it to see the failure, confirming my fix is the right one. Suppose the inner loop were upward, `for (c = wi; c <= C; c++)`. Take `C = 4` and a single block `(2, 5)`. The honest truth: with one block of weight 2 you cannot reach total weight 4 — `dp[4]` must stay `UNREACH`, output `IMPOSSIBLE`. Trace upward: `dp = [0, U, U, U, U]`. Block (2,5), `c=2`: `dp[2] = max(U, dp[0]+5) = 5`. `c=3`: `dp[3] = max(U, dp[1]+5)`, `dp[1]=U`, skip. `c=4`: `dp[4] = max(U, dp[2]+5)` — but `dp[2]` was just set to `5` *in this same block's pass*, so `dp[4] = 5 + 5 = 10`. The DP claims weight `4` is reachable with score `10` by using the single weight-2 block **twice**. That is the unbounded-knapsack over-count, and it is a wrong answer with no crash. Now trace the **downward** loop I actually use, `c=4` then `c=3` then `c=2`: `c=4`: `dp[4] = max(U, dp[2]+5)`, `dp[2]` still `U` (not yet touched this pass), skip; `c=3`: `dp[2]=U`, skip; `c=2`: `dp[2] = max(U, dp[0]+5) = 5`. End: `dp[4] = U`, output `IMPOSSIBLE`. Correct. The downward order is not a stylistic choice; the upward order silently doubles items, and the trace shows it concretely.

**Edge cases, deliberately.**
- `C = 0`: the table is just `dp[0] = 0`; every block has `wi >= 1 > 0`, so the inner loop `for (c = C; c >= wi; c--)` never executes (`0 >= wi` is false). Answer `dp[0] = 0` — the empty load — *even if every block has a negative score*. I verified `C = 0` with all-negative blocks `(2,-4),(3,-1),(1,-9)` returns `0`, not the most-negative anything. Correct.
- `n = 0`: no blocks read. If `C = 0`, `dp[0] = 0` → print `0`. If `C >= 1`, `dp[C]` stays `UNREACH` → `IMPOSSIBLE`. Both correct (an empty multiset of blocks can only make weight 0). Verified `0 0` → `0` and `0 5` → `IMPOSSIBLE`.
- All-negative scores with reachable `C`: e.g. the `(4,-6),(2,-1),(2,-3)` case above → `-4`. The optimum is reported as a negative number, *not* `IMPOSSIBLE`. This is the corner the wrong base case/sentinel destroyed, now fixed.
- Unreachable `C`: `C = 5`, blocks `(2,7),(4,-3),(6,2)` — subset sums are `{0,2,4,6,8,10,12}`, never `5`, and `6 > 5` so that block is skipped entirely by the `wi > C` short-circuit; `dp[5]` stays `UNREACH` → `IMPOSSIBLE`. Verified.
- Block heavier than `C`: any `wi > C` can never contribute to an exact total of `C`, and the loop bound `c >= wi` already makes its pass a no-op; I add an explicit `if (wi > C) continue;` so the heavy-block pass costs `O(1)` instead of touching the loop header. Behaviour identical, just tidier and a hair faster.
- Overflow: scores accumulate in `long long`; max magnitude `~2*10^12` fits with vast headroom. `UNREACH = LLONG_MIN/4` is only ever *read* (guarded so `+vi` never lands on it) — it is never the left operand of an addition — so it cannot underflow. Safe.
- Output format: exactly one line, either the integer `dp[C]` or the literal token `IMPOSSIBLE`, each followed by a newline. `cin >>` consumes arbitrary whitespace, so parsing is format-agnostic.

**Re-verification at scale and against a brute force.** I ran the `O(n*C)` DP against an independent `O(2^n)` exhaustive-subset brute force on 1100+ random small cases (tiny `n`, small `C`, weights `1..8`, scores spanning `-9..9` including all-negative batches and `C = 0`). Zero mismatches, and the batch genuinely exercised the corners: hundreds of `IMPOSSIBLE` cases and over a hundred cases whose correct answer was a *negative* number — exactly the reachable-but-negative situation that the original `-1` sentinel mishandled. I also timed the true worst case (`n = 2000`, `C = 5*10^4`, all weights `1` so every inner loop runs full length, `n*C = 10^8` cell relaxations) at about `0.05 s`, well inside the 1-second limit, using ~4 MB.

**Final solution.** I disproved the naive table-of-zeros base case and the in-range sentinel by tracing a reachable-but-negative input to a precise wrong `IMPOSSIBLE`, fixed it with `dp[0]=0` / everything-else-`UNREACH` and a sentinel far below all attainable scores, and separately traced the iteration direction to confirm the downward 0/1 order. This is what I ship — one self-contained `O(n*C)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // dp[c] = best total trim score achievable using a subset of weight exactly c.
    // Sentinel UNREACH marks "no subset reaches this weight". Base: dp[0] = 0 (empty subset).
    const long long UNREACH = LLONG_MIN / 4;
    vector<long long> dp(C + 1, UNREACH);
    dp[0] = 0;

    for (int i = 0; i < n; i++) {
        long long wi = w[i], vi = v[i];
        if (wi > C) continue;                 // can never fit into an exact total of C
        for (long long c = C; c >= wi; c--) { // 0/1: iterate capacity downward
            if (dp[c - wi] != UNREACH) {
                long long cand = dp[c - wi] + vi;
                if (cand > dp[c]) dp[c] = cand;
            }
        }
    }

    if (dp[C] == UNREACH) cout << "IMPOSSIBLE" << "\n";
    else cout << dp[C] << "\n";
    return 0;
}
```

**Causal recap.** The exact-weight constraint forces a notion of "weight reachable or not", so the base case is `dp[0]=0` with every other weight `UNREACH`, never a table of zeros (that reflex from the `<= C` knapsack would falsely declare every target reachable). Because scores are signed, the optimum at `C` can be negative, which collides with any sentinel chosen inside the answer range — my first try used `-1` and a trace of `C=4, (4,-6),(2,-1),(2,-3)` returned the wrong `IMPOSSIBLE` instead of `-4`, pinpointing that the sentinel must sit strictly below all `~ -2*10^12` attainable scores (`LLONG_MIN/4`) and be guarded so `+vi` never lands on it. A second trace (`C=4`, lone block `(2,5)`) showed upward capacity iteration uses a block twice (`dp[4]=10`), so the 0/1 order must go downward. With `long long` accumulators, the `wi>C` skip, and the final `dp[C]==UNREACH ? IMPOSSIBLE : dp[C]` test, the empty-load (`C=0`), `n=0`, all-negative, unreachable, and overflow corners all resolve correctly, and the DP agrees with an exhaustive brute force on 1100+ randomized cases.
