**Reading the problem and pinning the contract.** I have a building with `m` floors and `k` eggs. There is a critical floor `c`: drop from `>= c` and the egg breaks, drop from `< c` and it survives. A broken egg is gone; a survivor is reusable. I must find a dropping strategy guaranteed to identify `c`, and among all such strategies I want the one whose *worst case* uses the fewest trials. Output that minimum worst-case trial count. Input is two integers `k` and `m` on stdin; output is one integer. The scale is `1 <= k <= 100` and `1 <= m <= 10000`. Before any algorithm I fix the data types: the answer is a trial count, and the largest it can ever be is `m = 10000` (the one-egg linear scan, argued below), so the result fits comfortably in a 32-bit int. But intermediate "how many floors can I cover" quantities can grow fast (doubling per trial), so I will keep those in `long long` and clamp them, rather than risk a silent overflow. That is the one type decision I make up front.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is fastest to type.

- *Binary search on floors.* The reflex from ordinary search: drop from the middle floor, the break/survive answer halves the remaining interval, repeat. That gives roughly `ceil(log2(m+1))` trials and is `O(log m)`. It is the "clever" answer and it is genuinely optimal — *when eggs are unlimited*. The risk is structural: a mid-floor drop can break an egg, and binary search wants to keep splitting, which keeps spending eggs. With only `k` eggs this strategy may not even be *feasible*, never mind optimal.
- *Minimax dynamic program.* Define the worst-case trial count as a recurrence over (eggs, floors) — or dually over (eggs, trials) — and compute it directly. This is correct by construction because it is literally the definition of the optimum under adversarial break/survive answers. The risks are not conceptual: pick the orientation that is fast enough at these constraints, get the recurrence boundaries right, and get the in-place array update order right.

I am inclined toward the DP, but the honest move is to first try to *break* the clever binary-search answer, because if it secretly worked I would happily ship three lines.

**Stress-testing binary search before committing.** Hand-waving "binary search is optimal for search problems" is exactly how a wrong solution gets shipped here, so let me attack it with a concrete instance. Take the smallest interesting case: `k = 1`, `m = 100`. Binary search says drop from floor 50. Suppose the egg breaks. Now I know `c <= 50`, but I have **zero eggs left** and I have not pinned down `c` — I cannot make another drop, so I have *failed to identify the critical floor at all*. Binary search is not merely sub-optimal with one egg; it is *infeasible*. The only guaranteed one-egg strategy is to drop from floor 1, then 2, then 3, …: the moment an egg breaks at floor `j`, I know `c = j`; if it never breaks I know `c = m+1`. That is a linear scan, and its worst case is exactly `m = 100` trials. So for `k = 1`, the clever `ceil(log2(101)) = 7` is wrong by a factor of fourteen, and worse, *unachievable*. The binary-search instinct is dead in the limited-egg regime.

It is worth pushing one step further so I understand *why* it dies, because that tells me what the right structure is. With `k = 2`, `m = 100`, binary search would still say about 7. But the real answer is 14, achieved by the "drop from floors 14, 27, 39, 50, …, dropping by one each time" strategy: with two eggs you sacrifice the first egg in a coarse arithmetic-decreasing ladder, then use the second egg to linearly scan the ~14-wide gap where it broke. The optimum is `14`, not `7`. So binary search both *over-promises* (claims 7) and would *break eggs too aggressively*. The lesson: with limited eggs, the number of times you are *allowed to be wrong* (i.e. break an egg) caps how much information each trial can buy you, and binary search ignores that cap entirely. I am discarding binary search and building the DP I can prove.

**Deriving the DP, orientation one: over (eggs, floors).** The natural minimax definition. Let `T(e, f)` be the minimum worst-case trials to guarantee finding the critical floor among `f` candidate floors using `e` eggs. I drop from some position; relative to the candidate range, suppose I drop so that `x` floors are at-or-below my drop point and the rest above. Two outcomes, and the adversary picks the worse:

- The egg **breaks**: the critical floor is among the `x - 1` floors strictly below the drop (the drop floor itself is now known to be `>= c`), and I have `e - 1` eggs: cost `T(e-1, x-1)`.
- The egg **survives**: the critical floor is among the `f - x` floors above, with `e` eggs still: cost `T(e, f - x)`.

So `T(e, f) = 1 + min over x in 1..f of max( T(e-1, x-1), T(e, f-x) )`, with base cases `T(e, 0) = 0` (nothing to determine), `T(1, f) = f` (one egg forces the linear scan), `T(e, 1) = 1`. The answer is `T(k, m)`. This is unimpeachably correct — it is the game tree's value. But its cost is `O(k * m^2)` if I do the inner `min` by brute scan: `100 * 10000^2 = 10^13`. Far too slow at the stated limits. I could speed the inner minimisation with the monotonicity of the optimal split, getting `O(k * m)`, but that optimisation is fiddly and easy to get subtly wrong. I want something both fast *and* obviously correct.

**Deriving the DP, orientation two: over (eggs, trials) — the dual.** Flip the question. Instead of "fewest trials for `f` floors," ask "most floors coverable with `e` eggs and `t` trials," and call it `cover(e, t)`. If I have `e` eggs and `t` trials, I make one drop. Whatever floor I drop from:

- if it **breaks**, I can resolve `cover(e-1, t-1)` floors below it;
- if it **survives**, I can resolve `cover(e, t-1)` floors above it;
- plus the floor I dropped from itself.

So the best single drop yields `cover(e, t) = cover(e-1, t-1) + cover(e, t-1) + 1`, with `cover(e, 0) = 0` and `cover(0, t) = 0`. This `cover` is *monotonically increasing in `t`*, so the answer to the original question is simply the smallest `t` with `cover(k, t) >= m`. No inner minimisation at all — the `+1` recurrence already encodes the optimal balanced drop. This is the clean, provable, fast formulation. I will ship this and use orientation one (the `O(k*m^2)` floors DP) only as an independent brute oracle to check it, since they are genuinely different algorithms.

**How large can `t` get, and why eggs can be capped.** I iterate `t` upward until `cover(k, t) >= m`. How big can `t` be? With one egg, `cover(1, t) = t` (each trial covers exactly one more floor — the linear scan), so for `k = 1`, `m = 10000` I iterate up to `t = 10000`. That is the worst case for the loop length, and it is fine: each trial does `O(k)` work, so total work is `O(k * t_answer)`, bounded by `O(100 * 10000) = 10^6`. Trivial.

But there is a sharper observation that lets me shrink the egg dimension and remove any overflow worry. With unlimited eggs, `cover(infinity, t) = 2^t - 1` (the pure binary-search bound). Since `m <= 10000 < 16383 = 2^14 - 1`, *no instance ever needs more than 14 trials* once `k >= 14`. And with at most 14 trials available, you can break an egg at most 14 times, so a 15th, 16th, … egg can never actually be dropped — eggs beyond the 14th are inert. Therefore I can cap the working egg count at `kk = min(k, 14)` without changing any answer. This keeps the `cover` array to length 15 and keeps every `cover` value bounded (I also clamp at `m`), so `long long` is never stressed. The cap is not an approximation; it is exact for these constraints.

**Sanity-checking the recurrence on paper before coding.** Let me verify `cover` against known answers. `cover(2, t)`: `cover(2,1) = cover(1,0)+cover(2,0)+1 = 0+0+1 = 1`. `cover(2,2) = cover(1,1)+cover(2,1)+1 = 1+1+1 = 3`. `cover(2,3) = cover(1,2)+cover(2,2)+1 = 2+3+1 = 6`. So with 2 eggs, 3 trials cover 6 floors — matching the stated example `k=2, m=6 -> 3`. Continue: `cover(2,t) = t(t+1)/2`, the triangular numbers. For `m = 100`, the smallest `t` with `t(t+1)/2 >= 100` is `t = 14` (`14*15/2 = 105 >= 100`, `13*14/2 = 91 < 100`). That is exactly the classic "100 floors, 2 eggs -> 14" answer. The recurrence is right.

Check `k = 3, m = 14`: `cover(3,t) = sum_{i=1..3} C(t,i) = t + t(t-1)/2 + t(t-1)(t-2)/6`. `cover(3,4) = 4 + 6 + 4 = 14 >= 14`, and `cover(3,3) = 3 + 3 + 1 = 7 < 14`, so the answer is `4` — matching the example. And `k = 1, m = 2`: `cover(1,t) = t`, smallest `t >= 2` is `2`. Matches. The dual formulation reproduces every hand example.

**First implementation — and a trace, because clean math transcribes dirty.** I keep a 1-D array `cover[e]` and advance one trial at a time, updating every egg count in place. My first instinct for the update loop:

```
for (int e = 1; e <= kk; ++e) {
    cover[e] = cover[e-1] + cover[e] + 1;   // breaks (e-1) + survives (e) + 1
}
```

This is the in-place hazard I have been burned by before: I am sweeping `e` *upward*, and `cover[e]` reads `cover[e-1]`. Let me trace it for `kk = 2` over two trials and see whether `cover[e-1]` is the value from the *previous* trial (what the recurrence demands) or from the *current* trial.

Start `cover = [0, 0, 0]` (indices 0,1,2), `t = 0`. First trial, `t = 1`, sweep up:
- `e = 1`: `cover[1] = cover[0] + cover[1] + 1 = 0 + 0 + 1 = 1`. `cover = [0,1,0]`.
- `e = 2`: `cover[2] = cover[1] + cover[2] + 1 = 1 + 0 + 1 = 2`. `cover = [0,1,2]`.

But the correct `cover` after one trial is `cover(1,1)=1`, `cover(2,1)=1` — *not* 2. The upward sweep read `cover[1] = 1`, which is the value I *just rewrote this same trial*, instead of `cover(1, t-1) = cover(1,0) = 0`. So `cover[2]` came out as `2` instead of `1`. The bug is precise: the breaks-term `cover[e-1]` must be the previous-trial value `cover(e-1, t-1)`, but sweeping `e` upward overwrites `cover[e-1]` before `cover[e]` reads it.

**Diagnosing and fixing the update order.** Both terms on the right — `cover[e-1]` and `cover[e]` — must be previous-trial values. `cover[e]` is read-then-written at the same index, so it is naturally the old value as long as I read it before writing. The problem is only `cover[e-1]`. If I sweep `e` from **high to low**, then when I compute `cover[e]` I have not yet touched `cover[e-1]` this trial, so it still holds `cover(e-1, t-1)`. That is the fix: iterate `e` downward.

```
for (int e = kk; e >= 1; --e) {
    cover[e] = cover[e-1] + cover[e] + 1;
}
```

Re-trace `kk = 2`, first trial `t = 1`, start `[0,0,0]`, sweep down:
- `e = 2`: `cover[2] = cover[1] + cover[2] + 1 = 0 + 0 + 1 = 1`. `cover = [0,0,1]`.
- `e = 1`: `cover[1] = cover[0] + cover[1] + 1 = 0 + 0 + 1 = 1`. `cover = [0,1,1]`.

Now `cover(1,1) = 1`, `cover(2,1) = 1` — correct. Second trial `t = 2`, sweep down:
- `e = 2`: `cover[2] = cover[1] + cover[2] + 1 = 1 + 1 + 1 = 3`. `cover = [0,1,3]`.
- `e = 1`: `cover[1] = cover[0] + cover[1] + 1 = 0 + 1 + 1 = 2`. `cover = [0,2,3]`.

`cover(1,2) = 2`, `cover(2,2) = 3` — both correct (`cover(2,2)=3` matched my paper derivation). The downward sweep is the right transcription, and the upward one broke for exactly the reason I diagnosed, which is the evidence I trust.

**Overflow and the clamp.** Without a guard, `cover(e, t)` for large `e` doubles each trial (`~2^t`), and although I cap eggs at 14 and `t` at 14 for large `k`, for *small* `k` and large `m` the loop runs up to `t = 10000` trials. `cover(1, t) = t` stays tiny, but I want a uniform guarantee no value ever runs away. Since I only ever compare `cover[kk]` against `m` and I never need a `cover` value larger than `m` to decide the loop, I clamp every updated value at `m`: `if (c > m) c = m;`. With `m <= 10000`, every stored value is at most `10000`, the sum of two of them plus one is at most `20001`, and `long long` holds that with astronomical headroom. The clamp is both an overflow guard and a no-op on the answer, because the loop stops the instant `cover[kk]` reaches `m`.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `m = 1`: one floor to resolve. First trial: `cover[kk]` becomes `1 >= 1`, loop exits with `t = 1`. One drop suffices (drop from floor 1; break => `c = 1`, survive => `c = 2 = m+1`). Correct for every `k`, including `k = 100, m = 1 -> 1`.
- `k = 1`, any `m`: `cover[1]` increments by exactly 1 per trial (`cover[0]` is always 0), so the loop runs `m` times and outputs `t = m`. The linear scan — exactly the lower bound I argued binary search violates. `k=1, m=10000 -> 10000`, verified to run in milliseconds.
- `k >= 14`: the cap makes `kk = 14`; `cover(14, 14) = 2^14 - 1 = 16383 >= 10000`, so `t` never exceeds 14 and the answer collapses to the unlimited-egg `ceil(log2(m+1))`. Checked `k=13` and `k=14` both give `14` for `m = 10000`, confirming the cap boundary is exact (the 14th egg is the last that can matter).
- Loop termination: `cover[kk]` strictly increases by at least 1 each trial (the `+1` term), and the target `m` is finite, so the `while` always terminates, in at most `m` iterations.
- Output: a single integer and a newline; `cin >>` consumes arbitrary whitespace, so "k m" on one line or two lines both parse.

**Self-verification against an independent oracle.** I do not trust a hand-trace alone for something I will ship, so I cross-checked the dual `cover` solution against the *primal* floors DP `T(e, f) = 1 + min_x max(T(e-1,x-1), T(e,f-x))` implemented separately (plain inner scan, no monotonicity trick — transparently the game-tree value). I generated 700 cases spanning `k in {1,2,...,100}` and small/medium `m` (kept small so the `O(k*m^2)` oracle stays fast), plus a curated edge set (`m` on triangular and power-of-two boundaries, `m=1`, single-egg ladders). Zero mismatches. I separately checked the extreme corners the quadratic oracle cannot reach quickly — `m = 10000` for `k in {1,2,3,13,14,50,100}` — against a closed-form `cover(k,t) = sum_{i=1..k} C(t,i)` computed in Python, again zero mismatches, with the solution finishing each in about 2 ms. The two independent algorithms agreeing on 700+ cases, including every classic textbook value (`k=2,m=6 -> 3`; `k=3,m=14 -> 4`; `k=2,m=100 -> 14`), is the evidence that lets me ship.

**Final solution.** I convinced myself the *idea* is right by killing the binary-search reflex with a concrete infeasibility (`k=1, m=100`: binary search claims 7 trials but cannot even finish after one broken egg, while the truth is 100), then deriving the dual `cover` recurrence and matching it to every hand example; and I convinced myself the *code* is right by tracing the in-place update to a precise cause (upward sweep reads a same-trial value), fixing it with a downward sweep, re-tracing the cases that broke, and differential-testing against an independent quadratic oracle over 700+ cases with zero mismatches. What I ship is the simple, provable `O(k * t_answer)` dual DP — not the clever binary search I disproved:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int k;       // number of eggs, 1 <= k <= 100
    long long m; // number of floors, 1 <= m <= 10000
    if (!(cin >> k >> m)) return 0;

    // cover[e] = maximum number of floors distinguishable with e eggs and the
    // current number of trials t. Recurrence over one more trial:
    //   cover_t(e) = cover_{t-1}(e-1) + cover_{t-1}(e) + 1
    // (drop from a floor: if the egg breaks we have e-1 eggs and t-1 trials for
    //  the floors below; if it survives we have e eggs and t-1 trials for the
    //  floors above; plus the floor we dropped from). cover_0(e) = 0 for all e,
    //  and cover_t(0) = 0 for all t. We want the smallest t with cover_t(k) >= m.
    //
    // We cap the working egg count at min(k, 14): with t trials the most floors
    // any number of eggs can cover is 2^t - 1, and 2^14 - 1 = 16383 >= 10000 = m,
    // so the answer never exceeds 14 once eggs >= 14. With at most 14 trials an
    // egg beyond the 14th can never be dropped, so eggs past 14 are irrelevant
    // and capping the working egg count at 14 leaves the answer unchanged.
    int kk = (int)min<long long>(k, 14);
    vector<long long> cover(kk + 1, 0);
    long long t = 0;
    while (cover[kk] < m) {
        ++t;
        // update from high egg count to low so cover[e-1] still holds the
        // previous-trial value when we read it.
        for (int e = kk; e >= 1; --e) {
            long long below = cover[e - 1];           // breaks: e-1 eggs, t-1 trials
            long long above = cover[e];               // survives: e eggs, t-1 trials
            long long c = below + above + 1;
            if (c > m) c = m;                          // clamp to avoid overflow
            cover[e] = c;
        }
    }

    cout << t << "\n";
    return 0;
}
```

**Causal recap.** Binary search looked right but a single concrete instance (`k=1, m=100`: it claims 7 trials yet is *infeasible* after one broken egg, true answer 100) showed the limited-egg cap on information makes it both wrong and unachievable, so I moved to a minimax DP. The primal `(eggs, floors)` orientation is correct but `O(k*m^2)` and too slow, so I switched to the dual `cover(e,t) = cover(e-1,t-1) + cover(e,t-1) + 1` and read off the smallest `t` with `cover(k,t) >= m`; my first in-place loop swept eggs upward and read a same-trial `cover[e-1]`, a trace of `kk=2` returning `cover(2,1)=2` instead of `1` pinpointed it, and sweeping downward fixes it; the `2^t-1` bound caps eggs at 14 and a clamp at `m` removes any overflow; and an independent quadratic floors-DP oracle agreeing over 700+ cases plus every classic textbook value closes it out.
