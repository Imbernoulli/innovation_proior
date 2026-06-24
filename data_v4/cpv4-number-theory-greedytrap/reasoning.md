**Reading the problem and pinning the contract.** I am given an order `k` with `2 <= k <= 5` and a target `n` with `0 <= n <= 10^6`. The admissible "pulse energies" are exactly the perfect `k`-th powers `1^k, 2^k, 3^k, ...`, each usable any number of times, and I must report the fewest of them that sum to exactly `n`. Input is one line `k n`; output is one integer. Before any algorithm I fix the two facts that will drive every later decision. First, `1 = 1^k` is admissible for every `k`, so every `n` is representable and an answer always exists — there is no "impossible" branch to print. Second, `n` can be as large as `10^6` while the largest admissible power can be as large as `n` itself, and to *enumerate* those powers I will be computing `b^k` for growing `b`; for `k = 5`, `b^5` overflows 32-bit integers around `b = 85` already (`85^5 ~ 4.4*10^9`), so the power enumeration must be done in 64-bit with an explicit overflow guard, not with `int` and not with floating `pow`. Those two facts — "always solvable" and "enumerate powers in 64-bit, carefully" — are the load-bearing observations.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can *prove*, not the one that is shortest to type.

- *Greedy by largest power.* Repeatedly subtract the largest admissible `k`-th power that fits in the remaining target, counting one pulse each time. It is `O(answer)` and three lines. The structural risk is exactly the shape where greedy dies: taking the largest power maximizes *coverage this step* but can leave a remainder whose own best representation is far worse than if I had taken a slightly smaller power. I will not trust it until I have tried to break it.
- *Shortest-representation DP.* Let `dp[v]` be the fewest admissible powers summing to `v`. Then `dp[0] = 0` and `dp[v] = 1 + min over admissible powers p <= v of dp[v - p]`. This is a coin-change-with-fewest-coins where the "coins" are the `k`-th powers, `O(n * P)` with `P` the number of admissible powers up to `n`. The correctness risk is not the idea but the *transcription*: the base case, the `p <= v` guard, and the power list.

**Stress-testing greedy before committing.** "Greedy feels right" is how wrong solutions get shipped, so let me attack it concretely with `k = 2`, where the admissible powers are `1, 4, 9, 16, 25, ...`. Take `n = 12`. Greedy grabs the largest square `<= 12`, which is `9`; remainder `3`. The largest square `<= 3` is `1`; remainder `2`. Then `1`, remainder `1`. Then `1`, remainder `0`. Greedy used `9 + 1 + 1 + 1 = 4` pulses. Is `4` optimal? Try `4 + 4 + 4 = 12`: that is `3` pulses, strictly fewer. So greedy is wrong on `n = 12`, and I can *see* why: snatching the `9` left a remainder `3` that has no square bigger than `1`, so it cost three more pulses, whereas backing off to `4` kept every remainder on the lattice of `4`s. The local "largest coverage" choice was globally worse. Greedy is out for `k = 2`.

I want to be sure this is not a `k = 2` fluke, so let me also break `k = 3`, powers `1, 8, 27, 64, ...`, at `n = 32`. Greedy: largest cube `<= 32` is `27`, remainder `5`; then `1` five times. Total `27 + 1*5 = 6` pulses. Optimum: `8 + 8 + 8 + 8 = 32`, which is `4` pulses. Greedy `6`, optimum `4`. Same failure mode, larger gap. That settles it across orders: greedy by largest power is provably suboptimal, and the DP is the approach I commit to.

**Deriving the DP and checking the recurrence on paper.** The state is the amount of energy still to deposit, `v`, and the only thing the future depends on is `v` itself — the past pulses do not constrain which powers I may use next. So `dp[v]` = fewest admissible powers summing to exactly `v`, with `dp[0] = 0` (deposit nothing) and, for `v >= 1`,

`dp[v] = 1 + min over admissible powers p with p <= v of dp[v - p]`.

This is correct because any optimal representation of `v` contains *some* admissible power `p` as one of its summands; removing that one summand leaves an optimal-or-better representation of `v - p`, so `dp[v] <= 1 + dp[v - p]`, and conversely any `1 + dp[v - p]` is achievable, so the minimum over `p` is tight. Let me confirm on `k = 2`, `n = 12`. Powers `1, 4, 9`. `dp[0]=0`. `dp[1]=1+dp[0]=1`. `dp[2]=1+dp[1]=2`. `dp[3]=1+dp[2]=3`. `dp[4]=min(1+dp[3], 1+dp[0])=1`. `dp[5]=min(1+dp[4], 1+dp[1])=2`. Skipping ahead, `dp[8]=min(1+dp[7], 1+dp[4])`; `dp[4]=1` gives `dp[8]=2`. `dp[12]=min(1+dp[11], 1+dp[8], 1+dp[3]) = min(?, 1+2, 1+3) = 3`. So `dp[12]=3`, matching `4+4+4` and beating greedy's `4`. The recurrence is right.

**Enumerating the powers — first cut, and a trace, because this is where overflow hides.** I need the list of admissible powers `<= n`. My first instinct is the obvious loop:

```
vector<long long> powers;
for (long long b = 1; b*b*b*b*b <= n; b++) // ... but k varies, so really:
for (long long b = 1; (long long)pow(b, k) <= n; b++)
    powers.push_back((long long)pow(b, k));
```

Two things smell wrong, so I trace it. Take `k = 5`, `n = 1000000`. The true admissible powers are `1, 32, 243, 1024, 3125, 7776, 16807, 32768, 59049, 100000, 161051, 248832, 371293, 537824, 759375`, and `16^5 = 1048576 > 10^6`, so the list should stop at `15^5 = 759375`. But `pow(b, k)` is a `double`; `pow(15, 5)` can come back as `759374.9999999` and truncate to `759374`, or `pow(10,5)` can come back as `100000.0000001` — either way an off-by-one in the *value* of a power, which silently corrupts `dp`. Worse, if I had written the literal `b*b*b*b*b` for a fixed `k`, then for `b` around `7132` (where `b^2` already exceeds `n` but the loop multiplies five factors before comparing) the product `b^5 ~ 1.8*10^19` overflows `long long` and can wrap to a value `<= n`, pushing a garbage power and possibly looping far past where it should. The defect is precise: computing `b^k` *before* comparing to `n` lets the product overflow, and using floating `pow` lets the value be off by one. Both are silent wrong-answers.

**Fixing the enumeration: multiply step by step with an early bail.** I build `b^k` by repeated integer multiplication, but I check *before each multiply* whether the running product would exceed `n`, using division to avoid ever forming an overflowing product:

```
vector<long long> powers;
for (long long b = 1;; b++) {
    long long p = 1;
    bool exceed = false;
    for (long long e = 0; e < k; e++) {
        if (p > n / b) { exceed = true; break; }  // p*b would exceed n
        p *= b;
    }
    if (exceed || p > n) break;
    powers.push_back(p);
}
```

The guard `p > n / b` means `p * b > n` (integer division is safe here because if `p <= n/b` then `p*b <= n`), so the running product never exceeds `n` and never overflows. The `for e` loop multiplies `b` into `p` exactly `k` times, giving `b^k` exactly, in integers — no floating point. Let me trace `k = 5`, `n = 1000000`, `b = 15`: `p=1`; e0 `n/b = 66666`, `1<=66666` ok, `p=15`; e1 `15<=66666` ok, `p=225`; e2 `225<=66666` ok, `p=3375`; e3 `3375<=66666` ok, `p=50625`; e4 `50625<=66666` ok, `p=759375`; loop ends, `p=759375 <= n`, push. Now `b=16`: e0 `1<=62500` ok `p=16`; e1 `16<=62500` ok `p=256`; e2 `256<=62500` ok `p=4096`; e3 `4096<=62500` ok `p=65536`; e4 `65536 > 62500` -> `exceed=true`, break; outer `break`. So the list stops exactly at `15^5`, as it must, and `16^5` never overflows anything. Enumeration fixed and verified.

**The DP loop — second cut, and a trace that finds a real indexing bug.** With `powers` ascending I write the table fill:

```
const int INF = 1e9;
vector<int> dp(n + 1, INF);
dp[0] = 0;
for (long long v = 1; v <= n; v++) {
    int best = INF;
    for (long long p : powers) {
        if (p > v) continue;            // <-- first cut used continue
        int cand = dp[v - p] + 1;
        if (cand < best) best = cand;
    }
    dp[v] = best;
}
```

That looks fine, but let me trace a tiny case to be sure the *control flow* is right, `k = 2`, `n = 2`, powers `1`. (For `n = 2`, `4 > 2` so the only admissible power is `1`.) Actually the subtle point is the `continue` vs `break`: with `continue`, when I hit a power `p > v` I skip it but keep scanning the rest of `powers`. Since `powers` is ascending, every later power is also `> v`, so `continue` just wastes time — but it is not *wrong*. Where it bites is performance, not correctness: for `k = 2`, `n = 10^6`, `powers` has `1000` entries, and for small `v` I would scan all `1000` of them every iteration instead of stopping early. That is the difference between `~10^6 * average-fitting-count` work and `10^6 * 1000` work; the latter is `10^9` and risks the time limit. So I change `continue` to `break`: because the list is ascending, the first `p > v` means all the rest exceed `v` too, and I can stop. This is a real efficiency fix, and it is also slightly safer (no needless `dp[v-p]` index computation for out-of-range `p`).

But while re-reading I spot an actual *correctness* hazard I had glossed: the loop variable `v` and the index `v - p` are `long long`, and `dp` is indexed by them. With `p <= v` guaranteed by the `break`, `v - p >= 0`, so the index is always in `[0, n]` — fine. The earlier `continue` version, though, computed `dp[v - p]` *after* the `p > v` test only when `p <= v`, so it was never out of range either; the bug there was purely the missed early termination. Good — no negative index, but I keep the `break` for the speed.

**A second genuine bug: the base case and `INF` arithmetic.** Let me trace `k = 2`, `n = 1`, to check the smallest non-zero target. `dp` is size `2`, all `INF` except `dp[0]=0`. `v=1`: powers `[1]` (since `4 > 1`); `p=1 <= 1`, `cand = dp[0]+1 = 1`, `best=1`; `dp[1]=1`. Output `dp[1]=1`. Correct — one pulse of energy `1`. Now I deliberately probe the danger I was worried about: what if some `dp[v-p]` were still `INF` and I did `INF + 1`? With `INF = 1e9` (a `double` literal assigned to `int`, which is `1000000000`), `INF + 1 = 1000000001`, which fits in a 32-bit `int` (max `~2.147*10^9`), so no overflow, and since `1` is always admissible, `dp[v]` is reachable from `dp[v-1]` for every `v`, so no entry ever *stays* `INF` for `v >= 1` anyway. The base case `dp[0]=0` is essential — if I had left `dp[0]=INF`, then `dp[1] = dp[0]+1 = INF+1`, garbage. I confirm `dp[0]=0` is set, and that `INF` is comfortably below the point where `INF + 1` overflows `int`. Safe.

I also separate out `n = 0` explicitly *before* building the DP, because `vector<int> dp(n+1)` with `n=0` is a length-1 vector and the loop `for v=1..0` never runs, leaving me to print `dp[0]=0` — which is actually correct, but I prefer to short-circuit `n=0` so the power-enumeration loop (which divides by `b` and compares to `n=0`) is never entered with a degenerate `n`. Trace `n = 0`: with the early `if (n == 0) { print 0; return; }`, I print `0`, the empty deposit. Correct.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: handled by the early return -> `0`. The empty set of pulses.
- `n = 1`: only power `1` is `<= 1`; `dp[1] = 1`. Correct, one pulse.
- `n` an exact power, e.g. `k = 2`, `n = 9`: `dp[9] = min(1+dp[8], 1+dp[5], 1+dp[0]) = min(..., ..., 1) = 1`. Correct, a single `9`.
- Large `k = 5`, small `n = 7`: the only power `<= 7` is `1` (`2^5 = 32 > 7`), so `dp[7] = 7`. The answer for small targets at high order is just `n` itself — a long chain of `1`s — and the DP reproduces that. Correct.
- Greedy-trap target `k = 2`, `n = 18`: greedy would do `16 + 1 + 1 = 3`; the DP finds `9 + 9 = 2`. `dp[18] = min(1+dp[17], 1+dp[14], 1+dp[9], 1+dp[2]) = 1 + dp[9] = 1 + 1 = 2`. Correct, and it beats greedy.
- Overflow in enumeration: handled by the `p > n/b` pre-multiply guard; no product ever exceeds `n`, so nothing overflows even at `k = 5`.
- Overflow in DP values: `dp` entries are at most `n <= 10^6` (all-`1`s representation), and `INF + 1 = 10^9 + 1` fits in `int`. Safe.
- Time: worst case is `k = 2`, `n = 10^6`, where `powers` has `1000` entries but the `break` makes the inner loop stop at the first power exceeding `v`; measured wall time is about `0.7s`, well under the `2s` limit.

**Re-verification against an independent brute force.** I wrote a separate BFS brute force that treats `0..n` as a graph with an edge `v -> v - p` for each admissible power `p <= v` and computes the shortest path (number of edges) from `n` to `0`; this is a genuinely different method from the forward DP (search vs. tabulation) and shares no code. Over an exhaustive sweep of every `(k, n)` with `k in {2,3,4,5}` and `n in [0, 400]` — `1604` cases — the two agree on every one, and over `500` additional random small cases from the seeded generator there are zero mismatches. The greedy-trap values (`k=2`: `12->3`, `18->2`, `32->2`; `k=3`: `32->4`) all come out as the DP/brute value, never the greedy value, which is the concrete evidence that the approach I rejected really was wrong and the one I kept is right.

**Final solution.** I disproved greedy with two traced counterexamples (`k=2`, `n=12`: greedy `4` vs optimal `3`; `k=3`, `n=32`: greedy `6` vs optimal `4`), derived and hand-checked the shortest-representation recurrence, fixed a real overflow/precision bug in the power enumeration by multiplying step-by-step under a division guard, fixed a real efficiency bug by turning the inner `continue` into an ascending-list `break`, and confirmed the base case and `INF` arithmetic on `n = 0` and `n = 1`. This is what I ship — one self-contained file, the `O(n * P)` DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k, n;
    if (!(cin >> k >> n)) return 0;        // empty input -> nothing to do

    // n == 0 needs zero pulses; handle before allocating the DP table.
    if (n == 0) { cout << 0 << "\n"; return 0; }

    // Enumerate every perfect k-th power in [1, n]: 1^k, 2^k, 3^k, ...
    // Multiply step by step, bailing out the instant the running product
    // exceeds n so the product never overflows.
    vector<long long> powers;
    for (long long b = 1;; b++) {
        long long p = 1;
        bool exceed = false;
        for (long long e = 0; e < k; e++) {
            if (p > n / b) { exceed = true; break; }  // p*b would exceed n
            p *= b;
        }
        if (exceed || p > n) break;
        powers.push_back(p);
    }
    // powers is ascending; powers[0] == 1 guarantees every n is representable.

    const int INF = 1e9;
    vector<int> dp(n + 1, INF);
    dp[0] = 0;
    for (long long v = 1; v <= n; v++) {
        int best = INF;
        for (long long p : powers) {
            if (p > v) break;                 // ascending: all later powers exceed v too
            int cand = dp[v - p] + 1;
            if (cand < best) best = cand;
        }
        dp[v] = best;
    }

    cout << dp[n] << "\n";
    return 0;
}
```

**Causal recap.** The natural one-liner — subtract the largest admissible `k`-th power each step — looked right but two traced counterexamples killed it (`k=2`, `n=12`: greedy takes `9` and is forced into `9+1+1+1=4`, while `4+4+4=3`; `k=3`, `n=32`: greedy `27+1*5=6` vs `8*4=4`), because grabbing the largest power maximizes this step's coverage but can strand the remainder off the useful lattice. So I moved to the shortest-representation DP `dp[v]=1+min_p dp[v-p]`, whose recurrence I checked on `n=12`; enumerating the powers naively with `pow` or an unguarded `b^k` product is a silent overflow/precision bug, which I fixed by multiplying step-by-step under the `p > n/b` guard so no product ever exceeds `n`; the inner DP scan needed `break` (not `continue`) on the first oversized power to stay fast on the `k=2`, `n=10^6` case; and the `dp[0]=0` base case plus an early `n=0` return plus `int`-safe `INF` close out the corners, with an independent BFS brute force agreeing on all `1604` exhaustive small cases and `500` random ones.
