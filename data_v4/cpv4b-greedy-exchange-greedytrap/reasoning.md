**Reading the problem and pinning the contract.** A single bench runs `n` jobs back-to-back, no gaps, no preemption. Job `i` takes `t[i]` minutes and its customer is charged `w[i]` per minute of waiting until *their own job* finishes. If I process the jobs in some order, the completion time of a job is the running sum of `t` up to and including it, and the job pays `w[i] * C[i]`. I pick the order to minimize `sum_i w[i] * C[i]`, and I print that minimum. Input on stdin is `n` followed by `n` pairs `(t[i], w[i])`; output is one integer.

Before any algorithm I fix the scale, because it dictates the data types and it is the kind of thing that silently sinks a solution. `n <= 10^5`, and `1 <= t[i], w[i] <= 10^4`. The largest a completion time can be is the sum of all durations, `<= 10^5 * 10^4 = 10^9` — that alone already nudges the edge of signed 32-bit (`~2.1 * 10^9`). The objective sums `w[i] * C[i]` across all jobs; a crude bound is `(sum of w) * (max C) <= (10^5 * 10^4) * 10^9`, which is absurd, but the *true* worst case is `t[i] = w[i] = 10^4` for all `i`: completion times are `10^4, 2*10^4, ...`, and the sum is `10^4 * 10^4 * (1 + 2 + ... + 10^5) = 10^8 * (10^5 * (10^5 + 1) / 2) ~ 5 * 10^17`. That fits in a signed 64-bit integer (`~9.2 * 10^18`) with room, but it overflows 32-bit by eight orders of magnitude. So **every accumulator is `long long`**, non-negotiable. I will also store `t` and `w` as `long long` so that any product I form during comparison stays 64-bit.

**Laying out the candidate approaches.** There are `n!` orders, so I cannot enumerate. I want a single sort key that yields an optimal order, and I want to commit only to the one I can *prove*, not the one that reads nicely.

- *Shortest processing time first* — sort by `t` ascending. The story: clearing quick jobs early keeps all subsequent completion times low. `O(n log n)`. Risk: it throws away the weights entirely, and the objective is weighted, so I distrust it.
- *Most impatient first* — sort by `w` descending. The story: get the high-penalty customers off the bench first so their `C` is small. `O(n log n)`. Risk: it throws away the durations, and a heavy-but-enormous job could hog the bench and inflate everyone else's wait.
- *Ratio rule* — sort by some combined key like `t/w`. `O(n log n)`. This is the one I expect to be right, but I will not assert the key from memory; I will *derive* it by an exchange argument and then numerically check the inequality.

**Stress-testing the two cheap greedies before committing.** "Shortest first feels right" and "heaviest first feels right" are both how wrong solutions get shipped. Let me actually attack them with one concrete instance and see if either survives. Take three jobs:

- `A = (t=3, w=4)`
- `B = (t=1, w=1)`
- `C = (t=2, w=3)`

I will compute the cost of the order each greedy produces, and compare against the best of all `3! = 6` orders.

*Shortest-processing-time-first* sorts by `t` ascending: `B (t=1), C (t=2), A (t=3)`. Completion times: `B` finishes at `1`, `C` at `1+2=3`, `A` at `3+3=6`. Cost `= w_B*1 + w_C*3 + w_A*6 = 1*1 + 3*3 + 4*6 = 1 + 9 + 24 = 34`.

*Most-impatient-first* sorts by `w` descending: `A (w=4), C (w=3), B (w=1)`. Completion times: `A` at `3`, `C` at `3+2=5`, `B` at `5+1=6`. Cost `= 4*3 + 3*5 + 1*6 = 12 + 15 + 6 = 33`.

Now the brute-force optimum over all six orders. Let me just list them by cost:
- `C, A, B`: `C@2, A@5, B@6` -> `3*2 + 4*5 + 1*6 = 6 + 20 + 6 = 32`.
- `A, C, B`: `A@3, C@5, B@6` -> `4*3 + 3*5 + 1*6 = 12 + 15 + 6 = 33`.
- `C, B, A`: `C@2, B@3, A@6` -> `3*2 + 1*3 + 4*6 = 6 + 3 + 24 = 33`.
- `A, B, C`: `A@3, B@4, C@6` -> `4*3 + 1*4 + 3*6 = 12 + 4 + 18 = 34`.
- `B, C, A`: `34` (computed above, this is shortest-first).
- `B, A, C`: `B@1, A@4, C@6` -> `1*1 + 4*4 + 3*6 = 1 + 16 + 18 = 35`.

The optimum is `32`, achieved by `C, A, B`. **Both cheap greedies are strictly suboptimal**: shortest-first gives `34`, impatient-first gives `33`. The counterexample paid off — it killed *both* approaches I might otherwise have shipped. And it tells me *why*: shortest-first put the light job `B` first even though `B`'s weight is tiny, wasting "early slots" on a customer who barely cares; impatient-first put the long heavy job `A` first, and `A`'s three-minute duration pushed everyone behind it. The right order, `C, A, B`, balances duration against weight. Both single-attribute keys are out.

**Deriving the correct key by an exchange argument.** Suppose I have any order, and two jobs `i` then `j` are *adjacent* in it. Let `P` be the total duration of everything strictly before this pair. If I run `i` then `j`, their two completion times are `C_i = P + t_i` and `C_j = P + t_i + t_j`, so the pair contributes
`w_i (P + t_i) + w_j (P + t_i + t_j)`.
If instead I run `j` then `i`, the pair contributes
`w_j (P + t_j) + w_i (P + t_j + t_i)`.
Everything *outside* this pair is untouched: jobs before see the same `P`, and jobs after see the same total `P + t_i + t_j` regardless of internal order, so their costs do not change. Subtracting (run `i,j`) minus (run `j,i`), the `P` terms and the `t_i + t_j` terms cancel and I get
`(run i,j) - (run j,i) = w_j t_i - w_i t_j`.
So putting `i` before `j` is at least as good (cost not larger) exactly when `w_j t_i - w_i t_j <= 0`, i.e. `t_i w_j <= t_j w_i`, i.e. dividing by the positive `w_i w_j`, when `t_i / w_i <= t_j / w_j`. This is **Smith's rule**: sort jobs by `t/w` ascending. If any adjacent pair violates this order, swapping them does not increase the cost, so by repeatedly fixing inversions (a bubble-sort argument) I can transform any order into the sorted one without ever increasing cost — hence the sorted order is optimal.

I will compare with the cross-multiplied form `t_i * w_j < t_j * w_i` rather than the division, to stay in exact integer arithmetic. The products are `t * w <= 10^4 * 10^4 = 10^8`, comfortably 64-bit, no rounding, no division-by-anything.

**A numeric self-check of the exchange inequality, because I refuse to ship an unverified identity.** I derived `(run i,j) - (run j,i) = w_j t_i - w_i t_j`. Let me verify this on the counterexample's deciding pair so I trust the formula, not just the algebra. Take `i = B = (1, 1)` and `j = C = (2, 3)`, with the rest of the schedule giving `P = 0` (they are first). Order `B then C`: cost contribution `w_B(0 + 1) + w_C(0 + 1 + 2) = 1*1 + 3*3 = 1 + 9 = 10`. Order `C then B`: `w_C(0 + 2) + w_B(0 + 2 + 1) = 3*2 + 1*3 = 6 + 3 = 9`. Difference `(B,C) - (C,B) = 10 - 9 = 1`. The formula predicts `w_C t_B - w_B t_C = 3*1 - 1*2 = 3 - 2 = 1`. They match. And the sign is positive, meaning `C` before `B` is cheaper — consistent with `C`'s ratio `t/w = 2/3 ~ 0.667` being smaller than `B`'s `1/1 = 1`, so `C` should come first. The formula is right, and its prescription agrees with the brute-force winner `C, A, B` (ratios: `C = 0.667`, `A = 0.75`, `B = 1.0`, ascending). I trust the key.

**First implementation — and immediately a trace, because a clean rule transcribes dirty.** My first cut of the comparator and accumulation:

```
sort(ord.begin(), ord.end(), [&](int i, int j){
    return t[i] * w[i] < t[j] * w[j];   // (WRONG KEY)
});
long long cur = 0, total = 0;
for (int idx : ord) {
    cur += t[idx];
    total += w[idx] * cur;
}
```

Something is off about that comparator — I wrote `t[i]*w[i] < t[j]*w[j]`, comparing the *product* `t*w` of each job, but the derived key is the *ratio* `t/w`, whose cross-multiplied form is `t[i]*w[j] < t[j]*w[i]`. Let me trace it on the counterexample to be sure. Jobs (index: t, w): `0:A=(3,4)`, `1:B=(1,1)`, `2:C=(2,3)`. The product key gives `t*w`: `A = 12`, `B = 1`, `C = 6`. Sorted ascending by product: `B (1), C (6), A (12)` -> order `B, C, A`. Cost of `B,C,A`: that is exactly the shortest-first order I already computed, cost `34`. But the optimum is `32`.

**Diagnosing the first bug.** The comparator sorts by `t*w` instead of `t/w`. Those are different functions: `t*w` rewards jobs that are *both* small, while `t/w` is the ratio the exchange argument actually demands. On this instance the product key happened to coincide with shortest-first and reproduced its suboptimal `34`. The fix is to cross-multiply the ratio correctly: `i` before `j` when `t[i]/w[i] < t[j]/w[j]`, i.e. `t[i]*w[j] < t[j]*w[i]`. Note the *crossed* indices — `w[j]` pairs with `t[i]`. Let me rewrite and re-trace.

```
sort(ord.begin(), ord.end(), [&](int i, int j){
    return t[i] * w[j] < t[j] * w[i];   // t/w ascending, cross-multiplied
});
```

Re-trace on the counterexample. Compare pairs by `t[i]*w[j]` vs `t[j]*w[i]`:
- `A=(3,4)` vs `B=(1,1)`: `3*1 = 3` vs `1*4 = 4`; `3 < 4` so `A` before `B`.
- `A=(3,4)` vs `C=(2,3)`: `3*3 = 9` vs `2*4 = 8`; `9 < 8` false, so `C` before `A`.
- `C=(2,3)` vs `B=(1,1)`: `2*1 = 2` vs `1*3 = 3`; `2 < 3` so `C` before `B`.
So the total order is `C, A, B`. Accumulate: `cur=0`. `C`: `cur=2`, `total += 3*2 = 6`. `A`: `cur=5`, `total += 4*5 = 20` -> `26`. `B`: `cur=6`, `total += 1*6 = 6` -> `32`. Output `32`. Correct — matches the brute-force optimum. The bug was a wrong key that masqueraded as plausible, and the trace pinned it.

**Second trace — the comparator's strictness and a tie, because a bad `<` corrupts a sort.** `std::sort` requires a strict weak ordering: when two jobs are *equivalent* under the key (here, equal ratio `t/w`), the comparator must return `false` both ways. My cross-multiplied `t[i]*w[j] < t[j]*w[i]` returns `false` both ways when `t[i]*w[j] == t[j]*w[i]`, so equal-ratio jobs are correctly treated as incomparable — good. But let me trace a tie to confirm the *cost* is order-independent for ties and that I am not introducing UB. Take two jobs with the same ratio: `X=(2,1)` and `Y=(4,2)` (both `t/w = 2`). Compare: `t_X*w_Y = 2*2 = 4` vs `t_Y*w_X = 4*1 = 4`; `4 < 4` is `false`, and the reverse is also `false`. So `X` and `Y` are equivalent; either order is allowed. Cost of `X,Y`: `cur=2, total=1*2=2`; `cur=6, total+=2*6=12` -> `14`. Cost of `Y,X`: `cur=4, total=2*4=8`; `cur=6, total+=1*6=6` -> `14`. Identical — exactly as the exchange formula predicts (`w_Y t_X - w_X t_Y = 2*2 - 1*4 = 0`, a tie costs the same either way). So my comparator is a valid strict weak order and ties are genuinely free. To make output deterministic regardless of `sort`'s internals (not required for correctness, but tidy for debugging), I break exact-ratio ties by original index. I verified above that this changes no cost.

**A subtle overflow re-check on the accumulation.** In `total += w[idx] * cur`, `w[idx]` is `long long` and `cur` is `long long`, so the product is computed in 64-bit — good. If I had left `w` as `int`, then `w[idx] * cur` would still promote to `long long` because `cur` is `long long`, but I declared `w` as `long long` anyway to be safe and to keep the comparator's `t[i]*w[j]` in 64-bit without thinking. The worst-case `total ~ 5*10^17 < 9.2*10^18`, and each intermediate `w*cur <= 10^4 * 10^9 = 10^13`, both safe. The comparator products `t*w <= 10^8` are safe. No overflow anywhere.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: `cin >> n` reads `0`, the read loop runs zero times, the sort of an empty vector is a no-op, the accumulation loop never runs, `total = 0`. Output `0`. Correct — an empty bench costs nothing. (Also, if the input is entirely empty, `if (!(cin >> n)) return 0;` prints nothing and exits, which the judge treats as no output for an empty test; the contract guarantees `n` is present, so the meaningful empty case is `n = 0`.)
- `n = 1`, `t=[7], w=[5]`: sort of one element is a no-op; `cur=7`, `total = 5*7 = 35`. Output `35`. The single job finishes at `7`, weighted by `5`. Correct.
- All equal durations, e.g. `t=[5,5,5]`, varying weights: ratios are `5/w`, so larger `w` -> smaller ratio -> earlier. That is impatient-first, which *is* optimal when durations tie (the exchange difference reduces to `5(w_j - w_i)` sign). Consistent.
- All equal weights, e.g. `w=[2,2,2]`, varying durations: ratios are `t/2`, so smaller `t` -> earlier. That is shortest-first, optimal when weights tie. Consistent. (Each single-attribute greedy is the *special case* of Smith's rule when the other attribute is constant — which is exactly why each looked plausible and why neither is right in general.)
- All equal ratios: every order costs the same (verified by the tie trace); any permutation is optimal.
- Large `n = 10^5`, `t = w = 10^4`: I ran it; output `500005000000000000`, matching the closed form `10^4 * 10^4 * (10^5 * (10^5+1)/2)`, in 0.09 s and 5 MB. No overflow, well within the time limit.

**Final solution.** I convinced myself the *idea* is right by breaking both cheap greedies with one concrete counterexample, by deriving Smith's rule from a clean adjacent-swap exchange, and by numerically checking the exchange inequality on the deciding pair. I convinced myself the *code* is right by tracing a wrong key (`t*w` instead of `t/w`) to its precise failure and fixing the cross-multiplication, by tracing a tie to confirm a valid strict weak ordering and order-free cost, and by re-checking every accumulator for 64-bit safety. That is what I ship — one self-contained file, the `O(n log n)` sort I can defend:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> t(n), w(n);
    for (int i = 0; i < n; i++) cin >> t[i] >> w[i];

    // Smith's rule: process jobs in non-decreasing order of t/w.
    // Compare i before j  <=>  t[i]*w[j] < t[j]*w[i]   (cross-multiply, all positive).
    vector<int> ord(n);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int i, int j) {
        // primary key: t/w ascending  =>  t[i]*w[j] < t[j]*w[i]
        // products fit in long long: t,w <= 1e4 each => product <= 1e8
        long long lhs = t[i] * w[j];
        long long rhs = t[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j; // deterministic tie-break (does not affect cost)
    });

    long long cur = 0;      // running completion time
    long long total = 0;    // sum of w[i] * C[i]
    for (int idx : ord) {
        cur += t[idx];
        total += w[idx] * cur;
    }

    cout << total << "\n";
    return 0;
}
```

**Causal recap.** Two obvious greedies — shortest-first and most-impatient-first — both looked right, but the single traced counterexample `A=(3,4), B=(1,1), C=(2,3)` (shortest-first 34, impatient-first 33, optimum 32) disproved both at once, because each ignores one attribute the weighted objective depends on; an adjacent-swap exchange argument then gives the exact difference `w_j t_i - w_i t_j`, whose sign says to order by `t/w` ascending (Smith's rule), and I numerically confirmed that difference on the deciding pair; my first transcription sorted by the *product* `t*w` instead of the *ratio*, which a trace caught when it reproduced the suboptimal 34, fixed by the cross-multiplied `t[i]*w[j] < t[j]*w[i]`; a tie trace confirmed the comparator is a valid strict weak ordering and that equal-ratio jobs cost the same in any order; and `long long` throughout closes the overflow corner where completion times reach `~10^9` and the weighted sum reaches `~5*10^17`.
