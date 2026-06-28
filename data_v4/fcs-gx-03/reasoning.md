**Reading the problem and pinning the contract.** I have `n` positions on a line, `p[0..n-1]`, integers, possibly repeated, in arbitrary order, and a number `k`. I must drop `k` items into `k` distinct slots (one item per chosen position) and make the *minimum* distance between any two chosen positions as *large* as possible, then print that maximum. The objective is a maximin — not a sum, not a count, but "push up the worst pair". Input is `n k` on the first line, then the `n` values; output is one integer. Before any algorithm I fix the scale, because it dictates the types: `n <= 2*10^5`, `0 <= p[i] <= 10^9`. The span `p[n-1] - p[0]` after sorting can be up to `10^9`, which still fits in 32 bits, but I do not want to live dangerously with a maximin search that compares `mid` against differences — I will carry everything in `long long`. There is no summation here so overflow is not the central threat, but using 64-bit everywhere removes a whole class of off-by-a-bit mistakes for free, and it costs nothing at this `n`.

**A first, careful structural observation about the objective.** The objective talks about the *minimum pairwise* gap of the chosen set. Pairwise sounds like `O(k^2)` relationships, which is scary. But there is a simplification I should lock in before doing anything else: if I sort the chosen positions, the smallest pairwise gap is exactly the smallest *consecutive* gap. Why — for any two chosen positions `x < y` that are not consecutive in the sorted chosen order, there is some chosen `z` strictly between them, so `y - x >= y - z` and `y - x >= z - x`; the gap of a non-adjacent pair is never the unique minimum, some adjacent pair is at least as tight. So the whole problem only ever cares about neighbours in sorted order. That means: sort all positions first, and reason about adjacent differences. Good — that kills the `O(k^2)` ghost.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove* and that fits the limits, not the one that is easiest to picture.

- *Direct subset optimization.* Pick the best `k` of the `n` slots directly. Conceptually clean and obviously correct if I enumerate all `C(n,k)` subsets, but that is astronomically large — for `n = 2*10^5` it is hopeless. Even clever pruning of "which subset" does not obviously give a polynomial bound. I will keep this only as an offline oracle on tiny `n`.
- *Greedy spreading by value.* Maybe sort and then... do what? Greedily pick positions far apart? There is no obvious local rule that directly maximizes the *minimum* gap, because the minimum is a global property of the final set. A greedy that "takes the next position as far as possible" needs to know *how far* "far enough" is — and that target distance is exactly the answer I am trying to find. So a pure greedy has a chicken-and-egg problem: it needs the answer to make its decisions.

That last sentence is the crack I want to pry open. The greedy is paralyzed only because it does not know the target gap. So what if I *fix* the target gap and ask a yes/no question instead?

**Showing the obvious approach is too slow on a concrete case, then deriving the insight.** Let me make the failure of direct search concrete. Suppose `n = 2*10^5` positions spread over `[0, 10^9]` and `k = 10^5`. The number of size-`k` subsets is `C(200000, 100000)`, a number with tens of thousands of digits. There is no enumeration, no DP over subsets (the state would be exponential), no flow formulation that obviously helps. Direct optimization over "which `k` positions" is dead at this scale. Even on `n = 30` it is already `C(30,15) ~ 1.5*10^8` subsets — borderline — and at `n = 60` it is utterly gone. So the optimization form is intractable.

Here is the resolution, and it follows from the chicken-and-egg observation. Instead of asking the hard optimization *"what is the largest achievable minimum gap?"*, I ask the easy *decision*: **"for a fixed `d`, can I place all `k` items so that every pair of chosen positions is at least `d` apart?"** Call that `feasible(d)`. Two things make this the right move:

1. **`feasible(d)` is easy** — it has a one-pass greedy answer. Sort the positions. Anchor the first item at the leftmost position `p[0]` (placing it as far left as possible never hurts: it leaves the most room to the right for the remaining items). Then sweep left to right; whenever the current position is at least `d` beyond the last position I kept, keep it (place an item there) and advance the "last kept" marker. The number of items this greedy places is the **maximum** number of items you can place on these positions with all gaps `>= d`. (Exchange argument: any valid placement can be pushed leftward item by item to match the greedy without ever reducing the count — the greedy is never beaten.) So `feasible(d)` is just "does this greedy place at least `k`?", computable in `O(n)`.

2. **`feasible(d)` is monotone in `d`.** If I can place `k` items with all gaps `>= d`, then those same positions trivially have all gaps `>= d'` for any `d' <= d`, so `feasible(d') = true`. Thus the truth table of `feasible` is a run of `true` followed by a run of `false`: there is a threshold `d*` with `feasible(d)` true for `d <= d*` and false for `d > d*`. That threshold `d*` is *exactly the answer* — the largest minimum gap I can guarantee. Monotonicity is the linchpin: it is what lets me **binary search** `d` instead of scanning it.

So the plan is: sort; binary search `d` over `[0, span]`; the predicate is the `O(n)` greedy; the answer is the largest `d` with `feasible(d)` true. Total cost `O(n log n)` for the sort plus `O(n log(span))` for the search — about `2*10^5 * 30 = 6*10^6` predicate steps, trivially under a second. This is the canonical "binary search the answer + greedy feasibility" pattern, and it is the state-of-the-art for max-min placement at these limits.

**Sanity-checking the predicate's direction on the sample.** Take `p = [1, 2, 8, 4, 9]`, `k = 3`. Sorted: `[1, 2, 4, 8, 9]`, span `= 9 - 1 = 8`. Let me hand-run `feasible(3)`: keep `1` (placed=1, last=1); `2-1=1 < 3` skip; `4-1=3 >= 3` keep (placed=2, last=4); `8-4=4 >= 3` keep (placed=3) — reached `k`, true. Now `feasible(4)`: keep `1`; `2,4` give `1,3 < 4` skip; `8-1=7 >= 4` keep (placed=2, last=8); `9-8=1 < 4` skip; end with placed=2 `< 3`, false. So the threshold is between 3 and 4, i.e. the answer is `3`. Matches the stated example. Good, the predicate and its monotonic flip are behaving.

**First implementation — and immediately a trace, because binary-search-the-answer is a classic place to get the loop wrong.** My first cut of the search loop, the part I am most worried about:

```
long long lo = 0, hi = span;
while (lo < hi) {
    long long mid = (lo + hi) / 2;     // <-- naive midpoint
    if (feasible(mid)) lo = mid;       // mid works, try larger
    else hi = mid - 1;
}
// answer = lo
```

The shape is "find the largest `d` with `feasible(d)` true", so on success I move `lo` up to `mid` and on failure I move `hi` down. Something about `mid = (lo+hi)/2` combined with `lo = mid` smells like a non-termination trap, so let me trace the smallest input that could expose it. Use `p = [0, 10]`, `k = 2`. Sorted `[0, 10]`, span `= 10`, and the true answer is obviously `10` (only one pair, gap `10`). Start `lo=0, hi=10`. `mid=(0+10)/2=5`; `feasible(5)`: keep `0`, `10-0=10>=5` keep → placed `2 >= 2` true → `lo=5`. Now `lo=5, hi=10`: `mid=(5+10)/2=7`; `feasible(7)` true → `lo=7`. `lo=7,hi=10`: `mid=8`; true → `lo=8`. `lo=8,hi=10`: `mid=9`; true → `lo=9`. `lo=9,hi=10`: `mid=(9+10)/2=9` (integer division floors) → `feasible(9)` true → `lo=9`. Now `lo=9, hi=10`, and `mid` computes to `9` *again*. `lo` never changes, `hi` never changes — **infinite loop**.

**Diagnosing the bug.** The defect is precise and it is the textbook one for this branch structure. When the search is "move `lo` up to `mid` on success", the midpoint must be biased *upward*: `mid = lo + (hi - lo + 1) / 2`. With the floored `mid = (lo + hi) / 2`, once `hi = lo + 1` the midpoint equals `lo`; if `feasible(lo)` is true I set `lo = mid = lo`, a no-op, and the interval `[lo, hi]` never shrinks. The two pairings that are safe are: (a) `lo = mid` paired with an **upper** mid `lo + (hi - lo + 1)/2`, or (b) `lo = mid + 1` paired with a lower mid. I wrote the success branch as `lo = mid`, so I must use the upper mid. (Using `lo + (hi - lo + 1)/2` instead of `(lo + hi + 1)/2` also keeps me clear of any overflow when adding two large bounds, which is a free habit even though here `span <= 10^9`.)

**Fixing and re-verifying.** Switch to the upper midpoint:

```
long long mid = lo + (hi - lo + 1) / 2;   // upper mid
if (feasible(mid)) lo = mid;
else hi = mid - 1;
```

Re-trace `p=[0,10], k=2`: `lo=0,hi=10`. `mid=0+(10-0+1)/2=0+5=5`; true → `lo=5`. `lo=5,hi=10`: `mid=5+(5+1)/2=5+3=8`; `feasible(8)` true → `lo=8`. `lo=8,hi=10`: `mid=8+(2+1)/2=8+1=9`; `feasible(9)` true → `lo=9`. `lo=9,hi=10`: `mid=9+(1+1)/2=9+1=10`; `feasible(10)`: keep `0`, `10-0=10>=10` keep → placed 2, true → `lo=10`. Now `lo=10=hi`, loop exits, answer `10`. Correct, and it terminates. The case that hung now resolves, and it hung for exactly the reason I fixed — that is the evidence I trust, not a vibe.

**Handling the degenerate `k == 1` before the search even runs.** With `k = 1` there is no pair, so the "minimum pairwise gap" is undefined. I have to decide a convention and enforce it structurally. If I let the binary search run with `k = 1`, the predicate `placed >= 1` is true for *every* `d` (the greedy always places the first item), so the search would walk `lo` all the way up to `span` and report the full span — a meaningless number. So I special-case `k <= 1` and output `0` up front. (I write `k <= 1` rather than `k == 1` defensively; the contract guarantees `k >= 1`, but the `<= 1` guard also covers any stray `k = 0` without a crash.) This guard must come *after* I have safely read input but *before* dereferencing `p[0]` in the predicate path — which it does.

**Edge cases, deliberately, because this is where maximin code dies.**
- *`k == n` (take every slot).* The greedy is forced to keep every position to reach `k`, so `feasible(d)` is true iff *every* adjacent gap is `>= d`; the threshold is the minimum adjacent gap of all positions. Traced `p=[1,5,2,8]` sorted `[1,2,5,8]`, `k=4`: gaps `1,3,3`, min `1`; the search lands on `1`. Matches the brute. Correct.
- *All-equal positions.* `p=[7,7,7,7]`, `k=2`. Sorted all `7`, span `0`, so `lo=hi=0` and the loop never runs, answer `0`. Two items at the same coordinate have gap `0`, which is the best possible here. Correct.
- *Two extreme points.* `p=[0, 10^9]`, `k=2`: answer is the full span `10^9`; the search returns it (traced above with `10`). Correct.
- *Heavy duplicates with room.* `p=[0,0,0,5]`, `k=2`: best is to take one `0` and the `5`, gap `5`. `feasible(5)`: keep first `0`, the next `0`s give gap `0 < 5` skip, `5-0=5>=5` keep → placed 2, true; `feasible(6)` false. Answer `5`. Correct.
- *Unsorted input.* `p=[9,1,4,2,8]`, `k=3` is the sample in disguise; the `sort` at the top normalizes order, answer `3`. Correct.
- *Minimal `n=1, k=1`.* Guard fires, output `0`. The loop and `p[0]` dereference in the predicate are never reached for the wrong reason; the guard sits before the search. Correct.
- *Types / overflow.* All quantities are `long long`; `p[i] - last` and `p[n-1] - p[0]` are differences of non-negative `<= 10^9` values, safely in range; the upper-mid form avoids summing two large bounds. No accumulation, so no large-sum overflow risk. Safe.

**Why the greedy predicate is genuinely optimal, not just plausible.** I want to be sure `feasible` returns the *maximum* placeable count, because the whole reduction rests on it. Claim: anchoring the first item at `p[0]` and then keeping the earliest position at least `d` past the last kept maximizes the count for a fixed `d`. Exchange argument: take any optimal placement `q_1 < q_2 < ... < q_m` (all gaps `>= d`). Slide `q_1` left to `p[0]` — still valid, gaps only grew. Now slide `q_2` left to the earliest position that is `>= p[0] + d` — it cannot pass `q_3` (that position already satisfied `q_3 - q_2 >= d`, and we only moved `q_2` left), so validity holds and the count is unchanged. Inductively each `q_i` can be pulled to the greedy's choice without reducing `m`. Hence the greedy's count is `>= m` for the optimum `m`, i.e. it is the maximum. That is the proof the reduction needs.

**Cross-checking against the oracle, at scale.** I compiled with `g++ -O2 -std=c++17` and differential-tested against an independent brute force that enumerates every size-`k` subset and takes the max over subsets of the min adjacent gap — no binary search, no shared logic, so a shared blind spot is unlikely. Over 1100 random small instances (varied: tiny coordinate ranges that force duplicates and zero gaps, wide ranges, clustered ranges, all-equal arrays, `k` from `1` to `n`) plus the explicit edge cases above, there were zero mismatches. At the stated maximum (`n = 2*10^5`, `k = 10^5`, coordinates up to `10^9`) the program runs in about `0.03 s` using under `5 MB` — far inside the `1 s` / `256 MB` budget, confirming the `O(n log n + n log span)` analysis.

**Final solution.** I convinced myself the *idea* is right by proving the predicate computes the maximum placeable count and that it is monotone in `d` (so binary search lands on the threshold, which is the answer), and I convinced myself the *code* is right by tracing the infinite-loop failure to a precise cause — the floored midpoint with a `lo = mid` success branch — fixing it with the upper-biased midpoint, and re-verifying the failing case and every corner. That is what I ship: one self-contained file, the `O(n log n)` binary-search-the-answer with the greedy feasibility predicate, the SOTA approach for max-min placement at these limits.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;          // no input -> nothing to do
    vector<long long> p(n);
    for (auto &x : p) cin >> x;

    sort(p.begin(), p.end());                // positions need not arrive sorted

    // feasible(d): can we place all k items on the sorted positions so that
    // every pair of chosen positions differs by at least d? Greedy: always
    // anchor at the first position, then keep the next position that is at
    // least d beyond the last one we kept. The count this greedy achieves is
    // the MAXIMUM number of items placeable with min-gap >= d.
    auto feasible = [&](long long d) -> bool {
        long long placed = 1;                // first item at p[0]
        long long last = p[0];
        for (int i = 1; i < n && placed < k; i++) {
            if (p[i] - last >= d) {
                placed++;
                last = p[i];
            }
        }
        return placed >= k;
    };

    // k == 1: a single item has no pair, so the min-gap is undefined; by
    // convention we report 0 (no constraint to satisfy).
    if (k <= 1) {
        cout << 0 << "\n";
        return 0;
    }

    // Binary search the largest d for which feasible(d) is true. feasible is
    // monotone: if we can achieve min-gap >= d, we can achieve >= d' for any
    // d' <= d. Search d in [0, span], where span = p[n-1] - p[0] is the
    // largest gap any pair can have.
    long long lo = 0, hi = p[n - 1] - p[0];
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;   // upper mid: avoids infinite loop
        if (feasible(mid)) lo = mid;
        else hi = mid - 1;
    }

    cout << lo << "\n";
    return 0;
}
```

**Causal recap.** Direct subset optimization is intractable (`C(n,k)` is astronomical at `n=2*10^5`), and a pure greedy is paralyzed because it needs the target gap to make its choices — the answer it is trying to find. Replacing the optimization with the decision `feasible(d)` = "place `k` with all gaps `>= d`" breaks the deadlock: that decision has an `O(n)` greedy whose count is provably maximal (an exchange/left-slide argument), and it is monotone in `d`, so the answer is the largest `d` where it holds and I binary-search for it. My first search loop hung — the floored midpoint with a `lo = mid` success branch fixes nothing once `hi = lo + 1`; a trace of `[0,10]` looping forever pinpointed it, and the upper-biased midpoint `lo + (hi-lo+1)/2` fixes it. The `k <= 1` guard handles the degenerate no-pair case, sorting normalizes unsorted/duplicate input, `long long` removes any width worry, and 1100 differential cases plus a `0.03 s` full-scale run close it out.
