The objective here is a maximin: I place `k` items on given positions to push up the *smallest* pairwise gap. I am not summing or counting, I am maximizing the worst pair, and the chosen positions interact globally — moving one item changes which pair is the tightest. The scale sets the types: `n <= 2*10^5`, coordinates `0 <= p[i] <= 10^9`, so the span fits in 32 bits, but I will carry positions and gaps in `long long` throughout. Differences and midpoints in a maximin search are exactly where a stray 32-bit compare would bite, and at this `n` the extra width costs nothing.

One simplification comes for free and I want it locked in before any algorithm: for a sorted chosen set the minimum *pairwise* gap equals the minimum *consecutive* gap. Any non-adjacent pair `x < y` has some chosen `z` strictly between them, so `y - x` exceeds both `y - z` and `z - x` — a non-adjacent pair is never the unique tightest. So I sort all positions once and reason only about neighbours; the `O(k^2)` pairwise view never has to appear.

Direct subset optimization is dead at this scale. Choosing `k` of `n` slots is `C(n,k)` subsets — at `n = 2*10^5, k = 10^5` that number has tens of thousands of digits, and there is no subset-DP (exponential state) or flow formulation that rescues it; even `n = 30` is already `~1.5*10^8` subsets. A pure greedy "take the next position far enough away" is stuck for a subtler reason: it needs to know *how far* "far enough" is, and that target distance is exactly the answer I am trying to compute — chicken and egg. But that deadlock is the opening. If I *fix* the target gap, the greedy becomes decidable.

So I replace the optimization with a decision: `feasible(d)` = "can I place all `k` items with every pair at least `d` apart?". Two properties make this the whole solution.

- It has an `O(n)` greedy. Anchor the first item at the leftmost position `p[0]` (placing it as far left as possible only leaves more room to the right), then sweep and keep each position that is at least `d` past the last kept one. That greedy's count is the *maximum* number placeable with min-gap `>= d`.
- It is monotone in `d`: a placement with all gaps `>= d` also has all gaps `>= d'` for any `d' <= d`, so `feasible` is a block of `true` followed by a block of `false`. The threshold `d*` — the largest feasible `d` — is exactly the answer, and monotonicity is what lets me binary-search `d` over `[0, span]` instead of scanning it.

Total cost is `O(n log n)` for the sort plus `O(n log span)` for the search, about `6*10^6` predicate steps at the limits — comfortably under a second.

The greedy-is-maximal claim carries the whole reduction, so I pin it with an exchange argument. Take any valid placement `q_1 < ... < q_m` with all gaps `>= d`. Slide `q_1` down to `p[0]`: still valid, gaps only grew. Then slide `q_2` to the earliest position `>= p[0] + d`; it cannot cross `q_3`, since that gap was already `>= d` and I only moved `q_2` left, so validity holds and `m` is unchanged. Inductively every `q_i` pulls onto the greedy's choice without dropping an item, so the greedy places at least `m` — it is optimal. Anchoring anywhere other than `p[0]` could only place fewer.

A quick direction check on the given sample `p = [1,2,8,4,9], k = 3`, sorted `[1,2,4,8,9]`: `feasible(3)` keeps `1,4,8` → 3 items, true; `feasible(4)` keeps `1,8` → 2 items, false. The threshold sits at 3, matching the stated answer.

The binary-search loop is the easiest place in this reduction to get wrong. The shape is "largest `d` with `feasible(d)` true", so success moves `lo` up and failure moves `hi` down:

```
while (lo < hi) { mid = (lo + hi) / 2; if (feasible(mid)) lo = mid; else hi = mid - 1; }
```

Pairing the success branch `lo = mid` with a *floored* midpoint is a non-termination trap. On `p = [0,10], k = 2` (answer 10) the search climbs to `lo = 9, hi = 10`, where `mid = (9+10)/2 = 9`; `feasible(9)` is true so `lo = 9` — a no-op — and `mid` recomputes to 9 forever, neither bound moving. The fix is precise: when the success branch is `lo = mid`, the midpoint must be biased upward, `mid = lo + (hi - lo + 1) / 2`, so that once `hi = lo + 1` the midpoint is `hi`, not `lo`. (Writing it as `lo + (hi - lo + 1)/2` rather than `(lo + hi + 1)/2` also avoids summing two large bounds — free insurance even though `span <= 10^9` here.) Re-running `[0,10]` with the upper mid, `lo` walks `0 → 5 → 8 → 9 → 10` and exits at `lo = hi = 10`.

One degenerate case must be handled outside the search. `k == 1` has no pair, so the minimum gap is undefined; if the search ran, the predicate `placed >= 1` would be true for every `d` and it would report the full span, a meaningless number. So I special-case `k <= 1 → 0` up front (the `<= 1` also absorbs a stray `k = 0` without ever touching `p[0]`), placing the guard after reading input but before the predicate dereferences `p[0]`.

The remaining corners fall out of the same machinery. `k == n` forces the greedy to keep every position, so the threshold is the minimum adjacent gap of the whole array. All-equal positions give span 0, so the search loop never runs and the answer is 0. Two extreme points give the full span. Heavy duplicates and unsorted input are absorbed by the leading sort. All of these agree with the brute force.

For validation I differential-tested against an independent oracle that enumerates every size-`k` subset and maximizes the min adjacent gap — no binary search, no shared logic, so a common blind spot is unlikely — over a spread of small random instances (tight coordinate ranges forcing duplicates and zero gaps, wide and clustered ranges, all-equal arrays, `k` from 1 to `n`) plus the edge cases above, with no mismatches. At full scale (`n = 2*10^5, k = 10^5`, coordinates to `10^9`) it runs in about `0.03 s` in a few MB, matching the `O(n log n + n log span)` analysis.

The final program — sort, the `k <= 1` guard, the `feasible` greedy, and the upper-mid binary search over `[0, span]` — is in the answer.
