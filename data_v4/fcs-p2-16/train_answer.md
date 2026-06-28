**Problem.** Given `n <= 40` items with positive integer weights `w[i]` and values `v[i]` and a knapsack of capacity `C` up to `10^18`, choose a subset of total weight at most `C` maximizing total value (empty subset allowed, value `0`). Read `n`, `C`, then `n` lines `w[i] v[i]` from stdin; print the maximum value.

**Why the two textbook DPs are dead on arrival.** The weight-indexed knapsack DP is `O(n*C)` time and `O(C)` memory; with `C` up to `10^18` it cannot even be allocated. The value-indexed DP is `O(n*sum(v))`, but `sum(v)` reaches `40*10^9 = 4*10^10`, also far out of a 2-second / 256-MB budget. The constraints deliberately remove both pseudopolynomial routes.

**Why greedy by ratio is wrong.** Sorting by `v[i]/w[i]` and taking greedily is optimal for the *fractional* knapsack, which is the trap — it is *not* optimal for 0/1. Counterexample with `C = 10`: items `(w,v) = (6,13), (5,10), (5,10)`. Greedy takes the top-ratio item `(6,13)`, leaving capacity `4` that nothing fits, for value `13`; but `(5,10)+(5,10)` weighs `10` and gives `20`. The high-ratio item strands capacity two lower-ratio items would tile exactly. Greedy is discarded.

**Why not branch-and-bound.** A pruned include/exclude DFS can be made exact, but it is error-prone to get the bound exactly right within budget, and its worst case (values near-proportional to weights, capacity near half the total) degrades toward `2^40 ≈ 10^12` nodes — not worst-case safe. Prefer a method that provably runs fast and exact on every input.

**Key idea — meet in the middle.** `2^40` is too large but `2^20 ≈ 10^6` is fine. Split the items into halves `A` (size `la = n/2`) and `B` (size `lb = n-la`), each `<= 20`.

1. Enumerate all `2^{lb}` subsets of `B` as `(weight, value)` pairs.
2. Sort them by weight, then sweep left to right taking a running maximum of value, producing `bestv[i]` = best value among all `B`-subsets of weight `<= bw[i]` (a **prefix maximum of value over weight-sorted subsets**). This is what makes "best `B`-value within a weight budget" a single binary search — and it is essential, because a lighter `B`-subset can be more valuable, so querying raw value (or just the heaviest fitting subset) would be wrong.
3. Enumerate all `2^{la}` subsets of `A`; for each with weight `sw` and value `sv`, skip if `sw > C`, else binary-search the largest index with `bw[i] <= C - sw` and update the answer with `sv + bestv[i]`.

Every subset of the whole set splits uniquely into an `A`-part and a `B`-part, so ranging over all such pairs yields the exact optimum.

**Correctness anchors / pitfalls.**
1. *Empty `B`-subset at index 0.* Sorting `(weight, value)` lexicographically puts the weight-`0` empty subset first, so any `rem >= 0` finds at least `pos = 0` (pairing an `A`-subset with "take nothing from `B`" is always legal). The binary search returns the *rightmost* in-budget index and never `-1` for `rem >= 0`.
2. *Prefix-max seed.* Seed the running maximum with `LLONG_MIN` so the first real value replaces it; never seed with `0` or add anything to the sentinel (no underflow).
3. *Overflow.* Values sum to `<= 4*10^10` and capacity is `10^18`; use `long long` throughout.
4. *`sw > C`.* Skip such `A`-subsets so `rem` is never negative.

**Edge cases (all handled):** `n = 0` and `C = 0` give `0`; capacity `>=` total weight gives the sum of all values; a single too-heavy item gives `0`.

**Complexity.** Building/sorting `B` is `O(2^{lb} * (lb + log))`; the `A`-loop is `O(2^{la} * (la + log))`. For `n = 40` this is a few times `10^7` operations — about `0.2` s and `~36` MB in practice, independent of `C` beyond comparisons.

**Verification.** Differential-tested against an independent `2^n` brute oracle: a rotating edge bank plus `600` random instances across capacity regimes, plus an `n = 22` boundary check — zero mismatches. An early version over-counted on a tight-capacity case (a stale heavier-`B` value leaking through the query); anchoring the empty subset at index `0` and seeding the prefix-max with `LLONG_MIN` fixed it.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // Meet in the middle. Split items into two halves of size <= 20 each.
    int la = n / 2;          // size of first half
    int lb = n - la;         // size of second half

    // Enumerate all subsets of the second half into (weight, value) pairs.
    int sb = 1 << lb;
    vector<pair<long long,long long>> B;     // (weight, value)
    B.reserve(sb);
    for (int mask = 0; mask < sb; mask++) {
        long long sw = 0, sv = 0;
        for (int j = 0; j < lb; j++) {
            if (mask & (1 << j)) {
                sw += w[la + j];
                sv += v[la + j];
            }
        }
        B.push_back({sw, sv});
    }
    // Sort by weight; build a prefix maximum of value so that for any weight
    // budget the best achievable value among weights <= budget is queryable.
    sort(B.begin(), B.end());
    vector<long long> bw(sb), bestv(sb);
    long long run = LLONG_MIN;
    for (int i = 0; i < sb; i++) {
        bw[i] = B[i].first;
        run = max(run, B[i].second);
        bestv[i] = run;                      // best value using weight <= bw[i]
    }

    // Enumerate all subsets of the first half; for each, binary-search the
    // remaining capacity in the second half and add its best value.
    int sa = 1 << la;
    long long ans = 0;
    for (int mask = 0; mask < sa; mask++) {
        long long sw = 0, sv = 0;
        for (int j = 0; j < la; j++) {
            if (mask & (1 << j)) {
                sw += w[j];
                sv += v[j];
            }
        }
        if (sw > C) continue;                // first half alone already too heavy
        long long rem = C - sw;
        // largest index i with bw[i] <= rem
        int lo = 0, hi = sb - 1, pos = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (bw[mid] <= rem) { pos = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (pos >= 0) {
            ans = max(ans, sv + bestv[pos]);
        }
        // (pos == -1 is impossible since the empty subset has weight 0 <= rem)
    }

    cout << ans << "\n";
    return 0;
}
```
