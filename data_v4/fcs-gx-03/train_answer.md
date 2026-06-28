**Problem.** Given `n` positions on a line `p[0..n-1]` (integers, possibly repeated, in any order) and an integer `k` with `1 <= k <= n`, place `k` items into `k` distinct slots so that the **minimum** pairwise distance between chosen positions is **maximized**. Output that maximum minimum gap. Read `n k` then the `n` values from stdin; print one integer. Convention: `k == 1` has no pair, output `0`.

**Why the direct approaches fail.** Enumerating which `k` of the `n` slots to pick is `C(n,k)` — astronomical at `n = 2*10^5`, so direct subset optimization is dead. A pure greedy ("take positions far apart") is stuck too: to decide whether the next position is "far enough" it needs the target gap, which is exactly the answer it is trying to compute — a chicken-and-egg deadlock.

**Key idea — binary search the answer + greedy feasibility.** Break the deadlock by *fixing* the gap and answering a decision instead of the optimization. Define `feasible(d)` = "can all `k` items be placed with every pair at least `d` apart?". Two facts make this win:

- *The decision is easy.* Sort the positions. Anchor the first item at `p[0]` (leftmost is never worse — it leaves the most room), then sweep left to right keeping each position that is at least `d` past the last kept one. The count this greedy reaches is the **maximum** number placeable with min-gap `>= d` (exchange argument: any valid placement slides leftward onto the greedy without losing items). So `feasible(d)` is "greedy places `>= k`?", in `O(n)`.
- *The decision is monotone in `d`.* If `k` items fit with all gaps `>= d`, the same positions have all gaps `>= d'` for any `d' <= d`. So `feasible` is a run of `true` then `false`, and the threshold `d*` — the largest `d` with `feasible(d)` true — is exactly the answer. Monotonicity is what licenses **binary search** over `d ∈ [0, span]`, `span = p[n-1]-p[0]`.

Also: once positions are sorted, the minimum *pairwise* gap of a chosen set equals its minimum *consecutive* gap, so we only ever reason about neighbours — no `O(k^2)`. Total cost `O(n log n + n log span)`.

**Pitfalls to get right.**
1. *Binary-search loop shape.* With the success branch `lo = mid`, the midpoint must be biased **up**: `mid = lo + (hi - lo + 1) / 2`. The floored `(lo+hi)/2` makes `mid == lo` once `hi = lo + 1`, so `lo = mid` is a no-op and the loop **never terminates**. (A trace of `[0,10], k=2` spins forever with the naive midpoint.)
2. *Degenerate `k == 1`.* No pair exists; if the search runs, the predicate is true for all `d` and it returns the meaningless full span. Special-case `k <= 1 → 0` before searching.
3. *Predicate optimality.* `feasible` must compute the *maximum* placeable count; anchoring at `p[0]` and greedily keeping the earliest far-enough position is provably optimal — do not anchor elsewhere.

**Edge cases.** `k == 1 → 0`; `k == n` → answer is the minimum adjacent gap (greedy is forced to keep all positions); all-equal positions → span `0` → `0`; two extreme points → full span; heavy duplicates and unsorted input → handled by the leading `sort`.

**Complexity.** `O(n log n)` to sort plus `O(n log(span))` for the search; `O(n)` space. About `6*10^6` predicate steps at the limits — runs in ~`0.03 s` on `n = 2*10^5`.

**Code.**

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
