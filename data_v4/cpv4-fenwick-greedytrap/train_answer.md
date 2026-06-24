**Problem.** Given daily prices `a[0..n-1]` (values may be negative), pick a non-empty set of days `i1 < i2 < ... < ik` whose prices are **strictly increasing**, `a[i1] < a[i2] < ... < a[ik]`, maximizing the **sum** of the chosen prices. Read `n` and the values from stdin, print the maximum sum. For `n = 0` the answer is `0`.

**Why the obvious greedy is wrong.** "Increasing subsequence" baits two greedy heuristics: take the *longest* increasing chain, or take the *single largest* value. Both fail because the objective is the **sum**, not the length, and the constraint couples which elements can coexist. On `[1, 100, 2, 3, 4, 5, 6]` the longest increasing chain is `1,2,3,4,5,6` (sum `21`), the single largest value is `100`, but the optimal is `1, 100` (sum `101`). The heaviest increasing chain is neither the longest nor a singleton, so both greedies are discarded.

**Key idea — value-indexed DP accelerated by a max-Fenwick.** Let `f[i]` be the best score of an increasing subsequence that *ends* at index `i`:

`f[i] = a[i] + max(0, max{ f[j] : j < i and a[j] < a[i] })`

The answer is `max_i f[i]`. The `max(0, ...)` lets a chain start fresh at `i`, so negative predecessors are never forced in and all-negative inputs return the least-negative single element. The inner query — "maximum `f` over earlier elements with strictly smaller value" — is a prefix maximum on a value axis. Coordinate-compress the values to ranks `1..m` and keep a Fenwick tree supporting **prefix-max query** and **point-max update**. Process left to right: query ranks `[1 .. rank(a[i]) - 1]` (strictly smaller values), compute `f[i]`, then point-update `rank(a[i])` with `f[i]`. That is `O(n log n)` instead of the naive `O(n^2)`.

**Pitfalls to get right.**
1. *Strict vs. non-strict boundary.* Query `rank(a[i]) - 1`, NOT `rank(a[i])`. Duplicates share a rank; querying inclusive lets an *equal* predecessor extend a "strictly increasing" chain. (A trace of `[2, 2]` returning `4` instead of `2` exposes exactly this.)
2. *Negative predecessors.* Use `f = a[i] + max(bestPrev, 0)`, not `f = a[i] + bestPrev`. Forcing a negative predecessor's score in is wrong: on `[-5, -3]` it would give `-8`, but starting fresh at `-3` gives `-3`. The `max(..., 0)` is load-bearing for negatives, not just for the no-predecessor case.
3. *Sentinel safety.* Initialize empty Fenwick cells to `NEG = LLONG_MIN/4` so several `max`-folds cannot underflow; it is only ever compared, never summed (we add `a[i]` to `max(bestPrev, 0)`, never to a raw `NEG`).
4. *Overflow.* With `n` up to `2*10^5` and `|a[i]|` up to `10^9`, sums reach `~2*10^14` (and even `1..n` overflows 32-bit at `~2*10^10`); use `long long` for scores and Fenwick cells.

**Edge cases.** `n = 0` -> `0` (handled directly). `n = 1` -> the single element. Strictly decreasing -> the single largest element (no element has a smaller predecessor). Duplicate-heavy -> equal values cannot chain. All-negative -> least-negative element (chaining never helps).

**Complexity.** `O(n log n)` time (one Fenwick query + update per element after an `O(n log n)` sort), `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // Coordinate-compress the values so a Fenwick tree can be indexed by rank.
    vector<long long> srt(a);
    sort(srt.begin(), srt.end());
    srt.erase(unique(srt.begin(), srt.end()), srt.end());
    int m = (int)srt.size();
    auto rankOf = [&](long long v) -> int {
        // 1-based rank in the sorted distinct array.
        return (int)(lower_bound(srt.begin(), srt.end(), v) - srt.begin()) + 1;
    };

    // bit[r] = max f-value over compressed ranks in the Fenwick prefix ending at r.
    // We query the prefix max over ranks STRICTLY LESS THAN rank(a[i]) (values < a[i]),
    // then point-update rank(a[i]) with f[i].
    const long long NEG = LLONG_MIN / 4;
    vector<long long> bit(m + 1, NEG);

    auto queryPrefixMax = [&](int r) -> long long { // max over ranks [1..r]
        long long best = NEG;
        for (; r > 0; r -= r & (-r))
            best = max(best, bit[r]);
        return best;
    };
    auto updatePoint = [&](int r, long long val) {  // bit[r] = max(bit[r], val)
        for (; r <= m; r += r & (-r))
            bit[r] = max(bit[r], val);
    };

    long long answer = NEG;
    for (int i = 0; i < n; i++) {
        int r = rankOf(a[i]);
        // best f among earlier elements with strictly smaller value:
        long long bestPrev = queryPrefixMax(r - 1);
        long long f = a[i] + max(bestPrev, 0LL); // a[i] alone if no smaller predecessor
        updatePoint(r, f);
        answer = max(answer, f);
    }

    cout << answer << "\n";
    return 0;
}
```
