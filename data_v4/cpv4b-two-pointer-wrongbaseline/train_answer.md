**Problem.** A drone logs `n` integer per-tick net altitude changes `a[0..n-1]` (positive climbing,
negative descending). Given a threshold `S`, find the **shortest** contiguous window `[l, r]` whose
sum `a[l] + ... + a[r]` is `>= S`, and output its length `r - l + 1`, or `-1` if no window reaches
`S`. Windows are non-empty. Read `n`, `S`, then the `n` values from stdin; print one integer.

**Why the obvious two-pointer is wrong.** The textbook sliding-window two-pointer ("advance `r`,
shrink from `l` while the window sum stays `>= S`") solves the **all-positive** "smallest subarray
with sum >= S" because window sum is then monotone in the endpoints. Here `a[i]` can be **negative**,
which breaks that monotonicity, and the left pointer's one-way (non-decreasing) motion silently
discards left endpoints it can never revisit. On `a = [4, 2, 2, 2, 2, -3, 3, 6]`, `S = 7` the naive
window returns `3`, but `a[6..7] = 3 + 6 = 9 >= 7` has length `2`. The dip at the `-3` makes
`prefix[6] = 9 < prefix[5] = 12`, so the longer-window-means-larger-sum assumption fails. The naive
baseline is discarded.

**Key idea — monotonic deque over prefix sums.** Let `prefix[0] = 0`, `prefix[k] = a[0]+...+a[k-1]`.
A window has sum `prefix[r] - prefix[l]`, so we want the smallest `r - l` with
`prefix[r] - prefix[l] >= S`. Scan `r = 0..n`, keeping a deque of candidate left indices with
**increasing** `prefix` values:

- *Front-pop (extract):* while `prefix[r] - prefix[front] >= S`, record `r - front` and pop the
  front. Safe forever because `r` only grows, so this `front` can never give a shorter window later.
- *Back-pop (dominance):* while `prefix[back] >= prefix[r]`, pop the back. A later index with a
  smaller-or-equal prefix beats `back` on both validity (smaller prefix is easier to clear `S`) and
  length (larger index is shorter), so `back` is useless.

Each index is pushed once and popped at most once, giving `O(n)`.

**Pitfalls.**
1. *Wrong baseline.* The positive-only sliding window is incorrect for signed values; verify on a
   case (it returns `3` vs the true `2` above) rather than trusting the "standard" algorithm.
2. *Loop order.* Run the front-pop **before** the back-pop within each `r`. Doing dominance first can
   delete the front the current `r` still needs to pair with — a trace of `a = [-1], S = -1` wrongly
   yields `-1` instead of `1`.
3. *Overflow.* With `n` up to `2*10^5`, `|a[i]|` up to `10^9`, and `S` up to `2*10^14`, prefix sums
   reach `~2*10^14`; use `long long` for `prefix` and `S`. An `int` is a silent wrong-answer.

**Edge cases.** `n = 0` -> `-1` (no window). `S <= 0` -> a length-1 window can qualify (`[-1], S=-1`
gives `1`). All-negative with `S > 0` -> `-1`. `S` above the total achievable gain -> `-1`; exactly
the total -> length `n`.

**Complexity.** `O(n)` time, `O(n)` space for the prefix array and deque.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // prefix[k] = a[0] + ... + a[k-1], prefix[0] = 0, length n+1.
    // A window [l, r) (0 <= l < r <= n) has sum prefix[r] - prefix[l];
    // we want the shortest window with prefix[r] - prefix[l] >= S, i.e. minimize r - l.
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + a[i];

    // Monotonic deque of candidate left endpoints (indices into prefix), with
    // strictly increasing prefix values. For each r we pop from the front while
    // prefix[r] - prefix[front] >= S (that left can never be beaten by a larger r,
    // since r only grows), and we keep the deque increasing from the back so a
    // smaller-or-equal prefix at a later index dominates earlier larger ones.
    deque<int> dq;
    int best = INT_MAX;
    for (int r = 0; r <= n; r++) {
        while (!dq.empty() && prefix[r] - prefix[dq.front()] >= S) {
            best = min(best, r - dq.front());
            dq.pop_front();
        }
        while (!dq.empty() && prefix[dq.back()] >= prefix[r]) {
            dq.pop_back();
        }
        dq.push_back(r);
    }

    cout << (best == INT_MAX ? -1 : best) << "\n";
    return 0;
}
```
