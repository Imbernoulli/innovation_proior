**Problem.** Given `a[0..n-1]` (values may be negative, zero, or positive), score a non-empty window `a[l..r]` as `min(a[l..r]) * (r - l + 1)`. The empty window is allowed and scores `0`. Read `n` and the values from stdin, print the maximum score over all windows (so the answer is at least `0`).

**Why the naive scans fail.** Brute force over all `(l, r)` pairs is `O(n^2)` — correct but too slow at `n = 2*10^5`. And the histogram reflex "for each index take the *widest* window, because wider is better" is only valid when minimums are non-negative. With negatives, a wider window whose minimum is `-4` scores `-4 * width`, which gets *more negative* as the width grows; even with positives, a wider window can pull in a smaller minimum (on `[2,1,5,6,2,3]` the full array scores `6` but the window `[5,6]` scores `10`). So width is not monotone and the sign matters.

**Key idea — monotonic stack over "I am the minimum" windows.** Every window's minimum is attained at some index. For each index `i`, find the *widest* window in which `a[i]` is the minimum and score `a[i] * width_i`; the global max over `i` then equals the maximum over all windows, in `O(n)`. The width comes from nearest-smaller-element on both sides:

- `left[i]` = nearest index on the left with value **strictly less** than `a[i]` (`-1` if none),
- `right[i]` = nearest index on the right with value **less-or-equal** to `a[i]` (`n` if none),
- `width_i = right[i] - left[i] - 1`.

Both are computed with a single monotonic stack pass each.

**Pitfalls to get right.**
1. *Tie-break / equal runs.* Use **strict** comparison on the left pass (`pop while a[top] >= a[i]`) and **non-strict** on the right pass (`pop while a[top] > a[i]`). Symmetric operators on both sides double-count equal runs and can inflate a width past any real window, producing a too-large score. The asymmetry makes each maximal equal-run scored at full width by exactly one index (trace `[3,3,3]` -> answer `9`, claimed once by the rightmost `3`).
2. *Base case / sign.* The empty window scores `0`, so initialize `best = 0` and only ever raise it. Seeding `best` from a real window asserts that some non-empty window must be chosen, which returns a negative on all-negative inputs (trace `[-3,-1,-4]`: best non-empty is `-1`, but the answer must be `0`).
3. *Overflow.* With `n` up to `2*10^5` and `|a[i]|` up to `10^9`, a positive window's score reaches `~2*10^14`; use `long long`, and cast the width to 64-bit before multiplying. An `int` product is a silent wrong-answer on large positive tests.

**Edge cases (all handled by the `best = 0` floor + the recurrence).** `n = 0` (and empty input) -> `0`; a single negative -> `0`; all negatives -> `0`; all zeros -> `0`; equal values straddling a larger one are scored correctly by the asymmetric operators.

**Complexity.** `O(n)` time (two stack passes plus one scoring pass), `O(n)` extra space for the boundary arrays and the stack.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) { cout << 0 << "\n"; return 0; } // empty input -> empty subarray
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // For each i, find the widest window [L, R] in which a[i] is a minimum.
    // left[i]  = index of nearest position to the left with a value STRICTLY less than a[i] (-1 if none)
    // right[i] = index of nearest position to the right with a value <= a[i] (n if none)
    // This (strict-left, non-strict-right) tie-break makes each window counted once.
    vector<int> left(n), right(n);
    vector<int> st; // monotonic stack of indices

    st.clear();
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        left[i] = st.empty() ? -1 : st.back();
        st.push_back(i);
    }
    st.clear();
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        right[i] = st.empty() ? n : st.back();
        st.push_back(i);
    }

    // Empty subarray is allowed and scores 0; that is the base value.
    long long best = 0;
    for (int i = 0; i < n; i++) {
        long long width = (long long)(right[i] - left[i] - 1);
        long long score = a[i] * width;          // min over the window is exactly a[i]
        if (score > best) best = score;
    }

    cout << best << "\n";
    return 0;
}
```
