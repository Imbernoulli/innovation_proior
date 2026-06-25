**Problem.** Given `n` ice-thickness readings `a[0..n-1]` (values may be negative) and a tolerance `D`, count the contiguous stretches `[i, j]` (`0 <= i <= j < n`) whose range `max(a[i..j]) - min(a[i..j])` is **strictly less than** `D`. Read `n` and `D`, then the readings, from stdin; print the count. A singleton `[i, i]` has range `0`, so it is stable iff `D > 0`. The answer can reach `n*(n+1)/2 ≈ 2*10^10`.

**Key idea — sliding window with two monotonic deques.** The predicate is monotone in the window: if `[i, j]` is stable, every sub-window is stable, because shrinking a window cannot increase its range. So for a fixed right end `j`, the stable starts form a suffix `[L(j), j]`, and as `j` increases `L(j)` only moves right. Sweep `right` from `0` to `n-1`, keep `left` = smallest stable start, and add the number of stable windows ending at `right`, which is `right - left + 1`. Maintain the window max in a **decreasing** deque (front = max index) and the window min in an **increasing** deque (front = min index); each index enters and leaves each deque once, so the sweep is amortized `O(n)`.

For each `right`: push `right` onto both deques (popping smaller-or-equal from the max deque's back, larger-or-equal from the min deque's back), then advance `left` while the window is unstable, popping any deque front that falls out of `[left, right]`, and finally add `right - left + 1`.

**Pitfalls.**
1. *Strict vs non-strict (the deciding boundary).* "Stable" is `range < D`, so the shrink loop must run on the negation `range >= D`, not `range > D`. On a window whose range equals `D` exactly the two disagree: `> D` leaves `left` too small and over-counts. Trace `D = 4, a = [1, 5, 2]`: the correct `>= D` gives `4`; the wrong `> D` gives `5`, miscounting `[1, 5]` (range exactly `4`).
2. *Empty-window deque crash.* When `D` is small (e.g. `D = 0`), even a singleton is unstable, so `left` runs to `right + 1` and both deques empty. Re-reading `a[maxd.front()]` then dereferences an empty deque (undefined behaviour / segfault). Guard the shrink loop with `left <= right`; the empty window correctly contributes `right - left + 1 = 0`.
3. *Overflow.* The count reaches `~2*10^10` (past 32-bit) and `D`/ranges reach `2*10^9`; accumulate and compare in `long long`.

**Edge cases.** `n = 0` -> `0`; single element with `D = 0` -> `0`, with `D = 1` -> `1`; all-equal array with `D = 1` -> `n*(n+1)/2` (e.g. `n = 4` -> `10`); any array with `D = 0` -> `0`; mixed-sign readings handled by 64-bit arithmetic.

**Complexity.** `O(n)` time (each index pushed/popped from each deque once, `left` advances at most `n` times total), `O(n)` space for the deques. Runs `n = 2*10^5` in about `0.05 s`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Count contiguous subarrays whose (max - min) < D, via a sliding window.
    // maxd holds indices with decreasing values (front = window max),
    // mind holds indices with increasing values (front = window min).
    deque<int> maxd, mind;
    long long answer = 0;
    int left = 0;
    for (int right = 0; right < n; right++) {
        while (!maxd.empty() && a[maxd.back()] <= a[right]) maxd.pop_back();
        maxd.push_back(right);
        while (!mind.empty() && a[mind.back()] >= a[right]) mind.pop_back();
        mind.push_back(right);

        // Shrink from the left until the window [left, right] satisfies max - min < D.
        // Guard left <= right so the deques are never empty when we read their fronts.
        while (left <= right && a[maxd.front()] - a[mind.front()] >= D) {
            left++;
            if (maxd.front() < left) maxd.pop_front();
            if (mind.front() < left) mind.pop_front();
        }
        // All windows ending at 'right' with start in [left, right] are stable.
        // If left == right+1 the window is empty and this contributes 0.
        answer += (long long)(right - left + 1);
    }

    cout << answer << "\n";
    return 0;
}
```
