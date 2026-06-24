**Problem.** Given altitude readings `a[0..n-1]` (values may be negative) and a tolerance `D`, count the contiguous blocks `[l, r]` that span **at least two days** (`l < r`) and satisfy `max(a[l..r]) - min(a[l..r]) <= D`. Read `n`, `D`, and the values from stdin; print the count.

**Why the obvious brute force is too slow.** Enumerating every block `[l, r]` with a running max/min is `O(n^2)`; at `n = 2*10^5` that is `~4*10^10` operations, far over a 1-second limit. It is correct, so it is useful only as a verification oracle, not as the shipped solution.

**Key idea — two pointers with monotonic deques.** Fix the right end `r`. Shrinking a window can only lower the max and raise the min, so the spread `max - min` is non-increasing as the left end `l` rises; hence there is a threshold `L(r)` with `[l, r]` valid exactly for `l >= L(r)`, and `L(r)` is non-decreasing in `r`. So a single left pointer that **only moves forward** is correct. Maintain two monotonic deques of indices — a max-deque (values decreasing, front = window max) and a min-deque (values increasing, front = window min) — to read the current spread in `O(1)`. For each `r`: push `r` into both deques (popping dominated backs), then advance `l` while the spread exceeds `D`, evicting any deque front that falls outside the window. Total work is `O(n)` amortized.

**Counting per right end.** Once the smallest valid left end is `l`, every `l'` in `[l, r]` gives a valid block `[l', r]`. A block has length `>= 2` iff `l' <= r - 1`, so the number of qualifying blocks ending at `r` is `r - l` (which is `0` exactly when `l == r`, i.e. only the single-day window is valid). Sum `r - l` over all `r`.

**Correctness.** The forward-only `l` is justified by the monotonicity of `L(r)`; the deques give the exact window max/min, so the shrink loop stops at the true `L(r)`; and `r - l` counts precisely the length-`>= 2` valid blocks ending at `r`. Summing over `r` partitions all valid blocks by their right end, so no block is missed or double-counted.

**Two pitfalls to get right.**
1. *The length-`>= 2` off-by-one.* The contribution is `r - l`, **not** `r - l + 1`. Using `r - l + 1` counts the degenerate single-day window `[r, r]` for every `r`, over-counting by exactly `n`. A trace of `[4, 4]` with `D = 0` returning `3` instead of `1` exposes this (the extra `2` are the illegal `[0,0]` and `[1,1]`).
2. *The deque eviction boundary.* After `l++`, the window is `[l, r]`, so index `l` is still inside; evict only when `front < l`, never `<=`. Using `<=` drops the element sitting on the new left boundary and corrupts the running max/min. A trace of `[0, 5, 4]` with `D = 2`, where a deque front index equals the new `l`, confirms `<` keeps it correctly.

**Edge cases.** `n = 0` and `n = 1` -> `0` (no two-day block; the `r - l` formula yields `0`, whereas `r - l + 1` would wrongly give `1` at `n = 1`). `D = 0` with distinct values -> `0`; with all equal values of length `n` -> `C(n, 2)`. Overflow: the count can reach `C(2*10^5, 2) ~= 2*10^10`, which overflows 32-bit, so the accumulator is `long long`; `D` (up to `2*10^9`) and the spread are also handled in 64-bit.

**Complexity.** `O(n)` time, `O(n)` space for the deques (each index is pushed and popped at most once).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Count contiguous blocks [l, r] (0-indexed, inclusive) of length >= 2 with
    // max(block) - min(block) <= D.
    //
    // Two pointers with two monotonic deques (max-deque and min-deque holding
    // indices). For each right endpoint r we advance l to the smallest index so
    // that window [l, r] satisfies max - min <= D. Every l' in [l, r] yields a
    // valid window [l', r]; among those, the ones of length >= 2 are exactly the
    // l' in [l, r-1], i.e. (r - l) windows (clamped at 0 when l == r).
    deque<int> mx, mn; // indices, mx decreasing values, mn increasing values
    int l = 0;
    long long answer = 0;
    for (int r = 0; r < n; r++) {
        while (!mx.empty() && a[mx.back()] <= a[r]) mx.pop_back();
        mx.push_back(r);
        while (!mn.empty() && a[mn.back()] >= a[r]) mn.pop_back();
        mn.push_back(r);

        // Shrink from the left until window is valid.
        while (a[mx.front()] - a[mn.front()] > D) {
            l++;
            if (mx.front() < l) mx.pop_front();
            if (mn.front() < l) mn.pop_front();
        }

        // Windows [l', r] with l <= l' <= r are all valid. Length >= 2 needs
        // l' <= r - 1, so the count is (r - l). When l == r there are none.
        answer += (long long)(r - l);
    }

    cout << answer << "\n";
    return 0;
}
```
