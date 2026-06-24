**Problem.** Given a stream of `n` positive pulses `a[0..n-1]` (`a[i] >= 1`) and bounds `L, R`, count the contiguous runs `[i..j]` whose sum lies in `[L, R]` (distinct by position). Read `n L R` then the `n` values from stdin; print the count.

**Key idea — reduce a range count to two "at most" counts.** The number of subarrays with sum in `[L, R]` equals `atMost(R) - atMost(L-1)`, where `atMost(X)` = number of subarrays with sum `<= X`. Each `atMost(X)` is one `O(n)` two-pointer sweep, so the whole answer is `O(n)`.

**Why two pointers work.** Every `a[i] >= 1`, so for a fixed right endpoint the window sum is *strictly decreasing* as the start moves right. Sweep `right` left to right, keep a `left` pointer and the current `sum`; after adding `a[right]`, shrink while `sum > X`. Then `[left..right]` is the longest window ending at `right` with sum `<= X`, and **every** start in `[left, right]` is valid, contributing `right - left + 1`.

**Correctness.** `atMost(R)` counts all subarrays with sum `<= R`; `atMost(L-1)` counts those with sum `<= L-1` (i.e. strictly below `L`). Their difference counts exactly the subarrays with `L <= sum <= R`. Since `L <= R`, `atMost(L-1) <= atMost(R)`, so the result is non-negative. The per-step `right - left + 1` is exact because positivity makes "sum `<= X`" a contiguous suffix of start positions.

**Pitfalls.**
1. *Over-count from shrinking once.* The shrink must be a `while`, not an `if`: a single large `a[right]` can require dropping several left elements. With an `if`, on `[1,1,5]`, `X=2` the window stays oversized and `atMost` returns `5` instead of `3` — it counts windows that violate the constraint.
2. *Off-by-one at the boundary.* It must be `atMost(L-1)`, never `atMost(L)`. Using `L` erases the windows whose sum equals `L` exactly (trace `[3]`, `L=R=3`: answer must be `1`, but `atMost(R)-atMost(L)=0`).
3. *Out-of-range `X`.* Guard `if (X < 0) return 0;` so a non-positive boundary cannot drive `left` past `right` and read `a[left]` out of bounds.
4. *Overflow.* Sums reach `~2*10^14`, `R` up to `10^18`, and the count up to `n(n+1)/2 ≈ 2*10^10`; all of `a[i]`, `sum`, and `cnt` must be `long long`. An `int` count is a silent wrong-answer on all-ones tests.

**Edge cases.** `n = 0` -> `0`; single element below `L`, above `R`, or in range all handled; window above the total -> `0`; `L = 1, R >= total` -> `n(n+1)/2`; max `n = 2*10^5` with `a[i] = 10^9` returns the exact closed form in `long long`.

**Complexity.** `O(n)` time (two linear sweeps), `O(n)` space for the input (`O(1)` beyond it).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count contiguous subarrays whose sum lies in [L, R].
// Values a[i] >= 1, so prefix sums are strictly increasing; for a fixed right
// endpoint the set of valid left endpoints is a contiguous window -> two pointers.
// We count atMost(R) - atMost(L-1), where atMost(X) = number of subarrays with sum <= X.

static long long atMost(const vector<long long>& a, long long X) {
    // Number of contiguous subarrays with sum <= X. With a[i] >= 1 and X possibly
    // negative, the window logic must handle X < 0 (answer 0).
    if (X < 0) return 0;
    long long cnt = 0, sum = 0;
    int left = 0;
    int n = (int)a.size();
    for (int right = 0; right < n; right++) {
        sum += a[right];
        while (sum > X) {            // shrink until window sum <= X
            sum -= a[left];
            left++;
        }
        // [left .. right] is the longest window ending at right with sum <= X;
        // every subarray ending at right with start in [left, right] qualifies.
        cnt += (long long)(right - left + 1);
    }
    return cnt;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;
    vector<long long> a(n);
    for (auto& x : a) cin >> x;

    long long answer = atMost(a, R) - atMost(a, L - 1);
    cout << answer << "\n";
    return 0;
}
```
