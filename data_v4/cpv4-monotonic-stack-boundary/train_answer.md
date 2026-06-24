**Problem.** Given `a[0..n-1]`, sum the minimum over every contiguous subarray `a[l..r]`
(`0 <= l <= r <= n-1`) and print `S mod 1000000007`. There are `n*(n+1)/2` subarrays, so for
`n = 2*10^5` enumeration is impossible. Read `n` and the values from stdin, print the modular sum.

**Key idea — count each element's ownership.** The minimum of a subarray is one element, so rewrite
`S = sum_i c[i] * a[i]`, where `c[i]` is the number of subarrays whose minimum is `a[i]`. Index `i`
owns the subarrays whose left endpoint lies in `(prev, i]` and whose right endpoint lies in
`[i, nxt)`, where `prev`/`nxt` are the nearest "smaller" elements on each side. Hence
`c[i] = (i - prev) * (nxt - i)`. Two monotonic-stack passes (left-to-right for `prev`, right-to-left
for `nxt`) compute every boundary in `O(n)`.

**Pitfalls.**
1. *Inclusive span (off-by-one).* The reach counts must **include** index `i` itself, since `[i,i]`
   is a valid subarray: `left = i - prev`, `right = nxt - i`. The histogram-rectangle habit of
   writing `i - prev - 1` (the index as an excluded wall) is wrong here — it drops the single-cell
   subarray, and a trace of `[5]` returns `0` instead of `5`.
2. *Tie-breaking strictness.* When equal elements share a subarray's minimum, the subarray must be
   credited to exactly one owner. Use **strict** `>` on the left pop and **non-strict** `>=` on the
   right pop (or the mirror). Using the *same* strictness on both sides double-counts ties: `[2,2,2]`
   yields `20` instead of `12`. The asymmetric split makes the leftmost equal element the sole owner,
   and the invariant `sum_i c[i] = n(n+1)/2` then holds.
3. *Overflow.* `c[i]` reaches `~10^10` and `c[i]*a[i]` reaches `~10^19`, past `long long`. Reduce
   `left`, `right`, and a negative-safe `val = ((a[i] % MOD) + MOD) % MOD` modulo `MOD` *before*
   multiplying, so every product stays under `10^18` and the answer stays in `[0, MOD)`.

**Edge cases.** `n = 0` -> no subarrays -> `0`; `n = 1` -> the lone value mod `MOD` (e.g. `-7`
prints `1000000000`); strictly increasing/decreasing arrays give `left[i]` of `1`/`i+1`; arrays of
all-equal values are exactly where the tie convention is tested.

**Complexity.** `O(n)` time, `O(n)` space (two index stacks plus two reach arrays).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    const long long MOD = 1000000007LL;

    // For each i, count subarrays whose minimum is a[i].
    // left[i]  = number of consecutive elements strictly greater than a[i]
    //            immediately to the left (so a[i] is the min over that reach).
    // right[i] = number of consecutive elements >= a[i] immediately to the right.
    // The strict/non-strict split (strict left, non-strict right) makes each
    // subarray credited to exactly one index when minima tie.
    vector<long long> left(n), right(n);

    // left: previous strictly-smaller-OR-EQUAL element acts as the wall.
    // Pop while stack top value > a[i]  (those are strictly greater -> in reach).
    {
        vector<int> st; // indices, increasing-ish by value (non-decreasing)
        for (int i = 0; i < n; i++) {
            while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
            int prev = st.empty() ? -1 : st.back();
            left[i] = i - prev;           // count of positions in (prev, i]
            st.push_back(i);
        }
    }
    // right: next strictly-smaller element is the wall.
    // Pop while stack top value >= a[i] (those are >= -> still in reach).
    {
        vector<int> st;
        for (int i = n - 1; i >= 0; i--) {
            while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
            int nxt = st.empty() ? n : st.back();
            right[i] = nxt - i;           // count of positions in [i, nxt)
            st.push_back(i);
        }
    }

    long long ans = 0;
    for (int i = 0; i < n; i++) {
        long long cnt = (left[i] % MOD) * (right[i] % MOD) % MOD;
        long long val = ((a[i] % MOD) + MOD) % MOD;
        ans = (ans + cnt * val) % MOD;
    }
    cout << ans << "\n";
    return 0;
}
```
