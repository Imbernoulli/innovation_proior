**Problem.** Given `a[0..n-1]` of positive integers, assign every subarray `a[l..r]` to a single *owner*: the index of its minimum, breaking ties by the **leftmost** position attaining the minimum. With `c[i]` the number of subarrays owned by `i`, output `S = (sum_i i*c[i]) mod (10^9+7)` (0-based `i`). Read `n` and the values from stdin, print `S`.

**Key idea — per-index contribution via a monotonic stack.** For a fixed `i`, the subarrays owned by `i` form a rectangle `(L, i] x [i, R)`, so `c[i] = (i - L)*(R - i)`. Under the *leftmost-minimum* tie-break the two barriers are **asymmetric**:

- `L` = nearest index `j < i` with `a[j] <= a[i]` (an equal element on the left is a more-left minimum, so it must block — use `<=`).
- `R` = nearest index `j > i` with `a[j] < a[i]` (an equal element on the right keeps `i` as the leftmost minimum, so it does *not* block — use `<`).

Compute both in two linear nearest-smaller passes with a stack: left pass pops while `a[top] > a[i]`; right pass pops while `a[top] >= a[i]`. Then accumulate `i * (i-L)*(R-i)` modulo `p`. `O(n)`.

**Self-check invariant.** Every subarray has exactly one owner, so `sum_i c[i] = n(n+1)/2` for any input. Use it as an oracle: a double-count overshoots, a dropped subarray undershoots.

**Pitfalls.**
1. *Symmetric `<`/`<` is wrong.* If both barriers use strict nearest-smaller, an equal element blocks neither side and a subarray spanning two equal minima is claimed by both. Trace `[2,2]`: it yields `c=[2,2]`, sum `4`, but the truth is `c=[2,1]`, sum `3 = 2*3/2`. The leftmost rule needs `<=` on the left, `<` on the right so the shared subarray is credited only to the leftmost owner.
2. *Overflow / wrong accumulator type.* A single `c[i]` reaches `~n(n+1)/2 ~ 2*10^10` and the weighted sum `~10^15`; an `int` accumulator wraps before `% p` runs. Build counts in 64-bit and reduce the running sum mod `p` every iteration.
3. *Off-by-one in the weight.* The index is 0-based, so index 0 contributes `0`. Weighting by `i+1` (or looping `1..n`) silently shifts the whole answer; the sample `S = 26` (with the `0*1` term zero) pins the convention.

**Edge cases.** `n = 0` -> `0` (loop never runs). `n = 1` -> `0` (the lone subarray sits at index 0, weight 0). All-equal `[5]*6` -> only index 0 reaches `L=-1`, others have `L=i-1`; `c=[6,5,4,3,2,1]`, sum `21=6*7/2`, `S=35`. Strictly decreasing -> no `<=` barrier on the left, `L[i]=-1`, `R[i]=i+1`. The `n(n+1)/2` invariant holds in all of these.

**Complexity.** `O(n)` time (two stack passes + one accumulation), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // c[i] = number of subarrays whose minimum is a[i], with ties broken by the
    // LEFTMOST minimum index. a[i] owns [l, r] (l <= i <= r) iff:
    //   - no element < a[i] in [l, r]            (else that element's value is the min)
    //   - no element <= a[i] strictly left of i  (an equal element to the left is a
    //                                              more-left minimum => it owns instead)
    //   - elements equal to a[i] to the RIGHT are fine (i stays leftmost min).
    // So:  left[i]  = nearest j < i with a[j] <= a[i]   -> l in (left[i], i]
    //      right[i] = nearest j > i with a[j] <  a[i]   -> r in [i, right[i])
    //      c[i] = (i - left[i]) * (right[i] - i).
    // Mixing <= / < across the two sides is what prevents double counting equal mins.

    vector<int> left_b(n), right_b(n);
    vector<int> st;
    st.reserve(n);

    // left[i]: previous index with a[j] <= a[i]; pop while a[top] > a[i].
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        left_b[i] = st.empty() ? -1 : st.back();
        st.push_back(i);
    }
    st.clear();
    // right[i]: next index with a[j] < a[i]; pop while a[top] >= a[i].
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        right_b[i] = st.empty() ? n : st.back();
        st.push_back(i);
    }

    long long ans = 0;
    for (int i = 0; i < n; i++) {
        long long cnt = (long long)(i - left_b[i]) * (long long)(right_b[i] - i); // exact, fits in 64-bit
        cnt %= MOD;
        ans = (ans + (long long)(i % MOD) * cnt) % MOD;
    }

    cout << ans << "\n";
    return 0;
}
```
