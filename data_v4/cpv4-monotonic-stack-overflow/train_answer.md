**Problem.** Given `a[0..n-1]` (values may be negative), sum the **minimum** of every contiguous subarray `a[l..r]` over all `n*(n+1)/2` subarrays. Read `n` and the values from stdin, print the single sum (which may be negative).

**Why the obvious method is too slow.** Enumerating subarrays with two nested loops and a running minimum is `O(n^2)`. With `n` up to `3*10^4` that is up to `4.5*10^8` subarrays — fine as a correctness oracle, too slow for a 1 second limit.

**Key idea — per-element contribution via a monotonic stack.** Instead of grouping by subarray, group by which element supplies the minimum. Element `a[i]` is the minimum of exactly `left[i] * right[i]` subarrays, where `left[i]` is how far the left endpoint can extend (number of positions until a smaller element to the left) and `right[i]` how far the right endpoint can extend. The answer is `sum_i a[i] * left[i] * right[i]`. Both arrays come from one left-to-right and one right-to-left monotonic-stack pass, each finding the nearest smaller neighbour, so the whole thing is `O(n)`.

For the sample `a = [3, 1, 2, 4]`: contributions `3*1*1 + 1*2*3 + 2*1*2 + 4*1*1 = 3 + 6 + 4 + 4 = 17`.

**Pitfalls to get right.**
1. *Tie-breaking.* Equal values must be attributed to exactly one index or a subarray of equal minima is counted multiple times. Make the comparison **strict on one side, non-strict on the other**: the left pass pops while the top is `>= a[i]` (nearest *strictly* smaller left), the right pass pops while the top is `> a[i]` (an equal element to the right is a boundary). Symmetric non-strict-on-both double-counts — a trace of `[2, 2]` returning `8` instead of `6` exposes exactly this.
2. *Overflow (the big one).* A single product `left[i]*right[i]` reaches `~2.25*10^8`, and `a[i]*left[i]*right[i]` reaches `~2.25*10^17`; the total reaches `~4.5*10^17`. That fits in `long long` (max `9.2*10^18`) but overflows `int` (max `2.1*10^9`) by eight orders of magnitude. Make `left`, `right`, every product, and the accumulator 64-bit. An all-`int` version prints `446230528` on the `n=30000`, all-`10^9` case where the answer is `450015000000000000` — a silent wrong-answer. (No `__int128` needed; the scale bound guarantees `long long` suffices.)

**Edge cases.** `n = 0` -> `0` (no subarrays). `n = 1, a=[-5]` -> `-5` (the lone subarray's min); do **not** clamp at `0`, this sum can be negative. All-negative arrays return a negative sum. Monotone arrays: `[1,2,3]` and `[3,2,1]` both give `10`, matching brute force.

**Complexity.** `O(n)` time, `O(n)` space (two run-length arrays plus the stack).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // For each i, count subarrays in which a[i] is the (chosen) minimum.
    // left[i]  = number of consecutive positions ending at i (incl. i) for which
    //            a[i] is a STRICT minimum vs the left  (a[j] > a[i] for j in that run).
    // right[i] = number of consecutive positions starting at i (incl. i) for which
    //            a[i] is a NON-strict minimum vs the right (a[j] >= a[i]).
    // Strict on one side, non-strict on the other breaks ties so each subarray's
    // minimum is attributed to exactly one index.
    vector<long long> left(n), right(n);
    vector<int> st; // indices, values strictly increasing from bottom to top

    // left: previous index with a value STRICTLY less than a[i].
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        int prev = st.empty() ? -1 : st.back();
        left[i] = i - prev;                // run length to the left
        st.push_back(i);
    }
    st.clear();
    // right: next index with a value strictly less-OR-equal... we use next strictly
    // less to keep ties on the left side; here pop while top value strictly greater.
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        int nxt = st.empty() ? n : st.back();
        right[i] = nxt - i;                // run length to the right
        st.push_back(i);
    }

    long long answer = 0;
    for (int i = 0; i < n; i++) {
        // a[i] is the minimum of left[i] * right[i] subarrays.
        answer += a[i] * (left[i] * right[i]);
    }

    cout << answer << "\n";
    return 0;
}
```
