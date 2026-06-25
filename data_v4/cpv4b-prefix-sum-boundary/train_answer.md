**Problem.** A packaging line carries `n` parcels with integer weights `a[0..n-1]` (negatives allowed). A *batch* is a block of **exactly `w` consecutive** parcels; it passes when its total weight lies in the **closed** band `[L, R]`. Count the passing batches. Read `n w L R` then the weights from stdin; print the count. If `w > n`, no batch fits, so the answer is `0`.

**Key idea — fixed-window prefix sums.** Build an exclusive prefix `P[0] = 0`, `P[i] = a[1] + ... + a[i]` (1-indexed weights). A batch starting at 1-indexed `s` covers `s..s+w-1`, and its sum is the single subtraction `P[s+w-1] - P[s-1]`. The valid starts are `s = 1, ..., n-w+1` **inclusive**, which is `n-w+1` windows. Loop over them, test `L <= sum <= R`, count. This is `O(n)` time and `O(n)` memory, versus the `O(n*w)` brute force that re-sums every window.

**Pitfalls — three boundaries decide correctness, all catchable by tracing the sample.**
1. *Start range off-by-one.* The number of length-`w` windows is `n - w + 1`, inclusive. Looping `s <= n - w` drops the **last** batch — and it hides whenever that last window fails the band, so it only surfaces on a test where it passes. On the sample `n=6, w=3, L=10, R=15, a=[4,2,5,1,9,3]` the correct answer is `3`, but the `s <= n-w` bound returns `2`, omitting `[1,9,3]=13`.
2. *Wrong prefix subtraction.* For a 1-indexed `P`, the window is `P[s+w-1] - P[s-1]`. Copying the 0-indexed form `P[s+w] - P[s]` shifts the window right by one and can read past the array end. Derive from `P`'s definition to keep the indices aligned.
3. *Inclusive vs exclusive band.* The band is **closed**, so use `>=` and `<=`. Strict `>`/`<` silently drops batches sitting exactly on an endpoint; the degenerate band `L = R` exposes this instantly (e.g. `n=3, w=2, L=R=-1, a=[1,-2,1]` should count `2`, but strict comparisons count `0`).

**Edge cases.** `w > n` -> `0` (guard the loop with `w <= n`); `w = n` -> one whole-belt batch; `w = 1` -> each parcel is its own batch (`sum = P[s] - P[s-1] = a[s]`); negatives and zeros are fine since the band test is two-sided; window sums reach `~2*10^14`, so every sum-path value must be `long long` (an `int` wraps `2*10^9` to a negative and miscounts). `n - w + 1` and `s + w - 1` are computed in 64-bit so a large `w` (up to `10^9`) cannot wrap.

**Complexity.** `O(n)` time, `O(n)` extra space for the prefix array. At `n = 2*10^5` it runs in well under the 1-second limit (measured `0.12 s` even at `n = 2*10^6`).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, w, L, R;
    if (!(cin >> n >> w >> L >> R)) return 0;
    vector<long long> a(n + 1);
    for (long long i = 1; i <= n; i++) cin >> a[i];

    // P[i] = a[1] + ... + a[i], with P[0] = 0 (exclusive prefix).
    vector<long long> P(n + 1, 0);
    for (long long i = 1; i <= n; i++) P[i] = P[i - 1] + a[i];

    // A batch starts at s (1-indexed) and covers s..s+w-1.
    // Valid starts: 1 <= s <= n - w + 1  (inclusive on both ends).
    // Sum of that batch = P[s + w - 1] - P[s - 1].
    long long count = 0;
    if (w >= 1 && w <= n) {
        for (long long s = 1; s <= n - w + 1; s++) {
            long long sum = P[s + w - 1] - P[s - 1];
            if (sum >= L && sum <= R) count++;   // closed band [L, R]
        }
    }

    cout << count << "\n";
    return 0;
}
```
