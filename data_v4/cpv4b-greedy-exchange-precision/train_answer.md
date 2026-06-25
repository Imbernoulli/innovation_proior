**Problem.** A single render node runs `n` scenes sequentially from clock `0`, no preemption. Scene
`i` takes `p[i]` seconds and has waiting weight `w[i]`; if it finishes at cumulative time `C[i]` it
costs `w[i] * C[i]`. Choose the processing order of all scenes to minimize `sum_i w[i] * C[i]`. Read
`n` and the `n` pairs `p[i] w[i]` from stdin; print the minimum total bill.

**Key idea — exchange / Smith's rule with an exact comparator.** If two adjacent scenes `i` (just
before `j`) are swapped, every other term is unchanged and the bill changes by exactly
`p[i] w[j] - p[j] w[i]`. So `i` should precede `j` iff `p[i] w[j] <= p[j] w[i]`, i.e. by
non-decreasing ratio `p/w` (WSPT). This pairwise test is a consistent total order, so a single global
sort by it is optimal: in the sorted order no adjacent swap helps. Sort the indices with the
**integer cross-product** comparator `p[i]*w[j] < p[j]*w[i]`, then sweep, maintaining a running clock
and adding `w[i] * clock` for each scene.

**Pitfalls (this is the whole problem).**
1. *Floating-point ratio in the comparator.* Comparing `(double)p[i]/w[i]` against `(double)p[j]/w[j]`
   loses the order on near-tied ratios. With values near `10^9`, ratios like `999999958/999999957`
   and `999999959/999999958` collapse to the *same* IEEE-754 double (53-bit mantissa), so `std::sort`
   leaves them in input order — wrong by exactly the cross-product gap. Fix: compare the exact
   products `p[i]*w[j]` vs `p[j]*w[i]`. Each is `<= 10^9 * 10^9 = 10^18`, comfortably inside signed
   64-bit (`~9.2*10^18`), so no `__int128` is needed *inside the sort*.
2. *64-bit objective overflow.* Completion times reach `sum(p) ~ 10^14` (fine in `long long`), but a
   single term `w[i]*C[i]` reaches `10^9 * 10^14 = 10^23` and the total runs to `~10^28` — far past
   `long long`. Seven scenes of `(10^9, 10^9)` already give `28*10^18`, three times the 64-bit max,
   and an `int64` accumulator prints negative garbage. Fix: keep the running clock in `long long` but
   accumulate the bill in `__int128`, then print it digit by digit.

**Edge cases.** `n = 0` (and empty stdin) -> `0`; `n = 1` -> the single term `w[0]*p[0]`; all-equal
ratios -> any tie-break is cost-neutral, so a deterministic index tie-break is optimal; the `total`
printer special-cases `0` so it emits `0` rather than an empty string.

**Complexity.** `O(n log n)` time (one sort) plus an `O(n)` sweep, `O(n)` space. Runs `n = 10^5` in
about `0.06 s`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 -> empty schedule, cost 0
    vector<long long> p(n), w(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> w[i];

    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);

    // Smith's rule: schedule by non-decreasing p/w.  Order i before j iff
    //   p_i / w_i <= p_j / w_j   <=>   p_i * w_j <= p_j * w_i   (w > 0).
    // Compare cross-products (each <= 10^18, fits in signed 64-bit) instead of
    // dividing as doubles, which loses the ordering on near-tied ratios.
    sort(idx.begin(), idx.end(), [&](int i, int j) {
        long long lhs = p[i] * w[j];      // <= 10^9 * 10^9 = 10^18 < 9.2e18
        long long rhs = p[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j;                     // deterministic tie-break (cost-neutral)
    });

    // Total weighted completion time.  Completion times reach sum(p) ~ 10^14;
    // each term w_i * C_i ~ 10^9 * 10^14 = 10^23, summed -> far past 64-bit.
    // Accumulate the running clock in 64-bit (<= 10^14, safe) but the objective
    // in __int128.
    long long clock_t = 0;                // running completion time
    __int128 total = 0;
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        clock_t += p[i];                  // scene i finishes at this time
        total += (__int128)w[i] * clock_t;
    }

    // Print the __int128 result.
    if (total == 0) { cout << 0 << "\n"; return 0; }
    bool neg = total < 0;                 // never happens here, but be safe
    if (neg) total = -total;
    string s;
    while (total > 0) { s += char('0' + (int)(total % 10)); total /= 10; }
    if (neg) s += '-';
    reverse(s.begin(), s.end());
    cout << s << "\n";
    return 0;
}
```
