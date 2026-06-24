**Problem.** Given `n` non-negative integers `a[0..n-1]` (`n <= 10^5`, `0 <= a[i] <= 10^4`), compute the total handshake value `S = sum_{i<j} a[i] * a[j]`, the sum of the product over every unordered pair. Read `n` and the values from stdin, print `S`. If `n < 2` there are no pairs, so `S = 0`.

**Why the obvious double loop is wrong (here).** Summing `a[i] * a[j]` over all pairs is correct by definition but `O(n^2)`; at `n = 10^5` that is `~5*10^9` multiply-adds, far over a 1-second limit. It is only useful as a small-case reference.

**Key idea — closed form via square of the sum.** Let `T = sum_i a[i]` and `Q = sum_i a[i]^2`. Expanding the square over all ordered pairs,

`T^2 = (sum_i a[i])(sum_j a[j]) = sum_{i,j} a[i] a[j] = Q + sum_{i != j} a[i] a[j] = Q + 2S`,

since the off-diagonal counts each unordered pair `{i,j}` twice. Therefore

`S = (T^2 - Q) / 2`.

One `O(n)` pass accumulates `T` and `Q`; the finish is constant time. Check on `[3,1,4,1,5]`: `T = 14`, `T^2 = 196`, `Q = 52`, `S = (196 - 52)/2 = 72`.

**Pitfalls to get right.**
1. *Overflow — the headline trap.* With `n = 10^5` and `a[i] = 10^4`, `T` reaches `10^9`, the intermediate `T^2` reaches `10^18`, and the answer `S` reaches `~5*10^17`. All exceed a 32-bit `int` (`~2.1*10^9`) by many orders of magnitude. Crucially, making only the *result* `long long` is not enough: if `sum` and the per-element `x` are `int`, then `sum * sum` and `x * x` are computed in 32-bit and wrap *before* being widened. Every operand on the arithmetic path — `x`, `sum`, `sumsq` — must be `long long`. (`int` here is a silent wrong answer on the big tests.)
2. *The factor of 2.* `T^2 - Q` is the sum over ordered off-diagonal pairs and double-counts every handshake; you must divide by 2. Forgetting it returns exactly `2S` (e.g. `144` instead of `72` on the sample). The division is exact because `T^2 - Q = 2S` is always even.

**Edge cases (all handled by the formula, no special-casing):** `n = 0` -> `(0-0)/2 = 0`; `n = 1` -> `(a^2 - a^2)/2 = 0`; all zeros -> `0`; equal values `[5,5,5]` -> `(225-75)/2 = 75`.

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    long long sum = 0, sumsq = 0;
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        sum += x;
        sumsq += x * x;            // x up to 1e4 => x*x up to 1e8, fits; long long anyway
    }
    // sum over i<j of a[i]*a[j] = (sum^2 - sum of squares) / 2
    long long answer = (sum * sum - sumsq) / 2;
    cout << answer << "\n";
    return 0;
}
```
