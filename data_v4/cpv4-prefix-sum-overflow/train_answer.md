**Problem.** A station records `n` signed hourly net-flow readings `a[1..n]` (`-10^9 <= a[i] <= 10^9`).
Given `q` queries, each a window `[l, r]` whose answer is `a[l] + ... + a[r]`, output the **total
audited volume**: the sum of all `q` query answers as one integer. Bounds: `n <= 10^5`, `q <= 5*10^4`.

**Why the naive method is too slow.** Re-summing each window is `O(r-l+1)` per query, hence `O(n*q)`
worst case `= 10^5 * 5*10^4 = 5*10^9` additions — tens of seconds, past the 1-second limit. It is a
correct oracle, not a submission.

**Key idea — prefix sums.** Define `prefix[0] = 0` and `prefix[i] = a[1] + ... + a[i]`. By telescoping,
any window sum is one subtraction:

- `a[l] + ... + a[r] = prefix[r] - prefix[l-1]`  (`O(1)` per query after `O(n)` precomputation).

Accumulate these over all `q` queries. Total work `O(n + q)`.

**Correctness.** `prefix[r] - prefix[l-1] = (a[1]+...+a[r]) - (a[1]+...+a[l-1])`; the shared prefix
`a[1..l-1]` cancels, leaving exactly `a[l]+...+a[r]`. The `l = 1` boundary uses `prefix[0] = 0`, so the
window keeps its first element with no special case. Summing exact per-query answers gives the exact
total.

**Pitfalls.**
1. *Overflow (the main trap).* A single window sum can reach `n * max|a| = 10^5 * 10^9 = 10^{14}`,
   and the grand total over `q` windows can reach `q * 10^{14} = 5*10^{18}`, of either sign. The
   window sum alone overflows a 32-bit `int` (max `~2.1*10^9`); the total overshoots it by `~4*10^9`x.
   Use `long long` for the prefix array **and** the accumulator. (Concretely: with `int`, the sample's
   `prefix[3] = 3*10^9` wraps to `-1294967296`, and the program prints `1410065398` instead of the
   correct `9999999990`.) `long long` holds `~9.2*10^{18} > 5*10^{18}`; note the stated `q <= 5*10^4`
   bound is what keeps the total inside `long long` — a larger `q` would need `__int128`.
2. *Off-by-one at the left boundary.* The earlier endpoint is `prefix[l-1]`, not `prefix[l]`; with the
   `prefix[0] = 0` sentinel and 1-indexed readings, `l = 1` works without a special case. Writing
   `prefix[l]` would silently drop `a[l]` from every window.

**Edge cases.** `n = q = 1` with a single negative reading returns that reading. The all-`+10^9` /
all-`-10^9` whole-array windows repeated `5*10^4` times reach exactly `+-5*10^{18}`, both inside
`long long`. Zeros and mixed signs need no special handling — prefix arithmetic is sign-agnostic.

**Complexity.** `O(n + q)` time, `O(n)` extra space for the prefix array.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    // prefix[i] = a[1] + a[2] + ... + a[i], prefix[0] = 0.
    // Values reach n * max|a| = 1e5 * 1e9 = 1e14, far beyond 32-bit range,
    // so prefix sums MUST be 64-bit.
    vector<long long> prefix(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        prefix[i] = prefix[i - 1] + x;
    }

    // Sum over each query window [l, r] is prefix[r] - prefix[l-1].
    // The grand total over up to 5e4 windows of magnitude up to 1e14
    // reaches ~5e18, which still fits in long long but overflows int many times over.
    long long total = 0;
    for (int k = 0; k < q; k++) {
        int l, r;
        cin >> l >> r;
        total += prefix[r] - prefix[l - 1];
    }

    cout << total << "\n";
    return 0;
}
```
