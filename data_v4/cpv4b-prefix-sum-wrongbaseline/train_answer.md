**Problem.** Given a signed ledger `a[0..n-1]` (`-10^9 <= a[i] <= 10^9`) and a modulus `m`
(`1 <= m <= 2*10^5`), count the contiguous windows `[l, r]` whose sum is divisible by `m`. Endpoints
identify a window, so equal-sum windows over different days are counted separately. Read `n`, `m`, and
the values from stdin; print the count.

**Key idea — prefix-sum remainders.** Let `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]`. The sum of
window `[l, r]` is `P[r+1] - P[l]`, which is divisible by `m` exactly when `P[r+1] ≡ P[l] (mod m)`.
So bucket the prefix sums by remainder: scanning left to right, for each new prefix add to the answer
the number of earlier prefixes already in its bucket, then record it. Seed the remainder-`0` bucket
with count `1` for the empty prefix `P[0] = 0`. This is `O(n)` time, `O(m)` space.

**Why the textbook baseline is wrong here.** The standard line buckets by `prefix % m` directly. That
is correct for non-negative arrays, but this ledger is signed, so prefix sums go negative — and C++'s
`%` truncates toward zero, returning a *negative* remainder (`-2 % 5 == -2`, not `3`). The identity is
about the mathematical remainder in `{0, ..., m-1}`, so two congruent prefixes get filed under
different keys and the divisible window between them is missed. Concretely, on `n=2, m=5, a=[-2, 5]`
the prefixes are `0, -2, 3`; the window `[5] = 3 - (-2)` is divisible (`-2 ≡ 3 mod 5`), but raw `%`
files `P[1]` under `-2` and `P[2]` under `3`, so the baseline prints `0` while the answer is `1` (a
fixed-size-array version is worse — `cnt[-2]` is out-of-bounds). The fix is one normalization:
`r = prefix % m; if (r < 0) r += m;`.

**Pitfalls to get right.**
1. *Negative remainders.* Normalize into `[0, m)` before bucketing. Adding `m` once suffices because
   `prefix % m` lies in `(-m, m)`. Then a flat `vector<long long> count(m)` is safe to index and fast.
2. *Overflow, twice.* The running prefix reaches `~2*10^14`, and at `m = 1` every window qualifies so
   the count is `n(n+1)/2 ≈ 2*10^10`. Both the accumulator and the answer must be `long long`; an `int`
   answer is a silent wrong-answer on the `m = 1` tests.

**Edge cases.** `n = 0` -> `0` (no prefix added). `m = 1` -> `n(n+1)/2` (all windows). A single
divisible day -> `1`; a single non-divisible day -> `0`. All-negative divisible runs (e.g.
`[-4,-4,-4]`, `m=4` -> `6`) are exactly what the unnormalized baseline crashes or miscounts on.

**Complexity.** `O(n + m)` time, `O(m)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m;
    if (!(cin >> n >> m)) return 0;

    // count[r] = number of prefix sums seen so far whose normalized remainder mod m is r.
    // prefix sum 0 (the empty prefix, before index 0) is present from the start.
    vector<long long> count(m, 0);
    count[0] = 1;

    long long prefix = 0;     // running prefix sum (64-bit: can reach ~2*10^14)
    long long answer = 0;     // number of windows (64-bit: can reach ~2*10^10)

    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        prefix += x;
        // Normalize the remainder into [0, m): C++ % can be negative for negative prefix.
        long long r = prefix % m;
        if (r < 0) r += m;
        answer += count[r];   // every earlier prefix with the same remainder closes a divisible window
        count[r]++;
    }

    cout << answer << "\n";
    return 0;
}
```
