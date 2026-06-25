**Problem.** Given `n` readings `a[0..n-1]` with `0 <= a[i] < 2^30`, count the contiguous windows `[l, r]` whose XOR `a[l] XOR ... XOR a[r]` has an **even number of set bits** (even popcount). A length-one window's XOR is the reading itself; the empty strip (`n = 0`) has `0` windows. Read `n` and the values from stdin, print the count.

**The tempting false identity — and why it is wrong.** It is natural to claim "even popcount of `x`" equals "`x` is an even number" (lowest bit `0`) and then bucket prefix-XOR values by their low bit. This is **false**: `x = 3 = (11)_2` has popcount `2` (even, so balanced) but is an odd number; `x = 5 = (101)_2` has popcount `2` (even) and is odd too. The two predicates disagree whenever the set-bit count is even but the lowest bit is `1`. Bucketing by value parity would systematically miscount those windows. Disprove this on `x = 3` before building anything on it.

**Key idea — popcount parity is linear over XOR.** The correct, provable identity is

```
par(x XOR y) = par(x) XOR par(y),    where par(z) = popcount(z) mod 2.
```

Reason per bit column: a column contributes to `popcount(x) + popcount(y)` the values `0/1/1/2` for input bits `00/10/01/11`, and to `popcount(x XOR y)` the values `0/1/1/0`. They differ only in the `11` column, by `2` — an even amount — so summed over all columns `popcount(x XOR y)` and `popcount(x) + popcount(y)` share parity. (Check: `x=3,y=5 -> x^y=6`, pc `2`, par `0 = 0^0`; `x=1,y=3 -> x^y=2`, pc `1`, par `1 = 1^0`; `x=7,y=1 -> x^y=6`, par `0 = 1^1`. All agree.)

With prefix XOR `P[0]=0`, `P[k]=a[0] XOR ... XOR a[k-1]`, window `[l,r]` has XOR `P[r+1] XOR P[l]`, so it is balanced iff `par(P[l]) == par(P[r+1])`. Label each prefix by `q[k] = par(P[k])`, which is just the running XOR of the per-element popcount parities (`q[0] = 0`). A window is balanced exactly when its two endpoint prefixes share `q`. With `c0`, `c1` the class sizes over all `n + 1` prefixes,

```
answer = C(c0, 2) + C(c1, 2).
```

**Pitfalls.**
1. *The bit identity.* Do not equate "even popcount" with "even value" (`x & 1 == 0`); it is false (witness `x = 3`). Fold `__builtin_popcount(x) & 1`, not `x & 1`. A trace of single reading `[3]` returning `0` instead of `1` exposes this slip.
2. *The empty prefix.* Initialize the class of the empty prefix `q[0] = 0`, i.e. start `cnt0 = 1`. Omitting it drops every window that starts at index `0`.
3. *Overflow.* The answer can reach `C(2*10^5 + 1, 2) ≈ 2*10^10`, beyond 32-bit range; accumulate in `long long`. The all-equal-even strip hits this extreme (`= n(n+1)/2`).

**Edge cases.** `n = 0` -> `0` (loop skipped, `cnt0 = 1`, `C(1,2) = 0`); single even-popcount reading -> `1`; single odd-popcount reading -> `0`; value `0` (popcount `0`) never flips a class; wide values near `2^30 - 1` read as `unsigned int` and pass through `__builtin_popcount` unchanged.

**Complexity.** `O(n)` time, `O(1)` extra space — a single pass folding one parity bit and incrementing one of two counters.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;            // empty input -> n = 0 -> answer 0

    // p = running parity of popcount over the prefix; p == 0 for the empty prefix.
    // A window (l, r] has XOR with even popcount  <=>  p[l] == p[r]
    // (popcount parity is linear over XOR). Count prefixes in each parity class
    // (including the empty prefix) and pair equal-parity prefixes.
    long long cnt0 = 1, cnt1 = 0;          // empty prefix has parity 0
    int p = 0;
    for (int i = 0; i < n; i++) {
        unsigned int x;
        cin >> x;
        p ^= (__builtin_popcount(x) & 1);  // fold this element's popcount parity in
        if (p == 0) cnt0++; else cnt1++;
    }

    long long ans = cnt0 * (cnt0 - 1) / 2 + cnt1 * (cnt1 - 1) / 2;
    cout << ans << "\n";
    return 0;
}
```
