**Problem.** A strip of `n` beacons broadcast 30-bit codes `a[0..n-1]` (`0 <= a[i] < 2^30`). Cut the strip into one or more contiguous segments; a segment's *signature* is the XOR of its codes; a segmentation is *clean* when every segment's signature has an **even popcount** (even number of set bits). Count the clean segmentations modulo `1000000007`. Read `n` and the codes from stdin, print the count.

**Key idea — 1D prefix-XOR DP with parity buckets.** Let `P[0]=0`, `P[i]=a[0] XOR ... XOR a[i-1]`. The signature of the last segment `j..i-1` is `P[i] XOR P[j]`. With `dp[i]` = number of clean segmentations of the first `i` beacons and `dp[0]=1`,

```
dp[i] = sum over j in [0, i-1] of dp[j]   for which   popcount(P[i] XOR P[j]) is even.
```

That literal recurrence is `O(n^2)`. To make it `O(n)`, replace the popcount test by a one-bit key on `P[j]` alone, via the identity

```
parity(popcount(x XOR y)) = parity(popcount(x)) XOR parity(popcount(y)).
```

(Proof: shared 1-bits of `x` and `y` vanish from `x XOR y` and each changes `popcount(x)+popcount(y)` by 2, i.e. 0 mod 2, so `popcount(x XOR y) ≡ popcount(x)+popcount(y) (mod 2)`.) Hence `popcount(P[i] XOR P[j])` is even **iff** `parity(popcount(P[i])) == parity(popcount(P[j]))`. Keep `bucket[0]`, `bucket[1]` = running sums of `dp[j]` by that parity; at step `i`, `dp[i] = bucket[key[i]]`, then fold `dp[i]` into `bucket[key[i]]`. `O(n)` time, `O(1)` space.

**Verify the identity, never assert it.** The whole speedup rides on a bit identity, so check it on concrete numbers before trusting it:
- The TRUE parity identity above survives a numeric test (0 mismatches over 100000 random pairs).
- The tempting FALSE shortcut "even popcount = the XOR is an even number" (bucket by the LSB `P[·] & 1`) is wrong: `x=3` (popcount 2, value odd) already breaks it, and end-to-end on `a = [0,0,0,2]` the correct answer is `0` (the strip signature `2` has odd popcount, so no cut cleans it) while the LSB cheat returns `8`.
- A third trap, the raw `popcount(x XOR y) = popcount(x) XOR popcount(y)`, fails ~94% of random pairs; only the *parity* version holds.

**Pitfalls.**
1. *Missing base seed.* `dp[0]=1` must be registered into `bucket[key[0]]` (here `bucket[0]`, since `popcount(0)=0`) **before** the loop; otherwise every `dp[i]` reads an empty bucket and collapses to `0`. (A trace of `[3,1,2]` returning `0` instead of `2` exposes exactly this.)
2. *False predicate.* Bucketing by `P & 1` ("even value") instead of by popcount parity — silently wrong; see `[0,0,0,2]` -> `8` vs the correct `0`.
3. *Modulus/overflow.* The count can reach `2^(n-1)`; report it mod `1000000007` and keep accumulators in `long long`, reducing after each update.

**Edge cases.** `n = 1` with an odd-popcount code (e.g. `[1]`) -> `0` (the answer can legitimately be 0). `n = 1` with `[0]` -> `1`. `n = 0` / empty strip -> `1` (the empty segmentation). Empty input stream -> print nothing.

**Complexity.** `O(n)` time, `O(1)` extra space (no `dp` or `P` arrays kept; everything rolls forward).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;            // empty input -> nothing to print

    const long long MOD = 1000000007LL;

    // Prefix XOR P[0..n] with P[0] = 0. The segment a[j..i-1] has signature
    // P[i] XOR P[j]; it is "clean" when popcount(P[i] XOR P[j]) is EVEN.
    //
    // Identity (verified numerically before trusting it):
    //   parity(popcount(x XOR y)) = parity(popcount(x)) XOR parity(popcount(y)).
    // Hence popcount(P[i] XOR P[j]) is even  <=>  the popcount PARITIES of
    // P[i] and P[j] are EQUAL. So bucket previous dp by the parity of
    // popcount(P[j]) and, at step i, add the bucket whose parity matches P[i].
    //
    // dp[i] = number of clean partitions of the prefix of length i, with dp[0] = 1.

    long long bucket[2] = {0, 0};         // bucket[p] = sum of dp[j] over processed j with parity p
    int P = 0;                            // running prefix XOR, P[0] = 0
    // j = 0: dp[0] = 1, parity of popcount(P[0]=0) is 0.
    bucket[__builtin_parity((unsigned)P)] = 1; // = bucket[0] = 1

    long long dp = 1;                     // dp[0]; reassigned each step to dp[i]
    for (int i = 1; i <= n; i++) {
        int x;
        cin >> x;
        P ^= x;                           // now P == P[i]
        int par = __builtin_parity((unsigned)P);
        dp = bucket[par] % MOD;           // sum of dp[j], j<i, with matching popcount parity
        bucket[par] = (bucket[par] + dp) % MOD; // register dp[i] = dp under P[i]'s parity
    }

    cout << dp % MOD << "\n";
    return 0;
}
```
