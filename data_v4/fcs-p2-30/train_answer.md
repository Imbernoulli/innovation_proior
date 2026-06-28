**Problem.** A letter string over `A..Z` was encoded as digits by `A->1, ..., Z->26`, separators were dropped, and we are given the digit run `s` (`1 <= |s| <= 10^5`). Count how many letter strings could have produced `s` — equivalently, how many ways `s` splits into consecutive groups each valued in `1..26` (one digit `1..9`, two digits `10..26`) — modulo a prime `p` (`2 <= p <= 2^31 - 1`). Read `p` then `s` from stdin; print the count mod `p`.

**Why the tempting closed form is wrong.** It is seductive to factor `s` into maximal "fully flexible" blocks (every digit nonzero, every adjacent pair in `10..26`), assign each block of length `m` the Fibonacci count `F(m+1)` via fast matrix power, and multiply. It is even correct on uniform blocks: `"1212"` really does give `F(5) = 5`. But the "cut or not" choices are **not independent** once values can exceed `26`. On `s = "127"` the block model sees three nonzero digits and assigns `F(4) = 3`, yet the true count is `2` (`1|2|7 -> "ABG"`, `12|7 -> "LG"`; the `1|27` split is illegal because `27 > 26`). Zeros couple positions further: a `0` is legal only as the tail of `10` or `20`, so a stray `0` collapses the count to `0` — another fact the block product cannot express. Salvaging the matrix idea would require a per-position transfer matrix that depends on each specific pair, which is just the linear DP in disguise and batches across nothing. The closed form is discarded.

**Key idea — linear segmentation DP.** Let `dp[i]` be the number of valid decodings of the prefix of length `i`. The last group is either one digit or two:

- `dp[i] += dp[i-1]` when `s[i-1]` is `'1'..'9'` (one-digit group legal).
- `dp[i] += dp[i-2]` when `s[i-2]s[i-1]` has value in `10..26` (two-digit group legal).

with `dp[0] = 1` (the empty prefix has exactly one decoding) and the answer `dp[n] mod p`. Zeros are handled for free: a `0` never extends through the one-digit branch, so unless it is the second digit of `10`/`20` the count drops to `0` and propagates. Keep only a rolling window of the last two values for `O(1)` memory.

**Two pitfalls to get right.**
1. *The `dp[-1]` sentinel.* Initialize the second-back value `prev2 = 0`, not `1`. Setting it to `1` asserts a phantom "decoding of the prefix before the start," which only stays harmless because the `i >= 2` guard never reads it before the rolling slide overwrites it — a correct output resting on a wrong invariant, one refactor from breaking. The honest value is `0`: a two-digit group at the very first position has nothing to extend. (Tracing `"01"` and `"10"` exposes the fragility.)
2. *Modulus and width.* With `p` near `2^31`, a partial residue is up to `~2.1*10^9` and the sum of the two branches reaches `~4.3*10^9`, both overflowing 32-bit. Use `long long` and reduce mod `p` every step. An `int` is a silent wrong answer on large-`p` tests.

**Edge cases (all handled by the recurrence + base case):** leading or stray `0` -> `0`; boundary `26` legal, `27` illegal (inclusive `10..26` test); `p = 2` (explicit `1 % p` and per-step reduction); `|s| = 10^5` in one pass with no overflow.

**Complexity.** `O(n)` time, `O(1)` extra space; the maximum `10^5` case runs in a few milliseconds.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long p;
    string s;
    if (!(cin >> p >> s)) return 0;        // missing input -> nothing to do
    int n = (int)s.size();

    // dp[i] = number of decodings of the prefix s[0..i-1], reduced mod p.
    // dp[0] = 1: the empty prefix has exactly one decoding (the empty string).
    // dp[i] = (s[i-1] is '1'..'9'        ? dp[i-1] : 0)
    //       + (s[i-2]s[i-1] in 10..26    ? dp[i-2] : 0)
    // We keep only the two most recent values in a rolling window.
    // prev1 = dp[i-1], prev2 = dp[i-2]. Before the loop (i==1): prev1 = dp[0] = 1,
    // prev2 = dp[-1] = 0 (a phantom; the i>=2 guard makes sure it is never used as dp[-1]).
    long long prev1 = 1 % p;               // dp[0]
    long long prev2 = 0;                   // dp[-1], unused while i < 2

    for (int i = 1; i <= n; i++) {
        char c1 = s[i - 1];                // i-th character (1-indexed): the one-digit group
        long long cur = 0;

        // One-digit group: s[i-1] alone decodes iff it is '1'..'9' (a leading '0' is invalid).
        if (c1 != '0') {
            cur += prev1;                  // extend each dp[i-1] decoding by this single digit
        }

        // Two-digit group: s[i-2]s[i-1] must form a value in 10..26.
        if (i >= 2) {
            int two = (s[i - 2] - '0') * 10 + (c1 - '0');
            if (two >= 10 && two <= 26) {
                cur += prev2;              // extend each dp[i-2] decoding by this two-digit group
            }
        }

        cur %= p;
        prev2 = prev1;                     // slide window: dp[i-2] <- dp[i-1]
        prev1 = cur;                       // dp[i-1] <- dp[i]
    }

    cout << (prev1 % p) << "\n";           // prev1 holds dp[n]
    return 0;
}
```
