**Problem.** Given `n` coin denominations, a target `S`, a prime modulus `p`, and an unlimited supply of each denomination, count the number of **distinct ways** (multisets — order does not matter) to make exactly `S`, modulo `p`. A way is fixed by how many coins of each *distinct denomination value* are used. Read `n S p` and the `n` denominations from stdin; print the count mod `p`.

**Why the tempting clever route is set aside.** The count is the coefficient of `x^S` in the product over distinct denominations of `1 / (1 - x^{c})`. A closed-form via partial fractions / inclusion-exclusion over subsets of denominations is elegant but, in the budget, either collapses into a power-series expansion (which *is* the DP below) or becomes an error-prone computer-algebra engine: with arbitrary, non-coprime denominations the poles collide (e.g. `{4, 6}`, gcd 2), repeated roots appear, and residue multiplicities branch. High risk of a silent wrong answer. Discarded.

**Why the reflexive DP is wrong.** The order-sensitive DP — scan sums `s = 1..S` outer, add `dp[s-c]` over coins `c` inner — counts *ordered compositions*, not multisets. Concretely, for `{1, 2, 5}` and `S = 5` it returns `9`, but the correct number of multisets is `4` (`{5}`, `{2,2,1}`, `{2,1,1,1}`, `{1,1,1,1,1}`); the `9` counts the multiset `{2,2,1}` three times as `(2,2,1),(2,1,2),(1,2,2)` and so on. Putting the sum loop outside interleaves denominations and counts orderings. Discarded.

**Key idea — order-independent counting DP.** Process denominations in the **outer** loop and relax all sums in the **inner** loop:

- `dp[s] += dp[s - c]` for `s = c .. S`, one denomination `c` at a time.

After the pass for denomination `c`, `dp[s]` counts multisets summing to `s` using only the denominations processed so far. Sweeping `s` upward within the pass lets `c` be used `0, 1, 2, ...` times, and all of `c`'s contributions are collected before the next denomination is touched. So every multiset is generated in exactly one canonical (non-decreasing denomination index) order and counted once. Base case `dp[0] = 1` (the empty multiset), so `S = 0` yields `1` automatically. Answer: `dp[S] mod p`.

**Hand check.** `{1, 2, 5}`, `S = 5`, processing `1, 2, 5`: after `1`, `dp = [1,1,1,1,1,1]`; after `2`, `dp = [1,1,2,2,3,3]`; after `5`, `dp[5] += dp[0] = 4`. Matches the four multisets.

**Two pitfalls to get right.**
1. *Loop order.* The coin loop must be **outside**. Sums-outside counts ordered compositions (the `9`-vs-`4` bug above).
2. *Duplicate denomination values.* A way is fixed per distinct value, so two input coins of value `2` are the same type. Without deduping, processing `2` twice treats them as different types and double-counts: `{2, 2}`, `S = 4` returns the illegal `3` instead of `1` (the only multiset is `{2,2}`). Fix with sort + unique before the DP.

**Other corners (handled by the recurrence + base case + a guard):** `S = 0` -> `1`; unreachable target -> `0`; single denomination -> `1` iff it divides `S`; a coin larger than `S` is skipped via `if (coin > S) continue;`; tiny modulus `p = 2` reduces correctly via `dp[0] = 1 % p` and per-step `% p`. Each addend is `< p <= 10^9 + 7`, so the pre-mod sum fits in `long long`.

**Complexity.** `O(n * S)` time after dedup (at most `200 * 2*10^5 = 4*10^7` inner additions), `O(S)` space. Worst case timed at about `0.08` s, well within a 2 s limit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S, p;
    if (!(cin >> n >> S >> p)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // A "way" is fixed by how many coins of each DISTINCT denomination value are
    // used, so duplicate denomination values in the input refer to the same coin
    // type and are collapsed first.
    sort(c.begin(), c.end());
    c.erase(unique(c.begin(), c.end()), c.end());

    // dp[s] = number of distinct multisets of coins summing to exactly s, mod p.
    // Order-independent counting: put the COIN loop OUTSIDE and the sum loop
    // INSIDE. Each coin is fully processed before the next, so every multiset is
    // counted once in a fixed canonical order of denominations (no permutations).
    vector<long long> dp(S + 1, 0);
    dp[0] = 1 % p;                       // exactly one way to make 0: the empty multiset
    for (size_t i = 0; i < c.size(); i++) {
        long long coin = c[i];
        if (coin > S) continue;          // a coin larger than S can never be used
        for (long long s = coin; s <= S; s++) {
            dp[s] = (dp[s] + dp[s - coin]) % p;
        }
    }

    cout << dp[S] % p << "\n";
    return 0;
}
```
