**Problem.** Given a lowercase string `s` of length `n` (`n <= 10^6`), call a prefix `s[0..L-1]`
*tiled* if it is some strictly shorter block repeated **two or more** whole times: a tile length `d`
with `1 <= d < L`, `d | L`, and `s[i] == s[i-d]` for all `d <= i < L`. Print two integers: how many
prefixes are tiled, and the sum over tiled prefixes of their minimal tile length `d`. (`abc`, `a`,
and `abcab` are not tiled; `abcabc` is, with `d = 3`.)

**Key idea — one KMP failure-function pass.** Build the failure function `pi` indexed by prefix
*length* `L` (`pi[L]` = length of the longest proper border of `s[0..L-1]`). By the periodicity
lemma the **shortest period** of the length-`L` prefix is `d = L - pi[L]`, and that period tiles the
prefix into whole copies iff `d | L`. So a prefix is tiled iff

  `d = L - pi[L]`,  `d < L`,  and  `L % d == 0`,

with minimal tile length exactly `d`. Build `pi` in `O(n)`, then one linear scan answers everything.

**Pitfalls.**
1. *Inclusive vs. exclusive boundary (`d < L`, not `d <= L`).* An aperiodic prefix has `pi[L] = 0`,
   so `d = L` — the "period" is the whole string, i.e. one copy. The test must be the strict
   `d < L` to demand two or more copies. Writing `d <= L` certifies *every* prefix as tiled: a trace
   of `abc` then returns `3 6` instead of the correct `0 0`. This is the off-by-one the whole problem
   turns on; it is also why `(ab)^{500000}` answers `499999`, not `500000` (the length-2 prefix `ab`
   has `d = 2`, and `2 < 2` is false).
2. *Failure-function build off-by-one.* Build over lengths `i = 2..n`, leaving `pi[0] = pi[1] = 0`.
   Starting the loop at `i = 1` compares `s[0]` against itself and records an improper border
   `pi[1] = 1`; then at `L = 1` the period is `d = 1 - 1 = 0` and `L % d` is a division by zero. A
   trace of `aa` exposes this immediately.
3. *Overflow.* The tile-length sum can reach `~n^2/4 ≈ 2.5*10^{11}`; accumulate it in `long long`.

**Edge cases.** `n = 1` -> `0 0` (the build loop is empty, `d = 1`, `1 < 1` false). Trailing breaker
`aaab` -> `2 2` (the `b` makes `pi[4] = 0`, `d = 4`, skipped). Border-without-tiling `abcab` -> `0 0`
and `aba`-style near-misses are killed by `L % d == 0`. Unit period `aaaa` -> `3 3`.

**Complexity.** `O(n)` time, `O(n)` space for `pi` (an `int` array). One pass to build, one to scan.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int n = (int)s.size();

    // KMP failure function: pi[i] = length of the longest proper prefix of
    // s[0..i-1] that is also a suffix of s[0..i-1]. pi has size n+1, indexed by
    // PREFIX LENGTH L in [0, n]; pi[0] = pi[1] = 0 by definition. We build it on
    // 1-indexed lengths so the "period" arithmetic L - pi[L] is exact.
    vector<int> pi(n + 1, 0);
    int k = 0;                       // current matched border length
    for (int i = 2; i <= n; i++) {   // i = current prefix LENGTH
        // character being added is s[i-1] (0-indexed); compare against s[k]
        while (k > 0 && s[i - 1] != s[k]) k = pi[k];
        if (s[i - 1] == s[k]) k++;
        pi[i] = k;
    }

    // A prefix of length L (1 <= L <= n) is "tiled" iff its shortest period
    // d = L - pi[L] satisfies d < L (at least two copies) AND L % d == 0
    // (the period tiles the whole length exactly).
    long long count = 0;
    long long sumTile = 0;           // sum of minimal tile length d over tiled L
    for (int L = 1; L <= n; L++) {
        int d = L - pi[L];
        if (d < L && L % d == 0) {
            count++;
            sumTile += d;
        }
    }

    cout << count << " " << sumTile << "\n";
    return 0;
}
```
