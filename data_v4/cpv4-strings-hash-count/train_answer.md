**Problem.** Given a lowercase string `s` of length `n` and an integer `k`, count how many
**distinct** length-`k` substrings of `s` occur at **two or more** different starting positions. A
substring is its content, not its location: a string that can be read at three indices still counts
once. If `k < 1` or `k > n` there are no windows and the answer is `0`. Read `n k` then `s` from
stdin; print the count.

**Why the obvious sort is too slow.** Materializing all `n - k + 1` windows as strings and sorting
them is correct but `O(n * k * log n)`; at `n = 2*10^5`, `k ~ n/2` that is ~`10^{11}` character
comparisons — over a second. Keep it only as a brute-force oracle.

**Key idea — rolling hash + group count.** Fingerprint each length-`k` window with a polynomial hash
`h = c_0 B^{k-1} + c_1 B^{k-2} + ... + c_{k-1} (mod M)`, rolled in `O(1)`:

`h' = (h - c_{out} * B^{k-1}) * B + c_{in}   (mod M).`

The departing character carries weight `B^{k-1}` (not `B^k`). Use **two** independent moduli/bases and
pack `(h1, h2)` into one 64-bit key `(h1 << 32) ^ h2` to make false collisions negligible. Sort the
`n - k + 1` keys and sweep maximal runs of equal keys; **add `1` for each run of size `>= 2`**. That
run count is exactly "distinct substrings that repeat."

**Pitfalls.**
1. *Counting occurrences, not distinct substrings.* A substring appearing `t >= 2` times must add `1`,
   not `t`. Marking every window that has an equal neighbour quadruples the sample `"ababbaba", k=3`
   (gives `4` instead of `2`). Sweep groups, count one per group of size `>= 2`.
2. *Wrong roll power.* Removing `c_out * B^k` instead of `c_out * B^{k-1}` desyncs equal windows: on
   `"aa", k=1` the two identical `"a"` windows get different fingerprints and the answer collapses to
   `0` instead of `1`. Precompute `B^{k-1}` (loop `k-1` times).
3. *Window count / boundaries.* There are `n - k + 1` windows only when `1 <= k <= n`; guard
   `k < 1 || k > n -> 0`. `k = n` gives one window -> `0`. Read `s` only when `n > 0`, else `n = 0`
   would block on the missing string token.
4. *Hash collisions.* A single `~10^9` modulus would false-merge two different substrings over `2*10^5`
   windows; double hashing (`~10^{18}` space) makes that negligible.

**Edge cases.** `k = 0` -> `0`; `k > n` -> `0`; `k = n` -> `0` (single window can't repeat); `n = 0`
-> `0`; `k = 1` -> single-char windows handled by `B^{k-1} = B^0 = 1`; tiny alphabet with huge groups
counts each distinct repeated string once.

**Complexity.** `O(n)` to roll, `O(n log n)` to sort `n - k + 1` 64-bit keys, `O(n)` to sweep;
`O(n)` memory. ~0.01 s at `n = 2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k;
    if (!(cin >> n >> k)) return 0;
    string s;
    if (n > 0) cin >> s;            // when n == 0 there is no string token

    // Number of length-k windows. If k is out of [1, n] there are none.
    if (k < 1 || k > n) { cout << 0 << "\n"; return 0; }
    long long W = n - k + 1;        // window count: starts 0..n-k inclusive

    // Two independent polynomial rolling hashes, packed into one 64-bit key.
    const unsigned long long M1 = 1000000007ULL, M2 = 998244353ULL;
    const unsigned long long B1 = 131ULL, B2 = 137ULL;

    // Precompute B^(k-1) mod M: the weight of the leading char we remove when rolling.
    unsigned long long p1 = 1, p2 = 1;
    for (long long i = 0; i < k - 1; i++) { p1 = (p1 * B1) % M1; p2 = (p2 * B2) % M2; }

    vector<unsigned long long> keys;
    keys.reserve(W);

    unsigned long long h1 = 0, h2 = 0;
    // first window s[0..k-1]
    for (long long i = 0; i < k; i++) {
        unsigned long long c = (unsigned long long)(s[i] - 'a' + 1);
        h1 = (h1 * B1 + c) % M1;
        h2 = (h2 * B2 + c) % M2;
    }
    keys.push_back((h1 << 32) ^ h2);

    // roll: window starting at i uses removing s[i-1], appending s[i+k-1]
    for (long long i = 1; i < W; i++) {
        unsigned long long out = (unsigned long long)(s[i - 1] - 'a' + 1);
        unsigned long long in  = (unsigned long long)(s[i + k - 1] - 'a' + 1);
        // remove leading char (weight B^(k-1)), shift left by one, append new char
        h1 = (h1 + M1 - (out * p1) % M1) % M1;
        h1 = (h1 * B1 + in) % M1;
        h2 = (h2 + M2 - (out * p2) % M2) % M2;
        h2 = (h2 * B2 + in) % M2;
        keys.push_back((h1 << 32) ^ h2);
    }

    sort(keys.begin(), keys.end());

    // Count DISTINCT substrings whose group size is >= 2 (appears at >= 2 positions).
    long long ans = 0;
    long long i = 0, m = (long long)keys.size();
    while (i < m) {
        long long j = i;
        while (j < m && keys[j] == keys[i]) j++;
        if (j - i >= 2) ans++;     // one distinct substring, counted once
        i = j;
    }

    cout << ans << "\n";
    return 0;
}
```
