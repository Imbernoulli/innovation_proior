**Problem.** Given a lowercase string `s` of length `n` (`1 <= n <= 2*10^5`), for each prefix length `k = 1..n` let `occ(k)` be the number of start positions where the prefix `s[0..k-1]` occurs as a substring of `s` (overlaps count; distinct starts are distinct). Output the self-similarity score `score = occ(1) + ... + occ(n)`. Read `s` from stdin, print one integer.

**Why the obvious approach is too slow.** Searching `s` for each of the `n` prefixes separately is `O(n^2)`, about `4*10^10` operations at `n = 2*10^5` — hopeless under a 1-second limit. It is only useful as a brute-force oracle on tiny inputs.

**Key idea — one KMP prefix-function pass.** The patterns are the string's own prefixes, so the prefix function `pi` (where `pi[i]` = length of the longest proper border of `s[0..i]`) already encodes, for every position `i`, the longest prefix ending there; the shorter prefixes ending at `i` are its border chain. Aggregate occurrence counts down those chains:

1. For each position `i`, the prefix of length `pi[i]` ends at `i` as a border: `occ[pi[i]]++`.
2. Push counts from longer borders to shorter ones, longest length first: for `L` from `n` down to `1`, `occ[pi[L-1]] += occ[L]`.
3. Every prefix of length `L` also occurs once at the start of `s`: `occ[L]++` for `L = 1..n`.

Then `occ(L) = occ[L]` and `score = sum_{L=1}^{n} occ[L]`. (Bucket `occ[0]` is the empty-prefix chain sink and is never summed.) Total time `O(n)`.

**Pitfalls.**
1. *Integer overflow (the main trap).* The most self-similar string `"aaaa...a"` has `score = n(n+1)/2`, which at `n = 2*10^5` is `20000100000 ≈ 2*10^10` — almost ten times past the 32-bit signed cap `2147483647`. An `int` accumulator wraps to the impossible negative `-1474736480`. Worse, the intermediate buckets themselves grow to `~2*10^10` (mass funnels into `occ[0]`), so the *count array* must be `long long`, not only the final accumulator. Small random tests (`n <= 14`, score `<= 105`) never reach this regime, so you must reason about the large case explicitly. Use `long long` for `occ[]` and `score`.
2. *Off-by-one in the border length.* Step 1 indexes by `pi[i]` (the border *length*), not `pi[i]+1`. Getting this wrong shifts every count by one length.
3. *The "own occurrence" `+1`.* Each prefix occurs at least once at the start of `s`; that occurrence is not a border of anything, so it must be added separately in step 3.

**Edge cases.** Empty input -> `0`. `n = 1` -> `1` (the single prefix occurs once). Unique-alphabet string like `"abcd"` -> `n` (every prefix occurs exactly once). The loop bound `L = n..1` is safe: `occ[n] = 0` after step 1 (no `pi[i]` can equal `n`), so the `L = n` iteration is a no-op.

**Complexity.** `O(n)` time, `O(n)` memory. Runs in ~7 ms at `n = 2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    string s;
    if (!(cin >> s)) { cout << 0 << "\n"; return 0; } // empty input -> score 0
    int n = (int)s.size();

    // KMP prefix function: pi[i] = length of the longest proper border of s[0..i].
    vector<int> pi(n, 0);
    for (int i = 1; i < n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]) j = pi[j - 1];
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }

    // occ[len] = number of times the length-`len` prefix occurs as a substring of s.
    // Every border of length `pi[i]` ending at i is one such occurrence; chase the
    // border chain by propagating counts from longer borders to shorter ones, then
    // add 1 to every prefix length for its "own" occurrence at the start.
    vector<long long> occ(n + 1, 0);
    for (int i = 0; i < n; i++) occ[pi[i]]++;
    for (int i = n; i >= 1; i--) occ[pi[i - 1]] += occ[i];
    for (int len = 1; len <= n; len++) occ[len]++;

    // Self-similarity score = total number of (prefix, occurrence) incidences.
    long long score = 0;
    for (int len = 1; len <= n; len++) score += occ[len];

    cout << score << "\n";
    return 0;
}
```
