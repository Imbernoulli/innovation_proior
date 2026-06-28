**Problem.** Given two lowercase strings `s` and `t` (`1 <= |s|, |t| <= 3000`), compute the length of
their longest common subsequence (LCS) — the longest string obtainable from each by deleting
characters while preserving order. Read `s` on line 1 and `t` on line 2 from stdin; print the length.

**Why the obvious greedy is wrong.** The tempting "match-as-you-scan" greedy keeps a pointer into `t`
and, for each character of `s`, matches it against the earliest remaining equal character of `t`. It
fails because subsequence matching is a *global* alignment problem and the earliest match optimizes
the wrong objective. On `s = "abb"`, `t = "bba"`, greedy matches `s`'s leading `a` against `t`'s only
`a` (at the end), consumes all of `t`, and strands both `b`s, returning `1` — but the true LCS is
`bb`, length `2`. Committing to an early match forecloses a longer downstream alignment. Greedy is
discarded.

**Key idea — prefix dynamic programming.** Let `dp[i][j]` be the LCS length of `s[0..i-1]` and
`t[0..j-1]`, with base cases `dp[0][j] = dp[i][0] = 0`. The recurrence, by case analysis on the last
characters:

- If `s[i-1] == t[j-1]`: pair them. `dp[i][j] = dp[i-1][j-1] + 1` (diagonal `+1`). An exchange
  argument shows an optimal LCS may always use this matched pair, so the diagonal never loses.
- If `s[i-1] != t[j-1]`: at least one of the two final characters is unused, so drop one side and
  take the better: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.

The answer is `dp[n][m]`. Unlike greedy, every cell weighs *both* ways of dropping a character, so no
early commitment can strand a longer alignment.

**Memory layout.** Row `i` reads only row `i-1` (cells `prev[j]`, `prev[j-1]`) and the current row's
left neighbour (`cur[j-1]`). So a full `(n+1) x (m+1)` table (~36 MB at `3000 x 3000`) is unnecessary;
two rolling rows of length `m+1` (~24 KB) suffice, swapped each iteration. Column 0 stays `0`
throughout (never written by the inner loop), and `cur[0] = 0;` is set explicitly each row to keep
that invariant robust across the swap.

**Pitfalls to get right.**
1. *Index offsets.* `dp` indices run `1..n`/`1..m` while string indices run `0..n-1`/`0..m-1`; the
   recurrence reads `s[i-1]`, `t[j-1]`. Off-by-one here is the classic LCS bug.
2. *Match vs mismatch terms.* The match branch must use the *diagonal* `prev[j-1] + 1` (not `prev[j]`);
   the mismatch branch must use `max(prev[j], cur[j-1])` (drop from `s` vs drop from `t`).

**Edge cases.** Identical strings -> `|s|`; disjoint alphabets -> `0`; length-1 strings -> `0` or `1`;
the reordering trap `"abb"`/`"bba"` -> `2` (where greedy gives `1`). The answer is at most `3000`, so
a plain `int` never overflows.

**Complexity.** `O(n*m)` time (`~10^7` steps at the limit, well under 1 s) and `O(m)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s, t;
    if (!(cin >> s)) return 0;             // empty input -> LCS length 0
    if (!(cin >> t)) { cout << 0 << "\n"; return 0; }

    int n = (int)s.size(), m = (int)t.size();

    // dp[j] = LCS length of s[0..i-1] and t[0..j-1], rolled over rows of s.
    // Two rolling rows keep memory at O(m) while the recurrence stays O(n*m).
    vector<int> prev(m + 1, 0), cur(m + 1, 0);
    for (int i = 1; i <= n; i++) {
        cur[0] = 0;
        char si = s[i - 1];
        for (int j = 1; j <= m; j++) {
            if (si == t[j - 1])
                cur[j] = prev[j - 1] + 1;          // extend the diagonal match
            else
                cur[j] = max(prev[j], cur[j - 1]);  // drop one char from s or from t
        }
        swap(prev, cur);
    }

    cout << prev[m] << "\n";
    return 0;
}
```
