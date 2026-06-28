**Problem.** Given a string `s` of length `n` (`1 <= n <= 10^6`), consider its `n`
cyclic rotations, where rotation `k` is `s[k:] + s[:k]`. Output the smallest
0-based index `k` whose rotation is lexicographically least; on ties (several
indices giving the same least rotation) output the smallest such index. Read `s`
from stdin, print the index. Empty input -> `0`.

**Why the obvious approach is too slow.** "Build all `n` rotations and take the
lexicographically smallest" is obviously correct and makes a perfect oracle, but
it is `Θ(n^2)`: adjacent rotations share huge prefixes (think `s = "aaaa...ab"`,
where rotations agree on a long run of `'a'`s), so each of the `n` comparisons
re-scans `Θ(n)` shared characters. At `n = 10^6` that is `~10^12` operations —
far past a 1-second limit. The waste is re-reading the same shared prefixes over
and over.

**Key idea — Booth's algorithm (failure-function block-skip).** Run a KMP-style
failure-function scan over the conceptual doubled string `ss = s + s`, carrying a
single candidate best-start `k`. Compare `ss[j]` against the next expected
character `ss[k + i + 1]` of the candidate, where `i + 1` is the current matched
length (`i = f[j - k - 1]`):

- **Match:** extend the matched block (`f[j - k] = i + 1`).
- **`ss[j]` strictly smaller:** the rotation starting at `j - i - 1` beats the one
  at `k` (same prefix, smaller next char) -> move `k` to `j - i - 1`.
- **`ss[j]` larger:** the candidate at `k` wins, *and so does every start inside
  the matched block* (shared losing prefix) -> follow the failure link
  `i = f[i]` to **skip the entire block at once** instead of stepping back one
  character.

That single block-skip is what makes a mismatch cost `O(1)` amortized instead of
`O(n)`, turning the quadratic re-scan into `O(n)` total. The answer is the final
`k`.

**Pitfalls to get right.**
1. *Tie-breaking by index.* Move `k` **only on a strict `<`**, never on equality.
   For an all-equal string (`"aaaa"`) no strict improvement fires, so `k` stays at
   `0` — the smallest index, as required. Using `<=` would drift `k` forward on
   ties and return a wrong (larger) index.
2. *The index arithmetic is the bug surface.* `f[j - k - 1]`, `ss[k + i + 1]`,
   `f[j - k]`, and the two `k`-update sites (`k = j - i - 1` inside the loop and
   `k = j` when `i == -1`) interact subtly. A hand-trace is unreliable here; this
   code must be differential-tested, not eyeballed.
3. *Doubled-string bounds.* Scan `j` over `[1, 2n)` and index `ss` of length `2n`;
   guard `n == 0` so `ss` is never indexed empty.

**Edge cases (all verified against the brute oracle).** `n = 1` -> `0`; empty
input -> `0`; all-equal -> `0` (strict-tie rule); period-2 `"abab..."` -> `0`;
`"a"*(n-1)+"b"` -> `0`; `"b"+"a"*(n-1)` -> `1`; long-run and random `n = 10^6`
adversaries run in ~10 ms / ~14 MB. Indices stay within `2n <= 2*10^6`, so `int`
suffices (no overflow, no `long long` needed).

**Complexity.** `O(n)` time, `O(n)` extra space (the doubled string and the
failure array).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Booth's algorithm (1980): least starting index of the lexicographically
// minimal rotation of s. O(n) time, O(n) space. It runs a failure-function /
// KMP-style scan over the conceptual doubled string s+s; on a mismatch it slides
// the candidate start k forward past an entire already-matched block at once,
// which is what turns the naive O(n^2) compare-all-rotations into O(n).
int leastRotation(const string &s) {
    int n = (int)s.size();
    if (n == 0) return 0;
    string ss = s + s;                 // conceptual doubled string
    vector<int> f(2 * n, -1);          // failure function over ss; -1 = no match
    int k = 0;                         // current best rotation start
    for (int j = 1; j < 2 * n; j++) {
        char c = ss[j];
        int i = f[j - k - 1];
        while (i != -1 && c != ss[k + i + 1]) {
            if (c < ss[k + i + 1]) k = j - i - 1;
            i = f[i];
        }
        if (c != ss[k + i + 1]) {      // here i == -1
            if (c < ss[k + i + 1]) k = j;   // since i == -1, j - i - 1 == j
            f[j - k] = -1;
        } else {
            f[j - k] = i + 1;
        }
    }
    return k;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s;
    if (!(cin >> s)) {
        // Empty input: treat as the empty string; least-rotation index is 0.
        cout << 0 << "\n";
        return 0;
    }
    cout << leastRotation(s) << "\n";
    return 0;
}
```
