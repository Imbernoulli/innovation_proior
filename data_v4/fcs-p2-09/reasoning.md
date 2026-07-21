Longest common subsequence, read from two lines of stdin — `s` first, then `t`, both lowercase, each up to `3000` characters. The value I print is at most `3000`, so a plain `int` has room to spare and no arithmetic gets near a 32-bit ceiling. What the constraint really fixes is the shape of the budget: the strings are individually small, but their product `|s| * |t|` reaches `3000 * 3000 = 9 * 10^6`, so any method costing one step per pair of positions runs on the order of ten million operations — nothing against a 1-second limit. The trap worth naming up front is that "longest common subsequence" is not "number of shared characters": deletions must preserve order, and that ordering constraint is the whole problem.

Two approaches are worth weighing, and I want to commit to the one I can prove, not the one that is shortest to type. The first is a greedy match-as-you-scan: keep a pointer `j` into `t`, walk `s` left to right, and for each `s[i]` advance `j` to the first equal character of `t`, counting a match when one exists. That is a few lines and it *feels* like it builds a common subsequence by always taking the earliest match. The second is prefix DP: `dp[i][j]` = LCS length of `s[0..i-1]` against `t[0..j-1]`, filled by a short recurrence in `O(n*m)`. The DP idea is textbook and provable; its only risks are index arithmetic and whether the table fits. Greedy is the one I distrust, because subsequence matching is a *global* alignment problem and an early commitment can foreclose a longer alignment later. Distrust is not evidence, so I try to break it.

My first attempts don't cooperate. On `s = ab`, `t = ba`, greedy matches `a` at the end of `t`, then finds no `b`, reporting `1` — but the true LCS is also `1`, so no gap. On `s = abc`, `t = acb`, greedy takes `a` then `b` for `2`, and the optimum is `2` as well. Greedy is stubbornly correct on tiny inputs because there is rarely enough string for an early match to strand much. The break needs a character sitting both early in `s` and late in `t`, so that matching it consumes the tail later characters of `s` needed. `s = abb`, `t = bba` does it: greedy matches the leading `a` against `t`'s only `a` at position 2, exhausting `t`, and both `b`s of `s` are stranded — greedy returns `1`. But `bb` is common to both, so the true LCS is `2`. That is the disproof: earliest-match optimizes "match this character as soon as possible" when the objective is total alignment length. Greedy is out.

The DP fixes exactly this by never committing. With base cases `dp[0][j] = dp[i][0] = 0`, look at the last characters of the two prefixes. If `s[i-1] == t[j-1]`, an optimal LCS may always pair them — any common subsequence not using both final characters can be exchanged to use them without shrinking — so `dp[i][j] = dp[i-1][j-1] + 1`, the diagonal. If they differ, no common subsequence can end by matching them, so at least one is unused; drop one side and take the better, `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`. The answer is `dp[n][m]`. The mismatch max weighs *both* ways of dropping a character, which is exactly the freedom greedy lacked.

On the instance greedy failed, `s = abb`, `t = bba`, the table reaches `dp[3][3] = 2`: the second `b` comes back on the diagonal (`dp[2][1] + 1`) once the recurrence is free to drop the leading `a` instead of forcing it to match — exactly the alignment greedy stranded.

For memory, a full `(n+1) x (m+1)` `int` table is about `9 * 10^6 * 4 ≈ 36` MB, comfortably inside 256 MB, so it would work. But row `i` reads only row `i-1` (cells `prev[j]` and `prev[j-1]`) and the current row's left neighbour `cur[j-1]`; it never reaches further back. Since I only need the length, not the subsequence, two rolling rows of length `m+1` suffice — memory drops to ~24 KB and time stays `O(n*m)`. The one subtlety with rolling rows is that `cur` is reused after `swap(prev, cur)`, so it is not freshly zeroed each iteration. Every `cur[j]` for `j >= 1` is overwritten before it is read, so those are fine; the only cell read without being written is `cur[0]` (as `cur[j-1]` when `j = 1`). But column 0 is never touched by the inner loop, which runs `j` from 1, and both arrays start all-zero, so `[*][0]` stays `0` forever and `cur[0]` reads the correct base `dp[i][0] = 0` on every row regardless of swaps. I still set `cur[0] = 0;` explicitly at the top of each row so the invariant is documented rather than resting on a swap coincidence.

To catch a coding slip the math wouldn't — a swapped index, the wrong branch — I differential-test the rolling-row DP against an independent memoized top-down recursion `rec(i, j)` branching match/skip: a different enough formulation from the bottom-up rows that a sign error is unlikely to be mirrored in both. Over the greedy-hostile regimes — tiny `{a,b}` and three-letter strings short enough for exhaustive truth, exact shuffles of one string, short-vs-long pairs, a known subsequence embedded in noise, plus fixed edges like `ab`/`ba` and the textbook `abcbdab`/`bdcaba` — the two always agree, and the corners land where they should: identical strings give `|s|`, disjoint alphabets `0`, and the reordering trap `abb`/`bba` gives `2`. A `3000 x 3000` tiny-alphabet case, the most crossing matches, runs in a few milliseconds against a full-table reference — the `~10^7`-step estimate holds, far under the 1-second limit.

So I ship the rolling-row DP — the approach I can defend against the greedy I broke — as one self-contained file:

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
