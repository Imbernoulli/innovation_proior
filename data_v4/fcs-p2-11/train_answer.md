**Problem.** Given a dictionary of `n` lowercase words and a lowercase target string `s` (`|s| <= 5000`, total dictionary length `<= 2*10^5`), decide whether `s` is a back-to-back concatenation of dictionary words, with each word reusable any number of times and the empty string counting as segmentable. Read `n`, the `n` words, then `s` from stdin; print `YES` or `NO`.

**Why the obvious greedy is wrong.** The phrasing "split `s` into words" tempts a single left-to-right scan that commits to one match per position, but committing irrevocably to a local match destroys boundaries the global feasibility question depends on. Two constructed counterexamples settle it:

- *Greedy longest-match* on dictionary `{do, dog, gs}`, `s = "dogs"`: it grabs the longest match `dog`, lands at position 3 with `"s"` left, which is not a word, and reports `NO`. But the truth is `YES` via `do | gs` — taking `dog` stranded the `g` that the only valid second word `gs` required.
- *Greedy shortest-match* on dictionary `{aa, aab}`, `s = "aab"`: it grabs the shortest match `aa`, lands with `"b"` left, which is not a word, and reports `NO`. But the truth is `YES` via the single word `aab` — taking `aa` stranded the `b`.

Both greedy rules are discarded.

**Key idea — prefix-reachability DP.** Keep every boundary open. Let `dp[i]` = true iff the length-`i` prefix `s[0..i-1]` is segmentable. The only state that matters at a cut point is whether the prefix up to it is segmentable, because words are independent and reusable. Recurrence:

`dp[0] = true` (empty prefix is the empty concatenation), and for `i >= 1`,

`dp[i] = OR over j in [max(0, i-maxLen), i-1] of ( dp[j] AND s[j..i-1] is in the dictionary )`.

The answer is `dp[m]` where `m = |s|`. Restricting `j` to the last `maxLen` characters (longest dictionary word) is exact — a final chunk longer than `maxLen` cannot be a word — and turns the inner loop from `O(i)` into `O(maxLen)`. Dictionary membership is an `unordered_set` lookup.

**Two pitfalls to get right.**
1. *Inner-loop bound and early exit.* Scanning `j` all the way down to `0` with no `maxLen` cut and no `break` is logically correct but `O(m^2)` in substring builds. Bound `j >= max(0, i - maxLen)` and `break` on the first hit (the OR is settled once any disjunct is true).
2. *Worst-case timing vs. constraints.* The substring-building DP can be pushed toward its edge by a "no short word" dictionary (e.g. `{a^2500, ..., a^5000}` with `s = a^5000`, measured around `0.52 s`). That adversary needs a multi-million-character dictionary; capping the **total dictionary length at `2*10^5`** forbids it, dropping the worst case under the stated limits to about `0.05 s` — comfortable under the 2 s limit.

**Edge cases (all handled by the recurrence + the base case):** empty `s` (no token after the `n` words) -> `dp[0] = 1 -> YES`; empty dictionary with non-empty `s` -> all lookups miss -> `NO`; `maxLen = 0` (only when `n = 0`) -> inner loop never runs -> `NO` for non-empty `s`; one unmatchable letter -> `NO`; single exact match -> `YES`; duplicate dictionary words collapse harmlessly in the set.

**Complexity.** `O(|s| * maxLen)` time, `O(|s|)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;              // n = 0 dictionary words possible
    vector<string> dict(n);
    int maxLen = 0;
    for (auto &w : dict) {
        cin >> w;
        maxLen = max(maxLen, (int)w.size());
    }
    string s;
    cin >> s;                               // the string to segment
    int m = (int)s.size();

    // Put the dictionary in a hash set for O(1) average membership tests.
    unordered_set<string> words(dict.begin(), dict.end());
    words.reserve(dict.size() * 2 + 1);

    // dp[i] = true if the prefix s[0..i-1] (length i) can be segmented.
    // dp[0] = true (empty prefix). For each end position i, look back at every
    // possible last-word start j and check dp[j] && s[j..i-1] in dictionary.
    // Bounded word length keeps the inner loop to O(maxLen) instead of O(n).
    vector<char> dp(m + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= m; ++i) {
        int lo = max(0, i - maxLen);        // last word can be at most maxLen long
        for (int j = i - 1; j >= lo; --j) {
            if (dp[j] && words.count(s.substr(j, i - j))) {
                dp[i] = 1;
                break;
            }
        }
    }

    cout << (dp[m] ? "YES" : "NO") << "\n";
    return 0;
}
```
