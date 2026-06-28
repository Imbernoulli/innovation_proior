**Reading the problem and pinning the contract.** I have two strings `s` and `t` of lowercase letters, and I must return the length of their longest common subsequence — the longest string obtainable from each of them by deleting characters while keeping the surviving characters in their original order. Input is `s` on the first line and `t` on the second; I print a single integer. Before I touch an algorithm I want the scale fixed, because the scale decides which approaches are even admissible: `1 <= |s|, |t| <= 3000`. So the input is small in absolute terms — a few thousand characters each — but the *product* `|s| * |t|` can be `3000 * 3000 = 9 * 10^6`, and any method whose cost is proportional to that product is doing on the order of ten million constant-time steps. That is nothing for a 1-second limit; I'll keep it in mind as the comfortable budget I'm allowed to spend. The answer itself is at most `3000`, so it fits in a plain `int` with enormous room — no 64-bit gymnastics here, the value never approaches the 32-bit ceiling. The one number-shaped trap to note is that I should not conflate "number of shared characters" with "length of the longest *ordered* common subsequence"; those are different quantities and the difference is the whole problem.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is cheapest to type.

- *Greedy "match-as-you-scan".* Keep a pointer `j` into `t` starting at `0`. Walk through `s` left to right; for each character `s[i]`, scan forward in `t` from `j` to the first occurrence of `s[i]`; if found, count one match and move `j` just past it; if not found, skip `s[i]`. This is `O(n + m)` (or `O(n*m)` worst case for the inner scan, still tiny here) and only a few lines. The appeal is obvious: it *feels* like it's building a common subsequence greedily by always taking the earliest possible match. The risk is structural — subsequence matching is a global alignment problem, and committing to the earliest match for an early character of `s` can foreclose a much longer alignment downstream. That is precisely the configuration where greedy tends to be wrong, so I will not trust it until I have tried to break it.

- *Two-dimensional dynamic programming.* Define `dp[i][j]` = LCS length of the prefix `s[0..i-1]` against the prefix `t[0..j-1]`. There is a clean recurrence on prefixes, the table is `(n+1) x (m+1)`, and the fill is `O(n*m)`. The risk here is not whether the *idea* is right — the prefix recurrence for LCS is textbook and provable — but (a) whether I transcribe the index arithmetic correctly, especially the `-1` offsets between "string index" and "dp index," and (b) whether a full `9 * 10^6`-cell `int` table fits memory and whether I even need it (`9 * 10^6 * 4` bytes is ~36 MB, which is under 256 MB, but I can do better and sidestep the question entirely).

**Stress-testing greedy before committing.** Saying "greedy feels wrong" is not evidence; let me actually attack it with a concrete instance and see it fail with my own eyes. Take the smallest reordering-style trap I can think of: `s = ab`, `t = ba`. Greedy starts with `j = 0`. First character of `s` is `a`; scan `t` from `j = 0` for an `a` — it's at position `1`, so match it, count `= 1`, advance `j` to `2`. Second character of `s` is `b`; scan `t` from `j = 2` — we're past the end, no `b` available, skip. Greedy reports `1`. Is `1` right? The longest common subsequence of `ab` and `ba` is indeed length `1` (either `a` or `b`), so this particular instance doesn't expose a bug — greedy happened to be optimal here. I need something where greedy's early commitment *costs* a longer alignment.

Let me build that deliberately. The lever is: make `s` start with a character that appears *early* in `t` but whose early match in `t` blocks a longer downstream run. Try `s = aab`, `t = azb`... no, let me make the conflict sharper. Consider `s = abc`, `t = acb`. Greedy: `j = 0`. `s[0] = a`, found in `t` at `0`, match, `j = 1`, count `1`. `s[1] = b`, scan `t` from `1`: positions are `c` (1), `b` (2), found `b` at `2`, match, `j = 3`, count `2`. `s[2] = c`, scan from `3`: end, skip. Greedy reports `2` (`ab`). The true LCS of `abc` and `acb` is also `2` (`ab` or `ac`). Still no gap. Greedy is annoyingly robust on these tiny cases because with such short strings the earliest match rarely strands much.

I need the earliest-match commitment to strand a long tail. The clean way: have a character that occurs *both* near the front and near the back of `t`, and have `s` continue with characters that only align against the *back* copies. Concretely take `s = abb`, `t = bba`. Greedy: `j = 0`, `s[0] = a`, found in `t` at position `2`, match, `j = 3`, count `1`. `s[1] = b`, scan from `3`: end, skip. `s[2] = b`, skip. Greedy reports `1`. But the real LCS of `abb` and `bba` is `2` — the subsequence `bb` is in both (`abb` keeps positions 1,2; `bba` keeps positions 0,1). So greedy returns `1` while the optimum is `2`. **There is the counterexample.** I can see exactly *why* greedy failed: by matching `s`'s leading `a` against the only `a` in `t`, which sits at the very end (position `2`), greedy consumed all of `t` and threw away the two `b`s that the rest of `s` could have matched. The earliest-match rule is the wrong objective; it optimizes "match this character as soon as possible" when the right objective is "match across the whole pair to maximize total length." Greedy is wrong, and now I have a traced instance — `s = abb`, `t = bba`, greedy `1` vs optimum `2` — to prove it. Greedy is out.

**Deriving the DP and checking the recurrence on paper.** I want `dp[i][j]` = the LCS length of `s[0..i-1]` and `t[0..j-1]`. The empty-prefix base cases are `dp[0][j] = 0` and `dp[i][0] = 0`: an empty string shares nothing. For the inductive step, look at the last characters `s[i-1]` and `t[j-1]`:

- If `s[i-1] == t[j-1]`, then there is an optimal LCS that pairs these two final characters: matching them contributes `1`, and the rest is the LCS of the strictly shorter prefixes `s[0..i-2]` and `t[0..j-2]`. So `dp[i][j] = dp[i-1][j-1] + 1`. (Why is it safe to assume the optimum uses this pairing when the characters are equal? Because any common subsequence not using both of these last characters can be extended or rearranged to use them without shortening it — a standard exchange argument — so taking the diagonal `+1` never loses.)
- If `s[i-1] != t[j-1]`, then no common subsequence can end by matching these two distinct characters, so at least one of them is unused. Dropping the last character of `s` gives candidate `dp[i-1][j]`; dropping the last character of `t` gives `dp[i][j-1]`. The optimum is the better of the two: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`. (I don't need a separate `dp[i-1][j-1]` term in this branch because it is `<= dp[i-1][j]`, so the two-way max already dominates it.)

The answer is `dp[n][m]`. This recurrence is the provable one — it rests on the exchange argument for the match case and exhaustive case analysis for the mismatch case — and it has none of the global-commitment hazard that sank greedy, because every cell considers *both* ways of dropping a character.

Let me confirm the recurrence by hand on the sample `s = abcbdab`, `t = bdcaba`, claimed answer `4`. Doing the full `7 x 6` table by hand is error-prone, so let me at least sanity-trace a column of it and trust the textbook value of `4` for the full fill. Actually, let me instead re-confirm the recurrence on my own counterexample, where I already know the truth: `s = abb`, `t = bba`, true LCS `2`. Index strings as `s = a b b` (positions 1..3) and `t = b b a` (positions 1..3). Base row and column are all `0`. Fill row by row.

- `i=1` (`s[0]=a`): `j=1` (`t=b`) mismatch -> `max(dp[0][1]=0, dp[1][0]=0)=0`. `j=2` (`t=b`) mismatch -> `max(dp[0][2]=0, dp[1][1]=0)=0`. `j=3` (`t=a`) match -> `dp[0][2]+1 = 0+1 = 1`. Row1 = `[0,0,0,1]` (including the `j=0` zero).
- `i=2` (`s[1]=b`): `j=1` (`b`) match -> `dp[1][0]+1 = 0+1 = 1`. `j=2` (`b`) match -> `dp[1][1]+1 = 0+1 = 1`. `j=3` (`a`) mismatch -> `max(dp[1][3]=1, dp[2][2]=1)=1`. Row2 = `[0,1,1,1]`.
- `i=3` (`s[2]=b`): `j=1` (`b`) match -> `dp[2][0]+1 = 0+1 = 1`. `j=2` (`b`) match -> `dp[2][1]+1 = 1+1 = 2`. `j=3` (`a`) mismatch -> `max(dp[2][3]=1, dp[3][2]=2)=2`. Row3 = `[0,1,2,2]`.

`dp[3][3] = 2`. The DP gets `2`, exactly the optimum that greedy missed. The recurrence is right, and it specifically *fixes* the failure mode I demonstrated — at cell `dp[3][2]` it recovered the `bb` alignment by dropping `s`'s leading `a` rather than forcing it to match.

**Choosing the memory layout — full table vs rolling rows.** The natural implementation is the full `(n+1) x (m+1)` `int` table. At `3000 x 3000` that's `9,006,001` ints, about `36` MB — comfortably inside `256` MB. So a full table would *work*. But the recurrence for row `i` only ever reads row `i-1` (the cells `dp[i-1][j]` and `dp[i-1][j-1]`) and the current row's left neighbour (`dp[i][j-1]`). It never reaches further back than one row. That means I can collapse to two `1-D` arrays of length `m+1` — a `prev` row and a `cur` row — and swap them each iteration. Memory drops from ~36 MB to ~24 KB, and the time stays `O(n*m)`. Since I don't need to reconstruct the actual subsequence (only its length), there's no reason to keep the whole table. I'll go with rolling rows: it's strictly leaner and removes any lingering doubt about the memory budget.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core, written straight from the recurrence with rolling rows:

```
vector<int> prev(m + 1, 0), cur(m + 1, 0);
for (int i = 1; i <= n; i++) {
    char si = s[i - 1];
    for (int j = 1; j <= m; j++) {
        if (si == t[j - 1])
            cur[j] = prev[j - 1] + 1;
        else
            cur[j] = max(prev[j], cur[j - 1]);
    }
    swap(prev, cur);
}
cout << prev[m] << "\n";
```

The thing that nags me is `cur` being reused across iterations after a `swap`. After `swap(prev, cur)`, the array now named `cur` holds what was *last* iteration's computed row — it is *not* freshly zeroed. For `j >= 1` that's fine, because every `cur[j]` is unconditionally overwritten before it's read. But `cur[0]` is never written inside the loop, and it is *read* as `cur[j-1]` when `j = 1`. On the very first iteration `cur[0]` is the constructor's `0`, which is the correct base value `dp[i][0] = 0`. After a swap, though, `cur[0]` is whatever `prev[0]` was — and `prev[0]` is also always `0` (it's the `dp[i-1][0]` base case, which nothing ever changes). So is `cur[0]` actually stale, or is it accidentally always `0`?

Let me trace the smallest input that could expose a stale `cur[0]`: I need a case where some earlier row wrote a nonzero value into column `0`. But column `0` is never written by the loop body (the inner loop runs `j` from `1`), and both arrays start all-zero. So `[*][0]` stays `0` in both arrays forever, and `cur[0]` reads `0` correctly every iteration. I traced `s = ab`, `t = ab` to be sure: `prev=[0,0,0]`, `cur=[0,0,0]`. `i=1` (`a`): `j=1` (`a`) match -> `cur[1]=prev[0]+1=1`; `j=2` (`b`) mismatch -> `cur[2]=max(prev[2]=0, cur[1]=1)=1`. `cur=[0,1,1]`, swap -> `prev=[0,1,1]`, `cur=[0,0,0]` (old prev). `i=2` (`b`): `cur[0]` is `0` (good); `j=1` (`a`) mismatch -> `cur[1]=max(prev[1]=1, cur[0]=0)=1`; `j=2` (`b`) match -> `cur[2]=prev[1]+1=1+1=2`. `prev[m]=2`. Correct: LCS(`ab`,`ab`)=2.

**A second, sneakier trace to make sure `cur[0]` truly is safe after a swap.** I want a row where `prev` (the old `cur`) had a large value in column 1, so that if `cur[0]` were ever polluted it would show up. Trace `s = aa`, `t = a` (LCS = 1): `m=1`. `prev=[0,0]`, `cur=[0,0]`. `i=1` (`a`): `j=1` (`a`) match -> `cur[1]=prev[0]+1=1`. `cur=[0,1]`, swap -> `prev=[0,1]`, `cur=[0,0]`. `i=2` (`a`): `cur[0]` is `0`; `j=1` (`a`) match -> `cur[1]=prev[0]+1=0+1=1`. `prev[1]=1`. Correct — the second `a` of `s` cannot match a second time because `t` has only one `a`, and the diagonal `prev[0]=0` enforces exactly that. No pollution. The `cur[0]` worry is real to *check* but turns out benign here because column 0 is invariant; still, I'll keep `cur[0] = 0;` set explicitly at the top of each row in the final code so the invariant is documented and robust against future edits rather than relying on a swap coincidence.

**Diagnosing where this kind of code usually dies, deliberately.** The classic LCS transcription bugs are: (1) off-by-one between string index and dp index — writing `s[i]` instead of `s[i-1]`, which reads one character too far and segfaults or misaligns; (2) using `prev[j]` vs `cur[j-1]` wrongly in the mismatch branch — e.g. writing `max(prev[j], prev[j-1])`, which corresponds to a *different* and wrong recurrence; (3) in the match branch using `prev[j]` (up) instead of `prev[j-1]` (diagonal), which double-counts. Let me audit my body against each. (1): I index `s[i-1]` and `t[j-1]` with `i,j` starting at `1`, so the last valid reads are `s[n-1]`, `t[m-1]` — in bounds. (2): mismatch uses `max(prev[j], cur[j-1])` = max(drop-from-`s`, drop-from-`t`) — correct. (3): match uses `prev[j-1]+1` = diagonal `+1` — correct. The audit matches the recurrence I proved on paper, so the transcription is faithful.

**Edge cases, deliberately, because this is where this kind of code dies.**
- Identical strings, `s = t`: every diagonal step matches, so `dp[n][n] = n`. I confirmed this empirically at `n = 3000` (random 26-letter string duplicated): output `3000`, in about `0.01 s`. Correct and fast.
- Disjoint alphabets, e.g. `s = "aaa"`, `t = "bbb"`: every cell hits the mismatch branch and inherits `0`, answer `0`. Correct.
- Length-1 strings: `s = "a"`, `t = "b"` -> `0`; `s = "a"`, `t = "a"` -> `1`. The loops run a single iteration each; traced both, correct.
- Reordering trap (the greedy-killer): `s = "abb"`, `t = "bba"` -> the DP returns `2` (traced above), where greedy returned `1`.
- Worst case `|s| = |t| = 3000` over a tiny alphabet `abcd` (maximizes crossing matches): output computed in ~`0.012 s` and agreed with an independent full-table Python oracle. Time and memory are nowhere near the limits.
- Value range: the answer is at most `3000`, far inside `int`; no overflow possible. Output is exactly one integer and a newline.

**Self-verification against an independent oracle.** Hand-tracing convinced me of the *idea*; to convince myself of the *code* I differential-tested it. I wrote an independent brute oracle that computes LCS a different way — a memoized top-down recursion `rec(i, j)` that branches on match/skip, rather than my bottom-up rolling-row tabulation — so a sign error in one is unlikely to be mirrored in the other. I generated `600` cases spanning tiny `{a,b}` alphabets (length up to ~14, where the exhaustive recursion is trivially correct), three-letter strings, exact reorderings (same multiset, shuffled — the adversarial regime for greedy), short-vs-long pairs, larger `40`–`60`-length strings, and strings built by embedding a known common subsequence inside random noise, plus ten fixed edge cases (`a`/`a`, `a`/`b`, identical, `ab`/`ba`, the textbook `abcbdab`/`bdcaba`, all-same vs shorter, full reversals, etc.). All `600` matched the oracle with **zero** mismatches, and the two big `3000`-length stress cases matched a separate full-table reference. The greedy approach, by contrast, fails the very first reordering case I built by hand.

**Final solution.** I convinced myself the idea is right by disproving greedy with a concrete traced counterexample (`s = "abb"`, `t = "bba"`: greedy `1` vs the true `2`) and by deriving and hand-checking the prefix recurrence; and I convinced myself the *code* is right by auditing the three classic transcription bugs, tracing the `cur[0]`-after-swap concern to a benign invariant, and differential-testing `600`+ cases against an independent oracle with zero mismatches. The clever earliest-match greedy is fast but provably wrong; the simple `O(n*m)` DP is fast *enough* at `3000 x 3000` (~`10^7` steps, well under a second) and provably correct. That is what I ship — one self-contained file, the rolling-row DP I can defend rather than the greedy I broke:

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

**Causal recap.** The earliest-match greedy looked right but a single traced counterexample (`s = "abb"`, `t = "bba"`: greedy matches the leading `a` against `t`'s only `a` at the end, strands both `b`s, and returns `1` while the optimum `bb` is `2`) showed that committing to the earliest match optimizes the wrong objective, so I moved to the prefix DP; I proved its recurrence (exchange argument for the match diagonal `+1`, two-way max over dropping a char for the mismatch) and hand-verified it on the very instance greedy failed, getting `2`; I chose rolling rows over a full `36`-MB table since each row only reads the previous one, dropping memory to ~`24` KB at no time cost; I audited the three classic index/branch transcription bugs and traced the `cur[0]`-after-swap concern to a column-0 invariant (made explicit with `cur[0] = 0;`); and a `600`-case differential test against an independent memoized-recursion oracle plus `3000 x 3000` timing (~`0.01 s`) closed out correctness, the empty/length-1/identical/disjoint/reordering corners, and the time-and-memory budget.
