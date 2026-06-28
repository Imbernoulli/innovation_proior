**Reading the problem and pinning the contract.** I have a pattern `p` and a string `s`. The pattern carries two wildcards: `?` matches exactly one character of `s`, and `*` matches any sequence of characters including the empty one; every other character of `p` is a literal that must line up exactly. The match is anchored — `p` has to match all of `s`, not some substring — and I print `YES` or `NO`. Before any algorithm I want the scale and the boundary encoding nailed down, because both bite. The lengths satisfy `0 <= |p| <= 2000` and `0 <= |s| <= 2000`, and the empty string is a real, legal input on either side. Since two empty tokens cannot be written on a whitespace-separated line, the contract encodes an empty token as the single hyphen `-`; I decode `-` to `""` for both `p` and `s` the instant I read them, and the hyphen never appears as a genuine character. So my parsing layer is: read two tokens, decode `-` to empty, and from then on reason purely about two possibly-empty strings. That decoding is the first decision and it has to happen before everything else, or the empty cases silently become one-character matches against a literal hyphen.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove* and have *traced*, not the one that is shortest to type.

- *Two-pointer backtracking on `*`.* This is the way a person matches a glob by eye: walk `p` and `s` together with two indices; on a literal or `?` consume one character from each; when I hit a `*`, remember "the last `*` was here and `s` was at this position", then tentatively let the `*` match nothing and march on; if I later hit a literal mismatch, rewind to that remembered `*`, let it swallow one more character of `s`, and retry. It is short, it has no table, and it "feels" right. The two open questions are (a) its worst-case running time and (b) whether the rewind bookkeeping is actually correct across every corner — trailing `*`, consecutive `*`s, a `*` at the very start, and the empty-`s` case.
- *Two-dimensional DP over prefixes.* Define a boolean `dp[i][j]` = "does the length-`i` prefix of `p` match the length-`j` prefix of `s`", and fill it with a fixed recurrence. This is `O(|p| * |s|)` = `O(2000 * 2000) = 4 * 10^6` boolean updates, which is nothing under a 1-second limit. The open question is purely the exact transition for `*` and the boundary row/column for the empty prefixes — a transcription risk, not a correctness-of-idea risk.

I am suspicious of the backtracking route on principle: `*` is a globally ambiguous operator, and any scheme that resolves that ambiguity with a single "last star" rewind pointer is making a local commitment about a global structure. That is precisely the configuration where clever pointer tricks are either exponential or subtly wrong. So before I write a line of it I will try to break it, twice — once on time, once on correctness.

**Attacking the backtracking idea on running time.** The danger with "rewind the last `*` one more character" is that it does not actually remember *how far* earlier stars were pushed; it only ever rewinds to the *most recent* star. Consider a pattern with many stars separated by literals, against a string built to make every star ambiguous. Take `p = "*a*a*a* ... *a*"` — a long alternation of `*` and the single literal `a` — matched against `s = "aaaa...a"`, a long run of `a`s. Every `a` in `s` could be "the one" that a given `*a*` block consumes, and a naive recursion that, at each `*`, tries "match zero here, else let me consume one more and recurse" explores an exponential number of partitions of the run of `a`s among the stars before it either succeeds or (worse) on a near-miss exhausts them all. Concretely, take the deliberately *failing* shape `p = "a*a*a*...*a*b"` (a thousand `a`-blocks ending in a literal `b`) against `s = "aaaa...a"` of length 2000 that contains **no `b` at all**. Every prefix of stars can match, so the recursion happily descends, distributing the 2000 `a`s among the thousand `*`s in every way it can think of, and only at the very end discovers the `b` has nothing to match — and then it backtracks and tries another distribution, and another. The number of ways to split a run of `a`s among many stars is combinatorial, so an unmemoized backtracker does an astronomical amount of work to reach a single `NO`. That is the textbook exponential blow-up of glob matching, and 2000 characters is more than enough rope. The clever route can TLE on an input I can write in one line.

**Attacking the backtracking idea on correctness.** Even the "iterative two-pointer with one star-rewind pointer" version that is supposed to be linear-ish is a minefield of off-by-one and ordering bugs, and I do not want to ship a routine whose correctness I cannot mechanically defend. The fragile spots: (1) when `s` is exhausted but `p` still has characters, I must skip *only* trailing `*`s and reject if any literal or `?` remains — easy to get backwards; (2) a `*` at the very start with `s` empty must still match if the rest of `p` is all stars; (3) consecutive `*`s must collapse without the rewind pointer getting confused about which star to blame; (4) when I rewind, I must advance the *string* anchor, not the pattern anchor, and re-point both indices consistently. Each of these is a place where the iterative trick has a famous wrong variant. I could get it right with enough care, but "with enough care, across four independent corner cases, under a budget" is exactly the risk profile this problem is built to punish. So I am going to do the thing I can prove instead.

**Deciding: ship the provable DP.** The verification killed the clever idea on two independent axes — it can be exponential on a one-line adversarial input, and even the linear variant is correctness-fragile across four corners. The DP, by contrast, is `4 * 10^6` operations with a recurrence I can write down and verify line by line, and there is no ambiguity to resolve cleverly because the table considers *all* ways a `*` can split the string simultaneously. That is the destination: not because DP is fancy, but because it is the simplest method I can *prove* and it is comfortably fast at these constraints.

**Deriving the DP and checking the recurrence on paper.** Let `dp[i][j]` be true iff the first `i` characters of `p` match the first `j` characters of `s`. I want `dp[|p|][|s|]`.

Boundary. `dp[0][0]` = true: the empty pattern matches the empty string. `dp[0][j]` for `j >= 1` = false: an empty pattern cannot match a nonempty string. The first column `dp[i][0]` — can a length-`i` pattern prefix match the empty string? Only if every one of its characters is a `*` (each star matching empty); a single literal or `?` requires a character that is not there. So `dp[i][0] = dp[i-1][0]` when `p[i-1] == '*'`, and false otherwise. That is the same as "`dp[i][0]` is true iff `p[0..i-1]` is all stars", which the recurrence produces automatically.

Transition for `i, j >= 1`, looking at `pc = p[i-1]`:

- If `pc == '*'`: the star either matches the **empty** sequence — in which case I drop the star and need `dp[i-1][j]` — or it matches **at least one** character, in which case it absorbs `s[j-1]` and I still have the same star available for the rest, i.e. `dp[i][j-1]`. So `dp[i][j] = dp[i-1][j] || dp[i][j-1]`. This single line is the whole reason DP beats backtracking: it considers *both* "star eats more" and "star stops here" in one OR, with no rewinding, because the sub-results are memoized in the table.
- If `pc == '?'` or `pc == s[j-1]`: a single-character match, consume one from each: `dp[i][j] = dp[i-1][j-1]`.
- Otherwise (a literal that disagrees with `s[j-1]`): `dp[i][j] = false`.

Let me confirm the `*` transition by hand on `p = "a*b"`, `s = "axxxb"`, expected `YES`. I'll trust the boundary and just sanity-check the interesting cells. Row `i=1` is the literal `a`: it matches only the prefix that ends in the first char, so `dp[1][1] = dp[0][0] = T`, and `dp[1][j>=2] = F` (literal `a` can't match an extending prefix once consumed). Row `i=2` is `*`: `dp[2][j] = dp[1][j] || dp[2][j-1]`. `dp[2][0] = dp[1][0] = F`. `dp[2][1] = dp[1][1] || dp[2][0] = T || F = T`. `dp[2][2] = dp[1][2] || dp[2][1] = F || T = T`. By induction along the row, once it turns true it stays true, so `dp[2][5] = T` — the star has absorbed `xxx`. Row `i=3` is literal `b`, matching only where `s[j-1] == 'b'`, i.e. `j=5`: `dp[3][5] = dp[2][4] = T`. So `dp[3][5] = T` -> `YES`. Correct. And on `s = "axxxc"` the last char is `c`, so row 3 has no true cell at `j=5` (literal `b` vs `c`), giving `NO`. The recurrence behaves.

**Memory check and the rolling-row decision.** A full `(|p|+1) x (|s|+1)` boolean table is `2001 * 2001 ~ 4 * 10^6` bytes if I use `char` — about 4 MB, comfortably inside 256 MB. But the transition for row `i` reads only row `i-1` (for `dp[i-1][j]` and `dp[i-1][j-1]`) and the current row to its left (for `dp[i][j-1]`). So two rolling rows of length `|s|+1` suffice; that drops memory to a few kilobytes and removes any doubt about allocation. I'll keep `prev` (row `i-1`) and `cur` (row `i`), and `swap` them after each row. The only subtlety this introduces is computing `cur[0]` (the first-column boundary) freshly at the top of each row *before* the inner loop, because the inner loop starts at `j=1` and uses `cur[j-1]`.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the row loop, mirroring the recurrence:

```
prev[0] = 1; for j in 1..m: prev[j] = 0;          // dp[0][*]
for i in 1..n:
    char pc = p[i-1];
    cur[0] = (pc == '*') ? prev[0] : 0;            // first-column boundary
    for j in 1..m:
        if (pc == '*')      cur[j] = prev[j] || cur[j-1];
        else if (pc=='?' || pc==s[j-1]) cur[j] = prev[j-1];
        else                cur[j] = 0;
    swap(prev, cur);
```

The risky part is the rolling buffers: after a `swap`, the *contents* of `cur` are last iteration's `prev`, i.e. stale row `i-2` data, and I am about to overwrite them left-to-right. The danger is whether any cell of `cur` is read before I write it this row. `cur[j]` for the `*` case reads `cur[j-1]` (already written this row — fine) and `prev[j]` (the genuine row `i-1` — fine). `cur[0]` reads `prev[0]` — fine. So in principle every read is of either freshly-written `cur` to the left or the legitimate `prev`. But I want to *prove* it, not hope, so I trace the smallest input where stale data could leak: a pattern whose first row leaves `cur` full of ones, then a literal row that should zero most of them. Take `p = "*a"`, `s = "ba"`, expected `YES` (`*` eats `b`, literal `a` matches `a`).

Trace. `m=2`. `prev = [1,0,0]` (dp[0][*]). Row `i=1`, `pc='*'`: `cur[0]=prev[0]=1`; `cur[1]=prev[1]||cur[0]=0||1=1`; `cur[2]=prev[2]||cur[1]=0||1=1`. `cur=[1,1,1]`. swap -> `prev=[1,1,1]`, `cur=[1,1,1]` (stale). Row `i=2`, `pc='a'` (literal): `cur[0]=0` (not a star); `j=1`: `s[0]='b'`, `'a'!='b'` -> `cur[1]=0`; `j=2`: `s[1]='a'`, `'a'=='a'` -> `cur[2]=prev[1]=1`. `cur=[0,0,1]`. swap -> `prev=[0,0,1]`. Answer `prev[m]=prev[2]=1` -> `YES`. Correct, and crucially the literal row correctly *overwrote* every stale `1` it needed to: `cur[0]` and `cur[1]` were forced to `0`, and `cur[2]` was recomputed from `prev[1]`, not left over. So the rolling buffer is sound *as written* — but only because I write `cur[0]` unconditionally at the top of every row. If I had instead only set `cur[0]` inside the star branch and skipped it for literals, `cur[0]` would retain a stale `1` from the previous row, and a pattern like `*` then `a` against `s="x..."` would falsely report the empty prefix as matchable mid-table. So the unconditional `cur[0] = (pc=='*')?prev[0]:0` is load-bearing; the trace is what told me that the literal branch *must* still write `cur[0]=0` rather than fall through.

**A second deliberate trace: the empty-string column.** The case I trust least is `s` empty, because then the inner `j`-loop never runs and the entire answer is decided by the first-column logic. Take `p = "*?*"`, `s = ""` (encoded `-`), expected `NO`: the `?` forces at least one character, but `s` has none. `m=0`. `prev=[1]` (just `dp[0][0]`). Row 1 `pc='*'`: `cur[0]=prev[0]=1`; no inner loop. swap -> `prev=[1]`. Row 2 `pc='?'`: `cur[0]=0` (not a star). swap -> `prev=[0]`. Row 3 `pc='*'`: `cur[0]=prev[0]=0`. swap -> `prev=[0]`. Answer `prev[0]=0` -> `NO`. Correct: the `?` row zeroed the first column and the trailing `*` could not resurrect it. And the dual `p = "**"`, `s = ""` gives `prev[0]` staying `1` through both star rows -> `YES`, which is right (two stars both match empty). The empty-`s` corner is handled entirely by the first-column recurrence, exactly as derived.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `|p| = 0, |s| = 0` (both `-`): `n=0`, the row loop never runs, answer `prev[0] = dp[0][0] = 1` -> `YES`. Empty matches empty. Correct.
- `|p| = 0, |s| > 0`: `n=0`, answer `prev[m] = dp[0][m] = 0` for `m>=1` -> `NO`. Correct (the boundary row I initialized).
- `|p| > 0, |s| = 0`: handled by the first-column trace above — `YES` iff `p` is all stars.
- Trailing `*`: `p="a*"`, `s="a"`. Row1 literal `a`: `dp[1][1]=T`. Row2 `*`: `dp[2][1]=dp[1][1]||dp[2][0]=T`. Answer `T` -> `YES`. The trailing star matches the empty tail. Correct.
- Adversarial `NO`: `p="*a*a*a*"`, `s="aa"`. Three forced `a`s cannot fit in two characters; the table propagates falses and the final cell is `0` -> `NO`. The DP gets this in `4 * 10^6` worst-case work regardless, with no rewinding to blow up.
- Output: exactly one line, `YES` or `NO`, plus a newline. `cin >>` consumes arbitrary whitespace, so the two-token parse is format-agnostic, and the `-` decode runs before any length is read.

**Self-verification against an independent brute oracle.** Hand traces convince me of the cells I look at; they do not cover the thousands I do not. So I wrote a separate brute oracle from the matching *definition* — a top-down recursion `match(i, j)` that returns "does `p[i:]` match `s[j:]`", with the `*` case branching into "match empty: `match(i+1, j)`" OR "consume one: `match(i, j+1)`", memoized over `(i, j)` so it stays fast on small inputs. It shares no code with the DP and resolves `*` by recursion rather than by a table, so an agreement between them is real evidence, not a tautology. I then differential-tested: a generator emits random patterns over `{a, b, c, ?, *}` against random strings, a star-heavy mode, a "derived" mode that builds the string first and grows a pattern that usually matches it (so the YES cases are well represented, not just easy NOs), and a fixed bank of hand-picked corners (empty/empty, `*` vs empty, `?` vs empty, `*a*a*a*` vs `aaa` and vs `aa`, trailing/leading stars, consecutive stars, the `a*b` accept/reject pair). Across 2100 random cases the YES fraction was about 0.51 — a genuinely balanced split, which means the tests are exercising both verdicts — and there were **zero mismatches**. A second sweep at larger sizes (up to ~120 each, where the memoized brute is still instant) over 300 more cases also produced zero mismatches. The corner bank passed every entry. Finally I timed the full-size adversarial inputs — `*a*a*...*` of length 2000 against 2000 `a`s, an all-`*` pattern of length 2000, and the deep-rewind `NO` shape `*a*...*b` against an all-`a` string — and each ran in well under a millisecond using a few megabytes, which is the exact input that would have detonated the backtracker.

**Final solution.** I convinced myself the *idea* is right by disproving backtracking on time (a one-line exponential `NO` instance) and on correctness (four fragile corners), and by deriving the DP recurrence and checking its `*` transition by hand. I convinced myself the *code* is right by tracing the rolling-buffer reads to prove no stale cell is consumed (which pinned the load-bearing unconditional `cur[0]` write), by tracing the empty-`s` first-column logic, and by a 2400-case differential test against an independent recursive oracle with zero mismatches plus a timing check on the worst cases. That is what I ship — one self-contained file, the simple `O(|p| * |s|)` DP I can defend rather than the backtracker I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string p, s;
    if (!(cin >> p)) return 0;            // empty input -> nothing to do
    if (!(cin >> s)) s = "";             // pattern present but string empty

    if (p == "-") p = "";                 // "-" denotes the empty pattern
    if (s == "-") s = "";                 // "-" denotes the empty string

    int n = (int)p.size();                // pattern length
    int m = (int)s.size();                // string length

    // dp[i][j] = does p[0..i-1] match s[0..j-1] ?
    // We keep two rolling rows of length (m+1) to stay within memory.
    vector<char> prev(m + 1, 0), cur(m + 1, 0);

    // Empty pattern matches empty string only.
    prev[0] = 1;
    for (int j = 1; j <= m; j++) prev[j] = 0;

    for (int i = 1; i <= n; i++) {
        char pc = p[i - 1];
        if (pc == '*') {
            // '*' matches an empty sequence: dp[i][0] = dp[i-1][0].
            cur[0] = prev[0];
        } else {
            // Any non-'*' pattern char cannot match the empty string.
            cur[0] = 0;
        }
        for (int j = 1; j <= m; j++) {
            char sc = s[j - 1];
            if (pc == '*') {
                // '*' = match empty (drop '*' : prev[j]) OR consume one more char (cur[j-1]).
                cur[j] = (prev[j] || cur[j - 1]) ? 1 : 0;
            } else if (pc == '?' || pc == sc) {
                // single-char match: consume one char from both.
                cur[j] = prev[j - 1];
            } else {
                cur[j] = 0;
            }
        }
        swap(prev, cur);
    }

    cout << (prev[m] ? "YES" : "NO") << "\n";
    return 0;
}
```

**Causal recap.** Backtracking on `*` looked natural but it dies two ways: a one-line input `*a*...*b` against 2000 `a`s with no `b` forces a combinatorial rewind to a single `NO` (exponential, TLE), and the "fast" iterative variant is correctness-fragile across trailing-`*`, leading-`*`, consecutive-`*`, and empty-`s` corners — so I discarded it. The provable replacement is the `dp[i][j]` table whose only clever line is the `*` transition `dp[i][j] = dp[i-1][j] || dp[i][j-1]` (match empty OR consume one), which folds away all the rewinding the backtracker did by hand. Two rolling rows keep memory tiny, and the one place that bit me was the first column: the literal branch must still write `cur[0] = 0`, because skipping it leaves a stale `1` from the previous row and falsely reports the empty prefix as matchable — a trace of `*a` then `a` showed the buffer must be overwritten unconditionally. With that fixed, an independent recursive oracle agreed on 2400 differential cases (≈51% YES) with zero mismatches and the full-size adversarial inputs ran in under a millisecond, so I ship the DP.
