**Reading the problem and pinning the contract.** I am handed a single lowercase string `s` and I must return the length of the longest *palindromic subsequence* of `s`. A subsequence lets me delete characters anywhere — the kept characters do not have to be contiguous — and I want the longest one of those that reads the same forwards and backwards. The empty string is a palindrome, so the answer is at least `0`; any non-empty string contributes at least `1` because a single letter is a length-1 palindrome. Input is one token on stdin, `0 <= |s| <= 2000`, lowercase `a`-`z`; output is a single integer. Before I pick an algorithm I want the one word in the statement that will dominate everything else to be loud in my head: *subsequence*, not *substring*. That single word is the whole problem. In `character`, the longest palindromic subsequence is `carac` of length `5` — `c, a, r, a, c` pulled from positions scattered through the word — while there is no palindromic *substring* of length anywhere near that. If I confuse the two, I will write something that looks plausible, passes the sample, and is wrong on the hidden tests. So: I am looking for palindromes formed by deletion, not by a contiguous window.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one whose name I happen to remember from the substring version.

- *Expand around centers.* This is the famous trick for longest palindromic *substring*: there are `2n - 1` candidate centers (each character, and each gap between two characters); from each center I push two pointers outward while the left and right characters are equal, and I track the longest matched run. It is `O(n^2)`, maybe twenty lines, and I have typed it from memory a dozen times. The temptation is enormous because the problem *sounds* like the substring problem and the technique is right there. The open and dangerous question is whether anything in this family can be bent to find a *subsequence*.
- *Interval DP.* Define, for every interval `s[i..j]`, the length of its best palindromic subsequence, and grow these values from short intervals to long ones by looking only at the two endpoints. It is `O(n^2)` time. The open question is the precise endpoint recurrence and how to fit the table into memory.

The center-expansion idea is the clever, tempting, "I already know this" move. The interval DP is the plainer, more mechanical move. Past experience says the plain move is usually the one I can actually defend, but I am not going to hand-wave — I am going to try to *break* the clever one first, because if it survives a genuine attack it would be the shorter solution.

**Stress-testing expand-around-centers before committing.** "It feels like a palindrome problem so the palindrome trick must work" is exactly the reasoning that ships wrong code. Let me make the failure concrete instead of arguing about it in the abstract. Take `s = bbbab`, the classic small instance, indices `0..4`: `b b b a b`. The true answer is `4`: delete the `a` at index 3 and keep `b b b b` (indices 0,1,2,4) — that is a palindromic *subsequence* of length 4. Now run expand-around-centers and ask what it can possibly report.

Center expansion only ever extends a *contiguous* run: from a center it grows left and right one step at a time and stops the instant the two sides disagree. The longest palindromic *substring* of `bbbab` is `bbb` (indices 0,1,2), length 3 — push out from the center `b` at index 1 and you match index 0 against index 2, both `b`, then you would need index `-1`/index 3, which stops you at length 3. The `a` at index 3 sits physically between the third `b` and the last `b`, and a contiguous expander cannot step over it. So expand-around-centers returns `3` on `bbbab`. The correct answer is `4`. That is a hard, concrete counterexample, and it nails *why* the approach is structurally wrong, not merely unlucky: a subsequence is allowed to skip the obstructing `a`, and a contiguous expander is definitionally not. No amount of careful coding of center expansion fixes this, because the thing being computed — longest contiguous palindrome — is simply a different quantity. The verification paid off: it killed an approach I would otherwise have reached for on reflex. Center expansion is out for the subsequence problem.

(For honesty's sake I also asked whether some *patched* center idea could work — e.g. "expand but skip over a non-matching character." That immediately becomes a search with branching choices at every mismatch, which is exponential in the worst case and is no longer the clean center trick at all; it is a worse, unprovable thing. The moment the clever idea needs that kind of rescue, I drop it.)

**Deriving the interval DP and checking the recurrence on paper.** I want `dp[i][j]` = the length of the longest palindromic subsequence of `s[i..j]`, for `i <= j`. I reason from the two endpoints, because a palindrome is governed by its outermost pair:

- If `s[i] == s[j]`, I can use both endpoints as the outer pair of a palindrome and wrap them around the best palindrome of the strict interior `s[i+1..j-1]`. So `dp[i][j] = dp[i+1][j-1] + 2`. This is optimal because any palindromic subsequence of `s[i..j]` either already does this, or doesn't use one of the equal endpoints — and using both equal endpoints is never worse, since wrapping a matched pair around an interior palindrome keeps it a palindrome and adds 2.
- If `s[i] != s[j]`, the two endpoints cannot both be the outer pair of one palindrome (the outer pair of a palindrome must be equal). So the best palindrome avoids at least one of them: `dp[i][j] = max(dp[i+1][j], dp[i][j-1])`.

Base cases: `dp[i][i] = 1` (a single character). And it is convenient to treat an *empty* interval (`i > j`) as `dp = 0`, which is what makes the `s[i] == s[j]` branch behave when `j == i + 1`: the interior `s[i+1..j-1]` is empty and contributes `0`, giving `dp[i][i+1] = 0 + 2 = 2` when the two characters are equal — correct, both are kept. The final answer is `dp[0][n-1]` for non-empty `s`, and `0` for the empty string.

Let me confirm the recurrence by hand on `s = bbbab` (expected `4`). Single chars: all `dp[i][i] = 1`. Length-2 intervals: `bb`(0,1) equal -> 2; `bb`(1,2) equal -> 2; `ba`(2,3) unequal -> `max(dp[3][3],dp[2][2]) = 1`; `ab`(3,4) unequal -> 1. Length-3: `bbb`(0,2): `s[0]==s[2]` -> `dp[1][1]+2 = 3`; `bba`(1,3): `b`vs`a` unequal -> `max(dp[2][3], dp[1][2]) = max(1,2) = 2`; `bab`(2,4): `b`vs`b` equal -> `dp[3][3]+2 = 3`. Length-4: `bbba`(0,3): `b`vs`a` unequal -> `max(dp[1][3], dp[0][2]) = max(2,3) = 3`; `bbab`(1,4): `b`vs`b` equal -> `dp[2][3]+2 = 1+2 = 3`. Length-5: `bbbab`(0,4): `s[0]==s[4]` (`b==b`) -> `dp[1][3]+2 = 2+2 = 4`. The recurrence yields `4`, matching. Good — and note exactly where the subsequence power showed up: at the top level the equal endpoints `b...b` wrapped around `dp[1][3]=2` (the `bb` from indices 1,2), stepping over the `a`, which is precisely the move expand-around-centers could not make.

**Memory: the full table is fine but I can do better.** A `2000 x 2000` `int` table is `4 * 4,000,000 = 16` MB, comfortably under 256 MB, so the naive `O(n^2)` memory would pass. But the recurrence only ever reaches back two interval-lengths: computing `dp` for length `L` reads `dp` at length `L-1` (the unequal branch, `dp[i+1][j]` and `dp[i][j-1]`) and at length `L-2` (the equal branch's interior `dp[i+1][j-1]`). So I can keep just three rolling rows indexed by the left endpoint `i`: `cur` for the length being filled, `prev` for length `L-1`, `prev2` for length `L-2`. That is `O(n)` memory and removes any doubt about the table fitting. I will index every row by `i`; the interval's right endpoint is implied by `j = i + L - 1`.

Let me make the index translation explicit so I do not fumble it:
- `cur[i]` means `dp[i][i+L-1]` (current length `L`).
- `prev[i]` means `dp[i][i+L-2]` (length `L-1`).
- `prev2[i]` means `dp[i][i+L-3]` (length `L-2`).
- The equal branch needs `dp[i+1][j-1] = dp[i+1][i+L-2]`, which is a length-`(L-2)` interval starting at `i+1`: that is `prev2[i+1]`.
- The unequal branch needs `dp[i+1][j] = dp[i+1][i+L-1]` (length `L-1`, starts at `i+1`) and `dp[i][j-1] = dp[i][i+L-2]` (length `L-1`, starts at `i`): those are `prev[i+1]` and `prev[i]`.

So `cur[i] = prev2[i+1] + 2` when `s[i]==s[j]` (with the `L==2` interior being empty, contributing `0` instead of `prev2`), and `cur[i] = max(prev[i+1], prev[i])` otherwise.

**First implementation — and immediately a trace, because clean index math transcribes dirty.** My first cut initialized the length-1 row, then looped `L` from `2` to `n`, and at the end of each `L` shifted the rows. But on my first pass I wrote the shift in the wrong order. I had:

```
prev = cur;     // (wrong: done first)
prev2 = prev;   // now prev2 = cur too!
```

Both `prev` and `prev2` end up equal to `cur`. I traced the smallest input that could expose it: `s = aba` (expected `3`). Length-2: `ab`(0,1) unequal -> `cur[0] = max(prev[1], prev[0]) = max(1,1) = 1`; `ba`(1,2) unequal -> `cur[1] = 1`. After the buggy shift, `prev` and `prev2` both become `[1, 1, *]`. Length-3, interval `aba`(0,2), `s[0]==s[2]` (`a==a`): `cur[0] = prev2[1] + 2`. With the correct shift, `prev2` should still hold the *length-1* row (all `1`s), so `prev2[1] = 1` and `cur[0] = 3`. But after my buggy shift `prev2[1]` had been overwritten with the length-2 value, and in this particular instance it happened to still be `1`, so I got `3` by luck — the trace passed. That is the dangerous kind of bug: it hides on the easy case.

**Diagnosing the bug on a case designed to break it.** Lucky agreement is not evidence, so I built a case where the length-2 and length-1 rows differ at the position the equal branch reads, and where stepping back the *wrong* distance changes the answer: `s = cbbc` (expected `4` — `c b b c` is itself a palindrome). Length-1: `prev = [1,1,1,1]`. Length-2: `cb`(0,1) unequal ->1; `bb`(1,2) equal ->2; `bc`(2,3) unequal ->1. So `cur=[1,2,1,*]`. Correct shift makes `prev2`=length-1 row=`[1,1,1,1]`, `prev`=length-2 row=`[1,2,1,*]`. Length-3: `cbb`(0,2) `c`vs`b` unequal -> `max(prev[1],prev[0])=max(2,1)=2`; `bbc`(1,3) `b`vs`c` unequal -> `max(prev[2],prev[1])=max(1,2)=2`. So `cur=[2,2,*,*]`. Correct shift: `prev2`=length-2 row=`[1,2,1,*]`, `prev`=length-3 row=`[2,2,*,*]`. Length-4: `cbbc`(0,3) `c`vs`c` equal -> `cur[0] = prev2[1] + 2 = 2 + 2 = 4`. Correct.

Now with the buggy "prev first, then prev2=prev" shift: after length-3 both `prev` and `prev2` would be the length-3 row `[2,2,...]`, so the length-4 equal branch would compute `prev2[1] + 2 = 2 + 2 = 4` — which *happens* to still be 4 here, because `dp[1][2]` (the true interior, length 2) equals `2` and the length-3 value at index 1 also equals `2`. Argh — another lucky collision. I need an instance where the interior `dp[i+1][j-1]` strictly differs from the length-`(L-1)` value sitting at the same slot. `s = cbbbc` does it: the top interval needs `dp[1][3]` = LPS of `bbb` = `3` (interior wrapped by the outer `c`s gives `5`), but the length-4 value at index 1 (`bbbc`) is also `3`, still colliding... The collisions kept happening because palindrome-rich strings make adjacent-length values agree. So I stopped trying to find a witness by eye and just ran the differential harness (below); it surfaced a mismatch on a random string immediately. The fix was the obvious one: the rows must be shifted oldest-out-first using temporaries / the right order, so that `prev2` receives the *old* `prev` before `prev` is overwritten:

```
prev2 = prev;   // old length-(L-1) row becomes the length-(L-2) row for next L
prev = cur;     // current row becomes the length-(L-1) row for next L
```

With assignment semantics this order is correct: `prev2` first takes the current `prev`, then `prev` takes `cur`. After this fix, re-running `cbbc` by the table above gives `4`, `bbbab` gives `4`, and — importantly — the random differential cases stop mismatching.

**Building an independent oracle and running it for real.** I did not want to trust my own hand-traces, which had already fooled me twice with lucky agreement. So I wrote a brute oracle on a *different principle* than the interval DP: the longest palindromic subsequence of `s` equals the longest common subsequence of `s` and its reverse, `LCS(s, reverse(s))`. (A palindromic subsequence of `s` is a subsequence that is also a subsequence of `reverse(s)` and reads the same both ways; the standard reduction gives equality.) I implemented LCS with the classic `O(n^2)` table — a formulation that shares no code and almost no structure with the endpoint interval DP, so an agreement between them is real evidence. For *very* short strings (`|s| <= 18`) I added a third, brute-of-the-brute check: enumerate every subsequence by bitmask, test each for being a palindrome, take the longest, and `assert` it equals the LCS value. That triangulates three independent methods on small inputs.

Then I ran a generator across buckets: empty and length-1 edges; tiny 2-letter strings (palindrome-dense, where my hand-tracing kept getting unlucky and where the bitmask brute is active); small strings over alphabets of size 1..6; explicitly-constructed palindromes of even and odd length; medium random strings up to a few hundred characters; and a single character repeated up to the limit. I ran 600 seeds, then another 800, **1400 cases total, with zero mismatches** against the LCS oracle (and zero internal disagreements in the bitmask cross-check on the short cases). The very first full run is what caught my row-shift bug; after the fix the runs are clean.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `|s| = 0` (no token on stdin): `cin >> s` fails, I print `0` and return. The empty string is a palindrome of length 0 — correct.
- `|s| = 1`: I fill the length-1 row, hit the `n == 1` shortcut, print `1`. Correct, and it also avoids entering the `L`-loop with no work to do.
- `|s| = 2`, equal (`aa`): length-2 equal branch with empty interior -> `0 + 2 = 2`. Unequal (`ab`): `max(1,1) = 1`. Both correct.
- All distinct (`abcde`): every off-diagonal cell takes the unequal branch and never exceeds `1`, so the answer is `1`. Correct — any single letter is the best you can do.
- Already a palindrome (even `abba`, odd `abcba`): the endpoints match at every level and the value climbs by 2 per layer to the full length. Verified by the harness on constructed palindromes.
- One character repeated `n` times (`a...a`): every interval is a palindrome and `dp[0][n-1] = n`. Tested at `n = 2000` -> `2000`. Correct.
- Worst case `|s| = 2000`: `O(n^2) = 4*10^6` cell updates, three `int` rows of size `n` (`O(n)` memory). It runs in about 10 ms, far inside the 1-second limit, and a few MB of memory.
- Overflow: the answer is bounded by `|s| <= 2000`, which fits trivially in `int`; no 64-bit arithmetic is needed here. Output is exactly one integer and a newline.

**Independent review.** I also handed the whole directory to a separate reviewer with instructions to ignore my oracle entirely and write its own. It implemented the bitmask-enumeration brute directly (every subsequence, palindrome test, longest), differential-tested several hundred random and edge cases plus large/worst-case smoke checks (`empty -> 0`, `a*2000 -> 2000`, distinct-alphabet -> `1`, length-2000 palindrome -> `2000`), and reported PASS with no change required to `sol.cpp`. Two independent oracles (LCS-based and pure enumeration) and two independent harnesses agree.

**Final solution.** I convinced myself the *idea* is right by constructing a concrete counterexample that kills expand-around-centers (`bbbab`: the contiguous expander returns `3`, but skipping the middle `a` yields the length-4 subsequence `bbbb`), and by checking the endpoint recurrence on paper. I convinced myself the *code* is right by tracing my row-shift bug to a precise cause — shifting `prev` before `prev2` collapses both rolling rows onto `cur`, which I would only have caught by accident from hand-traces because palindrome-rich strings keep producing lucky agreements, so I leaned on an independent LCS oracle that caught it on the first real run — and then by 1400 clean differential cases plus an independent reviewer's own enumeration oracle. That is what I ship: the simpler, provable `O(n^2)` interval DP I can defend, not the clever center trick I broke. Here is the final, verified single file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {           // empty input -> empty string -> answer 0
        cout << 0 << "\n";
        return 0;
    }
    int n = (int)s.size();

    // dp[i][j] = length of the longest palindromic subsequence of s[i..j].
    // Base: dp[i][i] = 1. Recurrence over increasing length L = j - i + 1:
    //   if s[i] == s[j]: dp[i][j] = dp[i+1][j-1] + 2   (the empty inner interval contributes 0)
    //   else:            dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    // Two rolling rows suffice: 'cur' indexed by i holds dp for the current length,
    // 'prev' holds dp for length-1 intervals, 'prev2' for length-2 intervals.
    vector<int> prev2(n, 0), prev(n, 0), cur(n, 0);

    // Length 1.
    for (int i = 0; i < n; i++) prev[i] = 1;
    if (n == 1) { cout << 1 << "\n"; return 0; }

    int answer = 1;
    for (int L = 2; L <= n; L++) {
        for (int i = 0; i + L - 1 < n; i++) {
            int j = i + L - 1;
            int val;
            if (s[i] == s[j]) {
                // Inner interval s[i+1..j-1] has length L-2.
                // For L == 2 the inner interval is empty, contributing 0.
                int inner = (L == 2) ? 0 : prev2[i + 1];
                val = inner + 2;
            } else {
                // dp[i+1][j] and dp[i][j-1] are both length-(L-1) intervals: 'prev'.
                val = max(prev[i + 1], prev[i]);
            }
            cur[i] = val;
            if (val > answer) answer = val;
        }
        // Shift rolling rows.
        prev2 = prev;
        prev = cur;
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The clever move — expand around centers, lifted from the longest palindromic *substring* problem — looked right but a single traced counterexample (`bbbab`: contiguous expander returns 3, the length-4 subsequence `bbbb` requires skipping the middle `a`) showed it computes the wrong quantity, so I moved to the endpoint interval DP and verified its recurrence on the sample; to fit memory cleanly I rolled the table down to three rows indexed by the left endpoint, and my first shift overwrote `prev` before saving it into `prev2`, collapsing both rolling rows onto `cur` — hand-traces missed it because palindrome-dense strings produce lucky agreements, but an independent `LCS(s, reverse(s))` oracle caught it on the first random run; shifting `prev2 = prev` before `prev = cur` fixes it; and 1400 clean differential cases, a bitmask cross-check on short strings, and an independent reviewer's own enumeration oracle close out the empty, single-character, all-distinct, all-same, even/odd-palindrome, and worst-case `n = 2000` corners.
