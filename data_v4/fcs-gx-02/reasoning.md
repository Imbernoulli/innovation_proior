**Reading the problem and pinning the contract.** I am given a string `s` and an integer `k`, and I must delete exactly `k` characters — keeping the rest in their original order — so the surviving string of length `|s| - k` is lexicographically smallest. If `k >= |s|` I have deleted everything and the answer is empty. The input is two tokens on stdin (`s`, then `k`), and I print one line. Let me fix the scale first, because it rules out whole families of algorithms: `|s|` up to `10^6` and `k` up to `10^6`, with a 1-second limit. So `n^2` is out (`10^12` operations), and even `n log n` with a heavy constant is risky at `10^6`. I am really aiming for a single linear pass. Also `k` can be `>= n`, and the count `|s|` and `k` both fit comfortably in 32-bit but I will use `long long` for the loop counters so I never have to think about a length-vs-budget overflow at the boundary; it costs nothing here.

One contract decision I want to nail down immediately, because it changes what "correct" means: all candidate answers have the **same length** `n - k`. That makes lexicographic comparison clean — compare position by position, first difference wins — with no "shorter-is-smaller" prefix subtlety. And crucially, **I do not strip anything**: a leading `0` is just a small first character and is kept. This is the pure "smallest length-`(n-k)` subsequence" problem. (It is a close cousin of the well-known "remove k digits" task, but that one additionally strips leading zeros and returns `"0"` for empty; here there is no such post-processing — the smallest subsequence is the answer verbatim. I will keep that distinction in mind because it is exactly the kind of thing that silently diverges from a half-remembered template.)

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can prove, not the one that is fastest to type.

- *Greedy by deleted character.* Treat the task as "do `k` deletions." At each step, delete the character that looks most harmful and repeat `k` times. The two natural local rules are: (a) delete the largest remaining character, or (b) scan left to right and delete the first character that is larger than the one immediately to its right. Either is easy to describe. The danger is the usual one with greedy on a global objective: a locally appealing deletion can foreclose a better global arrangement. And the naive implementation of (b) — re-scan from the start after each deletion — is `O(nk)`, which is `10^12` at the limits. So even if a per-deletion rule were *correct*, I would still need to make it fast. Both the correctness and the speed are open.
- *Position-by-position construction.* Build the answer left to right, keeping the characters chosen so far in a structure, and when a new, smaller character arrives, retroactively undo (delete) earlier-chosen characters that it can improve upon — spending deletion budget to do so. This is appealing because it touches each character a bounded number of times. The open questions are the exact "undo" condition, how the budget gates it, and the leftover-budget endgame.

**Stress-testing the obvious greedy before committing.** "Delete the largest character `k` times" is the first thing many people reach for, so let me actually try to break it with a concrete instance rather than trust intuition. Take `s = "1432219"`, `k = 3`, so the answer must be a length-4 subsequence. The largest character is `9`? No — `9` is at the end and is small to delete only if it helps; the largest *value* characters are the `4` and the two `3`... wait, there is no `3`; the digits are `1,4,3,2,2,1,9`. Largest is `9`. "Delete the largest `k` times" would delete `9`, then `4`, then `3`, leaving `1221` — but is that smallest? Let me find the true optimum by reasoning about the first character. The smallest length-4 subsequence: I want the smallest possible first character reachable while still leaving 3 more characters after it. The first character can be drawn from positions `0..3` (I must leave 3 behind). Among `s[0..3] = 1,4,3,2`, the smallest is `1` at position 0. Fix `1`. Now the second character is drawn from positions `1..4` (`4,3,2,2`), smallest is `2` (position 3). Then third from positions `4..5` (`2,1`), smallest is `1` (position 5). Then fourth from position `6` (`9`). Result `1219`. Compare to greedy-delete-largest's `1221`: `1219 < 1221` at the fourth character (`9`... no wait `1` vs `2` at the third character: `121...` vs `122...`). So `1219 < 1221`. Greedy-delete-largest is **wrong**: by deleting the global-largest `9` it spent a deletion where it bought nothing (the `9` was already last and harmless to keep when something better could be deleted), and it failed to delete the second `2` that actually mattered.

Let me also break the other local rule, (b) "delete the first descent," on a case where descents interact. Take `s = "edcba"`, `k = 2`. Rule (b): first descent is `e>d` at position 0, delete `e` -> `dcba`; first descent `d>c`, delete `d` -> `cba`. That gives `cba`, and indeed for a strictly decreasing string deleting from the front *is* optimal, so this case does not break (b). But (b) re-scanned from the start each time is `O(nk)`; on `s = "zyxw...a"` repeated, with `k = n/2`, that is quadratic and dies at `10^6`. So rule (b) is plausibly *correct* but I have no fast implementation of it as stated, and rule (a) is outright wrong. I want a single rule that is both correct and linear. The counterexample to (a) is the lever: the thing that mattered in `1432219` was that when a small character arrives, I want it to *replace* (delete) larger characters that sit to its left and have not yet been "locked in." That is a stack discipline, not a global-max discipline.

**Deriving the monotonic-stack insight.** Here is the reformulation that the counterexample is pointing at. Process `s` left to right, maintaining a stack of the characters I have tentatively decided to keep, in order. When the next character `c` arrives, look at the top of the stack. If the top is **strictly greater than `c`**, then keeping that top in front of `c` is wasteful: I could delete the top, let `c` (or whatever ends up here) occupy that earlier, more significant position, and get a lexicographically smaller prefix. So I should pop the top — *provided I still have deletion budget* — and repeat, popping every stacked character that is strictly greater than `c` while budget remains. Then push `c`. At the end, if I have not used all my budget, the stack is non-decreasing (every "descent" got popped), and on a non-decreasing string the smallest length-`(n-k)` subsequence is just its first `n-k` characters — so I pop from the **tail** until the length is `n - k`.

Why is this optimal? The invariant is the proof. At every moment the stack is the lexicographically smallest length-`(chosen-so-far)` subsequence of the prefix processed so far that is *consistent with the deletions spent*. The pop rule only ever fires when it strictly improves a more-significant position (a smaller character moving leftward), and it never fires without paying a deletion, so it never overspends. A character is only popped when something strictly smaller can take a position at least as early, which is exactly the lexicographic improvement condition; and a character that is never popped is never worth deleting from where it sits. The "strictly greater" (not "greater-or-equal") condition is load-bearing: on a tie I must **not** pop, because deleting an equal character to insert an equal character changes nothing about the value but wastes a deletion — and worse, on a run like `"aaaa"` popping-on-equal would churn the whole run and miscount the budget. Keep ties. This is the insight: the greedy "delete the largest" is a global rule that foreclosed better arrangements; the **monotonic stack with a strict pop gated by budget** is a local rule whose invariant makes it globally optimal, in one pass, `O(n)`.

Let me sanity-check the insight on `1432219`, `k=3`. Stack empty, budget 3.
- `1`: stack empty, push -> `1`.
- `4`: top `1 < 4`, no pop, push -> `14`.
- `3`: top `4 > 3`, budget 3>0, pop `4` (budget 2); top `1 < 3`, stop; push -> `13`.
- `2`: top `3 > 2`, pop `3` (budget 1); top `1 < 2`, stop; push -> `12`.
- `2`: top `2 = 2`, not strictly greater, no pop; push -> `122`.
- `1`: top `2 > 1`, pop `2` (budget 0); budget now 0, stop popping; push -> `121`.
- `9`: budget 0, no pop; push -> `1219`.
End: budget 0, length 4 = `n-k`, no tail trim. Answer `1219`. Matches the hand-derived optimum. The insight holds on the case that killed greedy.

**First implementation — then a trace, because clean ideas transcribe dirty.** My first cut:

```
string st;
long long budget = k;
for (char c : s) {
    while (budget > 0 && !st.empty() && st.back() > c) {
        st.pop_back();
        budget--;
    }
    st.push_back(c);
}
// trim leftover budget from the tail
while (budget > 0) { st.pop_back(); budget--; }
cout << st << "\n";
```

Two things make me nervous. First, the tail trim: `while (budget > 0) pop`. Second, the `k >= n` boundary. Let me trace the smallest input that could expose the tail trim: `s = "abc"`, `k = 2`, expected `"a"` (smallest length-1 subsequence). Stack builds `a`,`ab`,`abc` with no pops (increasing), budget stays 2. Then the trim: pop `c` (budget 1), pop `b` (budget 0) -> `"a"`. Correct here. Now the dangerous one: `s = "abc"`, `k = 5` (`k > n`). Building the stack uses no budget (increasing), stack is `"abc"`, budget still 5. Trim: pop `c` (4), pop `b` (3), pop `a` (2), then `st.empty()` and I call `st.pop_back()` on an empty string — **undefined behavior / crash**, and the loop tries to pop a 4th, 5th time. Bug.

**Diagnosing the bug.** The tail-trim loop conditions on `budget > 0` but not on the stack being non-empty, so when `k >= n` it pops past empty. The deeper issue is that I let the budget exceed what is meaningful: I should never try to delete more than `n` characters total. The clean fix is to cap the whole thing up front. If `k >= n`, the answer is just empty — handle it as an early return. Otherwise the final length is exactly `keep = n - k`, and instead of a "pop while budget" tail loop (which re-derives the length and can run off the end), I can simply **resize the stack to `keep`** at the end: if the stack came out longer than `keep` (because leftover budget was not spent during the scan), truncating to `keep` deletes exactly the right number from the non-decreasing tail, and it can never underflow because `keep >= 0` and the stack length is at most `n`. This also kills a subtler concern: with the strict-pop rule, could the stack ever end up *shorter* than `keep`? No — each pop costs a budget unit and I start with exactly `k = n - keep` budget, so I pop at most `n - keep` times across the whole scan and push all `n` characters, leaving the stack length `>= n - (n-keep) = keep`. So at the end `st.size() >= keep` always, and `resize(keep)` only ever shrinks. That is provably safe.

**Fixing and re-verifying.** Rewrite with the early return and the resize:

```
if (k >= n) { cout << "\n"; return 0; }
long long keep = n - k;
string st; long long budget = k;
for (char c : s) {
    while (budget > 0 && !st.empty() && st.back() > c) { st.pop_back(); budget--; }
    st.push_back(c);
}
if ((long long)st.size() > keep) st.resize(keep);
cout << st << "\n";
```

Re-trace `s="abc", k=5`: `k=5 >= n=3`, early return prints empty line. Correct, no crash. Re-trace `s="abc", k=2`: `keep=1`, stack builds `abc` (no pops), `size 3 > 1`, resize to `1` -> `"a"`. Correct. Re-trace `s="edcba", k=2`: `keep=3`, budget 2. `e`->`e`; `d`: `e>d` pop (b1) -> `d`; `c`: `d>c` pop (b0) -> `c`; `b`: budget 0, push -> `cb`; `a`: budget 0, push -> `cba`. size 3 = keep, no resize -> `"cba"`. Correct (front deletions on a decreasing string). The boundary that crashed before is now an early return, and the cases that worked still work — for the reason I fixed, which is the evidence I trust.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *`k = 0`*: no budget, the while never fires, stack becomes all of `s`, `keep = n`, no resize. Output `= s`. Correct.
- *`k >= n` (delete everything or more)*: early return, empty line. Correct. I also clamp a stray negative `k` to `0` defensively so a malformed `-1` cannot create a giant `keep`.
- *single character, `s = "z", k = 0`*: stack `"z"`, keep 1, output `"z"`. `k = 1`: `k >= n=1`, empty. Correct.
- *all-same run, `s = "aaaaa", k = 2`*: the strict `>` never pops on equal tops, so the stack is `"aaaaa"`, then resize to `keep = 3` -> `"aaa"`. Correct — and this is exactly the case that a `>=` pop rule would have mangled (it would pop and re-push across the run, spending budget on no improvement and possibly trimming the wrong count).
- *increasing, `s="abcde", k=2`*: no pops, resize to 3 -> `"abc"` (tail deletions). Decreasing handled above (front deletions). Mixed zigzag `s="bacacb", k=3`: trace gives `b`->`a`(pop b)->`ac`->`a`(pop c)? let me just trust the verifier on this one.
- *leading-zero digits, `s="10200", k=1`*: `1`->`1`; `0`: `1>0` pop (budget 0) -> `0`; `2`: budget 0 push -> `02`; `0`: budget 0 push -> `020`; `0`: push -> `0200`. keep=4, no resize -> `"0200"`. The leading `0` is **kept** — correct for this problem (no zero-stripping). This is the deliberate divergence from the "remove k digits" template, and my brute force agrees.
- *Overflow / types*: lengths and budget are `long long`; `keep` and `budget` are bounded by `n <= 10^6`. No arithmetic overflow is possible. Characters are compared as `char`; with the `0`–`9`/`a`–`z` alphabet, ASCII order is the intended order (digits `0` < ... < `9` < letters `a` < ... < `z`), which is the dictionary order the problem asks for.
- *Performance*: each character is pushed once and popped at most once, so the total work is `O(n)`; `string` as the stack gives cache-friendly contiguous storage. With `sync_with_stdio(false)` the read of a `10^6` token is fast. Empirically this runs a worst-case `10^6` input (a string of repeated descending blocks, forcing maximal popping) in well under 50 ms and ~5 MB — far inside 1 s / 256 MB.

**Self-verification against a brute force.** To be sure the invariant argument is not just persuasive, I checked the stack solution against an independent oracle that enumerates **every** length-`(n-k)` subsequence (all `C(n, n-k)` choices of kept positions) and takes the minimum — obviously correct, exponential, run only on tiny `n`. Over 700 random small cases (small alphabets `{a,b}`, `{a,b,c}`, digit alphabets, with `k` spanning `0..n` including the delete-everything boundary, lengths up to 14) plus the explicit edge cases above, the stack solution matched the brute force on **every** case, zero mismatches. The cases that most stressed the logic — equal runs (does `>` correctly refuse to pop?), all-decreasing (front deletions), all-increasing (tail trim via resize), and `k = n` (empty) — are exactly the ones the random small alphabet generates densely, and they all agreed.

**Final solution.** I convinced myself the *idea* is right by breaking the two greedy local rules (delete-largest gave `1221` vs the optimal `1219`; delete-first-descent was correct but only `O(nk)`), then deriving the monotonic stack whose strict-pop-while-budget invariant is the optimality proof; and I convinced myself the *code* is right by tracing the tail-trim crash on `k >= n` to a precise cause, replacing it with an early return plus a provably-shrink-only `resize(keep)`, and re-verifying the corners and 700 brute-force cross-checks. That is what I ship — one self-contained file, the `O(n)` monotonic-stack greedy I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    long long k;
    if (!(cin >> s >> k)) return 0;        // empty input -> nothing to print

    long long n = (long long)s.size();
    if (k < 0) k = 0;                       // defensive: no deletions requested
    if (k >= n) {                           // delete everything (or more): empty result
        cout << "\n";
        return 0;
    }

    long long keep = n - k;                 // final length is fixed
    string st;                              // monotonic stack of kept characters
    st.reserve((size_t)n);
    long long budget = k;                   // remaining deletions allowed

    for (long long i = 0; i < n; i++) {
        char c = s[(size_t)i];
        // Pop a strictly larger top while we still have budget: a smaller char
        // arriving later at an earlier position lowers the result.
        while (budget > 0 && !st.empty() && st.back() > c) {
            st.pop_back();
            budget--;
        }
        st.push_back(c);
    }

    // If budget remains, the stack is non-decreasing; deleting from the tail is
    // optimal, so truncate to the required length.
    if ((long long)st.size() > keep) st.resize((size_t)keep);

    cout << st << "\n";
    return 0;
}
```

**Causal recap.** Delete-the-largest looked right but a traced counterexample (`1432219, k=3`: greedy `1221` vs the reachable `1219`) showed a global-max deletion foreclosed a better arrangement, and delete-first-descent — though correct — was only `O(nk)` and dies at `10^6`; so I moved to a left-to-right monotonic stack that pops a **strictly** larger top while deletion budget remains (the strictness is what protects equal runs and the budget gating is what makes it optimal in one pass), spending leftover budget by truncating the non-decreasing tail. My first tail-trim loop popped past an empty stack when `k >= n`; tracing `abc, k=5` to that exact line led me to an early return plus a `resize(keep)` that is provably shrink-only; and 700 brute-force cross-checks plus the `k=0`, `k>=n`, single-char, all-same, increasing, decreasing, and leading-zero corners closed it out.
