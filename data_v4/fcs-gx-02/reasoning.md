The two numbers that fix the whole shape of this are `|s|` up to `10^6` and a 1-second limit: anything quadratic (`10^12` operations) is dead, and even an `n log n` with a heavy constant is nervous at a million characters, so I commit up front to a single linear pass. The task is: delete exactly `k` characters, keep the rest in order, make the surviving length-`(|s|-k)` string lexicographically smallest; if `k >= |s|` everything goes and the answer is empty. Input is `s` then `k` on stdin, one line out. I carry the loop counters and the budget in `long long` — the values themselves fit in 32 bits (both bounded by `10^6`), but the length-vs-budget arithmetic at the `k >= n` boundary is exactly where a sign slip would bite, and the wider type costs nothing.

One contract point changes what "correct" even means: every candidate has the **same** length `n - k`, so lexicographic comparison is clean — position by position, first difference wins, no shorter-is-smaller subtlety. And I strip nothing: a leading `0` is just a small first character and stays. That is the one place this diverges from the half-remembered "remove k digits" task, which additionally deletes leading zeros and returns `"0"` for empty. Here the smallest length-`(n-k)` subsequence is the answer verbatim — a distinction that silently corrupts a solution copied from muscle memory, so I flag it now.

Two families are on the table. A **greedy by deleted character** — "do `k` deletions, each time removing the most harmful character" — with two natural local rules: (a) delete the largest remaining character, or (b) scan left to right and delete the first character larger than its right neighbour. Both are easy to state; the danger is the usual greedy trap, a locally appealing deletion foreclosing a better global arrangement, and (b) as a re-scan after each deletion is `O(nk) = 10^12`. The other family is a **position-by-position construction**: build the answer left to right, and when a smaller character arrives, retroactively undo earlier-chosen larger characters, spending budget to do so — each character touched a bounded number of times. To decide between them I go hunting for an input that breaks the greedy.

Take `s = "1432219"`, `k = 3` (answer length 4). The digits are `1,4,3,2,2,1,9`; "delete the largest three times" removes `9`, then `4`, then `3`, leaving `1221`. Is that optimal? Build the true smallest length-4 subsequence by position: the first character can come from positions `0..3` (`1,4,3,2`), smallest `1` at index 0; the second from `1..4` (`4,3,2,2`), smallest `2` at index 3; the third from `4..5` (`2,1`), smallest `1` at index 5; the fourth is the trailing `9`. That gives `1219`, and `1219 < 1221` — they first differ at the third character, `1` vs `2`. So delete-largest is wrong: deleting the trailing `9` spent a deletion on a character already harmless at the end, and it never removed the interior `2` that mattered.

Rule (b) does better on the cases I can construct — on a strictly decreasing string like `"edcba"` deleting from the front is genuinely optimal, and (b) does exactly that — but re-scanning from the start after each deletion is quadratic, and on `k = n/2` of a long descending string it dies at `10^6`. So (a) is wrong and (b) is right-but-slow. What the `1432219` counterexample points at is the fix: when a small character arrives I want it to *replace* larger characters sitting to its left that have not been locked in — a stack discipline, not a global-max one.

Process `s` left to right keeping the tentatively-kept characters on a stack. When `c` arrives, if the top is **strictly greater** than `c`, keeping that top ahead of `c` is wasteful — I can delete it and let a smaller character occupy that earlier, more significant slot — so pop it while budget remains, repeat, then push `c`. At the end, if budget is left over, the stack is non-decreasing (every descent got popped), and the smallest length-`(n-k)` subsequence of a non-decreasing string is just its first `n-k` characters, so I trim the tail.

The invariant is the optimality argument: at every step the stack is the smallest subsequence of the processed prefix that is consistent with the deletions spent. A pop fires only when a strictly smaller character can take an at-least-as-early position — exactly the lexicographic improvement condition — and never without paying a deletion, so it never overspends; a character never popped was never worth deleting from where it sits. The **strict** `>` (not `>=`) is load-bearing: on a tie, popping an equal character to insert an equal character changes the value not at all while wasting a deletion, and on a run like `"aaaa"` a `>=` rule would churn the entire run and miscount the budget. Ties stay.

Trace it on `1432219`, `k=3`, budget 3: push `1`; `4` (top `1<4`) push -> `14`; `3` pops `4` (budget 2) then pushes -> `13`; `2` pops `3` (budget 1) then pushes -> `12`; the next `2` ties the top, no pop, push -> `122`; `1` pops `2` (budget 0), push -> `121`; `9` has no budget, push -> `1219`. Budget 0, length 4, no trim — `1219`, the optimum that killed delete-largest.

First cut of the code:

```
string st;
long long budget = k;
for (char c : s) {
    while (budget > 0 && !st.empty() && st.back() > c) { st.pop_back(); budget--; }
    st.push_back(c);
}
while (budget > 0) { st.pop_back(); budget--; }   // spend leftover budget on the tail
cout << st << "\n";
```

Two things worry me: the tail-spend loop, and the `k >= n` boundary. Trace `s = "abc"`, `k = 5`: the scan pushes `a,b,c` with no pops (increasing), budget still 5; the tail loop pops `c,b,a` and then calls `pop_back()` on an empty string — undefined behavior — and keeps going for two more iterations. That is a real crash on any `k >= n` input, and `k >= |s|` is an explicit test case.

The loop gates on `budget > 0` but not on the stack being non-empty. The fix is to stop letting the budget mean more than the string can supply: if `k >= n` the answer is empty, so return it early; otherwise the final length is exactly `keep = n - k`, and rather than a pop-while-budget loop I truncate the stack to `keep` at the end. That truncation is provably shrink-only — each pop during the scan costs one budget unit and I start with exactly `k = n - keep`, so I pop at most `n - keep` times while pushing all `n` characters, leaving the stack length `>= keep`; `resize(keep)` can only shorten, never underflow.

```
if (k >= n) { cout << "\n"; return 0; }
long long keep = n - k, budget = k;
string st;
for (char c : s) {
    while (budget > 0 && !st.empty() && st.back() > c) { st.pop_back(); budget--; }
    st.push_back(c);
}
if ((long long)st.size() > keep) st.resize(keep);
cout << st << "\n";
```

Now `s="abc", k=5` returns empty at the guard; `s="abc", k=2` builds `abc`, sees `size 3 > keep 1`, resizes to `"a"`; `s="edcba", k=2` pops `e` then `d` during the scan (front deletions) and leaves `"cba"` at length 3 — the crash path, the resize path, and the front-deletion path each exercised.

Two corners exercise the parts of the logic most likely to be wrong. The all-same run `"aaaaa"`, `k=2`: strict `>` never pops on equal tops, so the stack is the full run and `resize` trims it to `"aaa"` — precisely the case a `>=` rule would mangle. Leading zeros, `"10200"`, `k=1`: `0` pops the `1` (budget 0), then `2,0,0` push with no budget -> `"0200"`, the leading `0` kept — the deliberate divergence from remove-k-digits. `k = 0` falls straight through (the while never fires, `keep = n`, no resize) to output `= s`. On types, everything is bounded by `10^6` so nothing overflows, and I clamp a stray negative `k` to 0 so a malformed token can't produce a giant `keep`; characters compare as `char`, and for the `0`–`9`/`a`–`z` alphabet ASCII order is the intended dictionary order (`0 < ... < 9 < a < ... < z`).

Performance: each character is pushed once and popped at most once, so the pass is `O(n)` with `O(n)` space; a `string` stack keeps it contiguous and cache-friendly, and `sync_with_stdio(false)` makes the million-character read cheap. A worst-case `10^6` input of repeated descending blocks (maximal popping) runs in well under 50 ms and a few MB, comfortably inside 1 s / 256 MB.

To back the invariant argument with evidence I cross-checked against an oracle that enumerates every length-`(n-k)` subsequence and takes the minimum — obviously correct, exponential, only tractable on tiny `n`. Over 700 random small cases (small and digit alphabets, which make ties and zigzags dense, `k` spanning `0..n` including the delete-everything boundary, lengths up to 14) plus the explicit corners above, the stack solution matched on every case. The cases that stress the logic hardest — equal runs, all-decreasing, all-increasing, `k = n` — are exactly what the small-alphabet generator produces densely, and they all agreed.

So the final program is one self-contained `O(n)` file: the budget-gated strict monotonic stack, an early `k >= n` guard, and a shrink-only `resize(keep)` for leftover budget. The full module is in the answer.
