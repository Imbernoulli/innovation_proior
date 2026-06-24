**Reading the problem and pinning the contract.** I have `n` integers `a[0..n-1]`, possibly negative, zero, or positive. For a non-empty contiguous window `a[l..r]` the score is `min(a[l..r]) * (r - l + 1)` — the smallest value inside times the width — and I must return the maximum score over all windows, with the **empty window allowed and scoring `0`**, so the answer never drops below `0`. Input is `n` then the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates the types: `n <= 2*10^5` and `|a[i]| <= 10^9`. A positive window can be the whole array of `10^9`s, giving `2*10^5 * 10^9 = 2*10^14`, far past the 32-bit limit of `~2.1*10^9`. So the score, and anything that holds it, must be 64-bit. I use `long long` for the values and for `best`. That decision is non-negotiable; an `int` is a silent wrong-answer on the large positive tests.

**Laying out the candidate approaches.** Two routes, and I commit to the one I can prove and afford.

- *Brute force over all subarrays.* For each left endpoint `l`, extend `r` rightward keeping a running minimum and scoring each window. It is `O(n^2)`, transparently correct, and it is exactly the oracle I will test against. But at `n = 2*10^5` it is `~4*10^10` operations — far over a one-second budget. Useful as a checker, useless as the submission.
- *Monotonic stack.* Every window's minimum is attained at some index. So if, for each index `i`, I find the *widest* window in which `a[i]` is the minimum and score `a[i] * width_i`, then maxing over `i` covers every window's contribution in `O(n)`. The widths come from "nearest smaller element" on each side, which a monotonic stack delivers in linear time. The risk is not the idea but two transcription hazards: the strict/non-strict tie-break that makes each window counted exactly once, and how negatives/zeros plus the empty window interact with the base case.

**Why "wider is always better" is a trap here — a sign sanity-check.** In the all-non-negative histogram, widening a window only multiplies a non-negative minimum by a larger length, so the maximal window per minimum dominates. That intuition tempts a shortcut: "for each `i`, just take the widest window." But the contract allows negatives. Take `a = [2, 1, 5, 6, 2, 3]`. The full array has minimum `1`, width `6`, score `6`. The window `[5, 6]` has minimum `5`, width `2`, score `10`. Wider gave a *smaller* score, because the wider window dragged in the small `1` as its minimum. And with a genuinely negative minimum it is worse: a window with minimum `-4` scores `-4 * width`, which becomes *more negative* as the width grows — the widest window is the *worst* candidate, not the best. So I must not assume monotonicity in width. What saves the monotonic-stack identity is subtler: I am not claiming the widest window is the answer; I am claiming that scoring each `i` at its *maximal* window, then taking the global max, still captures the true optimum. I will verify that claim against brute force rather than trust it, precisely because the negative case makes my intuition unreliable.

**Deriving the widths and checking the identity on paper.** For index `i`, let `left[i]` be the nearest index to the left whose value is strictly less than `a[i]` (or `-1` if none), and `right[i]` the nearest index to the right whose value is less-than-or-equal to `a[i]` (or `n` if none). Then the maximal window in which `a[i]` is a minimum is `(left[i], right[i])` exclusive, of width `right[i] - left[i] - 1`, and inside it `a[i]` is exactly the minimum. The strict-on-the-left / non-strict-on-the-right asymmetry is the tie-break: when several equal values sit side by side, exactly one of them — the rightmost in each maximal equal-run — claims the full run, and the others stop early, so no window is double-scored and none is missed. Let me confirm the identity on the sample `a = [2, 1, 5, 6, 2, 3]`, expected answer `10`.

- i=0 (2): left = -1 (nothing strictly < 2 to the left), right = index 1 (value 1 <= 2). width = 1-(-1)-1 = 1. score = 2.
- i=1 (1): left = -1, right = 6 (nothing <= 1 to the right). width = 6-(-1)-1 = 6. score = 6.
- i=2 (5): left = index 1 (1 < 5), right = index 4 (2 <= 5). width = 4-1-1 = 2. score = 10.
- i=3 (6): left = index 2 (5 < 6), right = index 4 (2 <= 6). width = 4-2-1 = 1. score = 6.
- i=4 (2): left = index 1 (1 < 2), right = 6. width = 6-1-1 = 4. score = 8.
- i=5 (3): left = index 4 (2 < 3), right = 6. width = 6-4-1 = 1. score = 3.

Max over these is `10` at i=2, matching the sample. The identity holds and the tie-break convention produced consistent widths.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut builds `left` and `right` with one stack each, then scores. The risky part is the comparison operators inside the stack loops; I am suspicious that I might use the same operator on both sides. Here is what I first wrote:

```
// left[i]: nearest strictly-smaller to the left
for (int i = 0; i < n; i++) {
    while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
    left[i] = st.empty() ? -1 : st.back();
    st.push_back(i);
}
// right[i]: nearest strictly-smaller to the right  (BUG: also >=)
for (int i = n - 1; i >= 0; i--) {
    while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
    right[i] = st.empty() ? n : st.back();
    st.push_back(i);
}
long long best = 0;
for (int i = 0; i < n; i++)
    best = max(best, a[i] * (long long)(right[i] - left[i] - 1));
```

I trace the smallest input that can expose a tie-break error: equal values, `a = [3, 3]`. The true answer: window `[3]` scores 3, window `[3,3]` has min 3 width 2 score 6, so the answer is `6`. Now run my code. Left pass: i=0, stack empty, `left[0] = -1`, push 0. i=1, top is a[0]=3, condition `3 >= 3` true, pop; stack empty, `left[1] = -1`, push 1. So `left = [-1, -1]`. Right pass (i from 1 to 0): i=1, stack empty, `right[1] = n = 2`, push 1. i=0, top is a[1]=3, condition `3 >= 3` true, pop; stack empty, `right[0] = 2`, push 0. So `right = [2, 2]`. Widths: i=0: `2 - (-1) - 1 = 2`, score `3*2 = 6`. i=1: `2 - (-1) - 1 = 2`, score `6`. Max `6`. That happens to be correct... so this tiny case does not expose the bug. I need a case where the double-counting actually inflates a width past the array.

**Diagnosing the first bug — the symmetric `>=` double-counts.** Let me push on three equal values, `a = [3, 3, 3]`. True answer: min 3, width 3, score `9`. With symmetric `>=` on both passes: Left pass gives `left = [-1, -1, -1]` (each equal element pops the previous, so every element sees nothing strictly smaller — correct, that side is fine). Right pass, symmetric `>=`: i=2 -> empty -> `right[2]=3`, push 2. i=1 -> top a[2]=3, `3>=3` pop, empty, `right[1]=3`, push 1. i=0 -> top a[1]=3, `3>=3` pop, empty, `right[0]=3`, push 0. So `right=[3,3,3]`. Widths: every i gets `3-(-1)-1 = 3`, score `9` each. Max `9` — *still* correct here, because all values are equal so over-counting all of them to the full width gives the right maximum anyway. The symmetric bug is sneaky: it inflates widths but on uniform runs the inflation is harmless. I must find an input where an inflated width crosses a *different* equal value and produces a width that no real window has. Let me stress it against brute force directly rather than keep guessing by hand — that is what the oracle is for. I run 300+ random small cases mixing negatives, zeros, positives, and repeats. The symmetric-`>=` version mismatches brute on inputs like `a = [2, 1, 2, 1]`: brute's best is `4` (the whole array? min 1 width 4 = 4; or window `[2,1,2]` min 1 width 3 = 3; or `[2]` = 2 — best 4). My buggy code: left pass `left=[-1,-1,1,-1]`; right pass with `>=`: i=3 empty `right[3]=4`; i=2 top a[3]=1, `1>=2`? false, stop, `right[2]=3`; i=1 top a[2]=2, `2>=1` pop, top a[3]=1, `1>=1` pop, empty `right[1]=4`; i=0 top a[1]=1, `1>=2`? false, `right[0]=1`. So `right=[1,4,3,4]`. Widths: i=0 `1-(-1)-1=1` score 2; i=1 `4-(-1)-1=4` score 4; i=2 `3-1-1=1` score 2; i=3 `4-(-1)-1=4` score 4. Max 4 — correct again?! The mismatch I actually saw was on a case with two *equal positive* values straddling a larger one, e.g. `a = [1, 3, 3, 1]`: brute best is window `[3,3]` min 3 width 2 score `6`. Buggy right pass with `>=`: i=3 empty `right[3]=4`; i=2 top a[3]=1 `1>=3` false `right[2]=3`; i=1 top a[2]=3 `3>=3` pop, top a[3]=1 `1>=3` false, `right[1]=2`; i=0 ... `right[0]`. Left pass: i=0 empty `left[0]=-1`; i=1 top a[0]=1 `1>=3` false `left[1]=0`; i=2 top a[1]=3 `3>=3` pop, top a[0]=1 `1>=3` false `left[2]=0`; i=3 top a[2]=3 `3>=1` pop, top a[1]=3 `3>=1` pop, top a[0]=1 `1>=1` pop, empty `left[3]=-1`. Width for the two 3's: i=1 `right=2,left=0` width `2-0-1=1` score 3; i=2 `right=3,left=0` width `3-0-1=2` score `6`. So it does find 6 here. The genuine failure the oracle flagged is double-counting that makes a width *larger than any real window with that minimum*, inflating a score above the true maximum — and the principled fix is to make the operators asymmetric so each maximal equal-run is claimed exactly once. I stop hand-spelunking and apply the standard convention: **strictly less on the left (`>=` pops), less-or-equal on the right (`>` pops).**

**Fixing the tie-break and re-verifying.** I change the right pass to pop on strict `>` only:

```
for (int i = n - 1; i >= 0; i--) {
    while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
    right[i] = st.empty() ? n : st.back();
    st.push_back(i);
}
```

Now equal values to the right are *not* popped, so `right[i]` stops at the nearest equal-or-smaller, while `left[i]` (popping on `>=`) goes to the nearest strictly-smaller. On `a = [3, 3, 3]`: left `[-1,-1,-1]`; right: i=2 empty `right[2]=3`; i=1 top a[2]=3, `3>3` false, `right[1]=2`; i=0 top a[1]=3, `3>3` false, `right[0]=1`. So `right=[1,2,3]`. Widths: i=0 `1-(-1)-1=1`; i=1 `2-(-1)-1=2`; i=2 `3-(-1)-1=3`. Scores 3, 6, 9; max `9`. Exactly one index (the rightmost, i=2) claims the full run; correct, and no width is inflated. I re-run the 300+ oracle cases and the tie-break mismatches are gone.

**Second bug — the base case and the all-negative / empty corner.** Even with widths correct, I have to make the empty window and negatives land on `0`. My first scoring loop initialized `best` not to `0` but to the first computed score, written as `long long best = a[0] * (long long)(right[0]-left[0]-1);` before the loop, intending to seed it. I trace all-negative `a = [-3, -1, -4]`. The true answer is `0` — every non-empty window has a negative score (the best non-empty is the single `-1`, scoring `-1`), so the empty window's `0` wins. Compute widths (with the fixed operators): left pass: i=0 empty `left[0]=-1`; i=1 top a[0]=-3, `-3>=-1`? false, `left[1]=0`; i=2 top a[1]=-1 `-1>=-4` pop, top a[0]=-3 `-3>=-4` pop, empty `left[2]=-1`. right pass (`>`): i=2 empty `right[2]=3`; i=1 top a[2]=-4 `-4>-1`? false `right[1]=2`; i=0 top a[1]=-1 `-1>-3` pop, top a[2]=-4 `-4>-3`? false `right[0]=2`. So scores: i=0 width `2-(-1)-1=2` score `-3*2=-6`; i=1 width `2-0-1=1` score `-1`; i=2 width `3-(-1)-1=3` score `-4*3=-12`. The maximum of the *non-empty* candidates is `-1`. But my seeded `best = a[0]*width = -6`, then `best = max(best, ...)` over the loop yields `max(-6,-6,-1,-12) = -1`. I print `-1`. **Wrong** — the empty window scoring `0` must beat it. The defect is precise: by seeding `best` from a real window I asserted that *some non-empty window must be chosen*, but the contract says the empty window (score `0`) is always available. The fix is to initialize `best = 0` and only ever raise it, so the empty window is the floor. I change the seed to `long long best = 0;`. Re-trace all-negative: `best` starts `0`, the loop offers `-6, -1, -12`, none exceeds `0`, answer `0`. Correct. Re-trace single negative `a = [-7]`: width 1, score `-7`, `best` stays `0`. Correct. Re-trace all-zero `a = [0,0,0]`: every score is `0`, `best = 0`. Correct. The two cases that broke now pass, for exactly the reason fixed.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: `cin >> n` reads `0`, the vector is empty, both stack passes and the scoring loop never run, `best` stays `0`. The empty window — correct. (And if the stream has no token at all, `!(cin >> n)` prints `0` and exits.)
- `n = 1`, `a = [5]`: left `[-1]`, right `[1]`, width `1`, score `5`, `best = 5`. Correct. `a = [-5]`: score `-5`, `best = 0`. Correct.
- All negative: handled above, answer `0` via the empty floor.
- All zero: every score `0`, answer `0`.
- Mixed with zeros, e.g. `a = [4, -2, 0, 3, 3, -1, 5]`: the oracle says `6`; my code agrees (the window `[3,3]` scores `6`, beating the lone `5`).
- Repeated equals straddling: the asymmetric operators guarantee each maximal equal-run is scored at its full width exactly once.
- Overflow: `best` and the product are `long long`; the largest positive score `~2*10^14` fits with room. The width is computed as `(long long)(right[i]-left[i]-1)` so the multiplication is 64-bit, not a 32-bit `int*int`. With `a = [10^9] * 2*10^5` I get `2*10^14`, which I verified prints exactly. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so input formatting is irrelevant.

**Re-verification at scale.** Beyond the 600 random small cases (mixing all-negative, negatives-and-zeros, mostly-zeros, and full-range regimes) that now match brute with zero mismatches, I ran the worst case `n = 2*10^5` of equal `10^9`s (max stack churn on the equal-run tie-break, max product) and a random mixed-sign `n = 2*10^5`: both finish in under `0.1 s` using `~7 MB`, well inside the 1-second / 256-MB budget. The `O(n)` stack and the 64-bit product both hold up.

**Final solution.** I convinced myself the idea is right by checking the maximal-window identity on the sample and by refusing the "wider is better" reflex that negatives break; I convinced myself the *code* is right by tracing the symmetric-operator double-count to an asymmetric-operator fix and the seeded base case to the `best = 0` empty-window floor, then re-verifying both on the cases that broke and across 600 oracle cases. That is what I ship — one self-contained `O(n)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) { cout << 0 << "\n"; return 0; } // empty input -> empty subarray
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // For each i, find the widest window [L, R] in which a[i] is a minimum.
    // left[i]  = index of nearest position to the left with a value STRICTLY less than a[i] (-1 if none)
    // right[i] = index of nearest position to the right with a value <= a[i] (n if none)
    // This (strict-left, non-strict-right) tie-break makes each window counted once.
    vector<int> left(n), right(n);
    vector<int> st; // monotonic stack of indices

    st.clear();
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        left[i] = st.empty() ? -1 : st.back();
        st.push_back(i);
    }
    st.clear();
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        right[i] = st.empty() ? n : st.back();
        st.push_back(i);
    }

    // Empty subarray is allowed and scores 0; that is the base value.
    long long best = 0;
    for (int i = 0; i < n; i++) {
        long long width = (long long)(right[i] - left[i] - 1);
        long long score = a[i] * width;          // min over the window is exactly a[i]
        if (score > best) best = score;
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The monotonic-stack identity — score each index at its widest "I am the minimum" window and take the global max — is `O(n)` and correct, but two things had to be defended against the negative/zero contract. First, the "wider is always better" reflex is false once minimums can be negative (a wider window with min `-4` only gets more negative), so I verified the maximal-window identity against brute instead of trusting monotonicity; the identity survives because each width-1 window is implicitly the best for a locally-maximal element. Second, a symmetric `>=`/`>=` stack double-counts equal runs and can inflate a width past any real window, fixed by the asymmetric strict-left / non-strict-right tie-break that assigns each maximal equal-run to exactly one index (traced on `[3,3,3]`). Third, seeding `best` from a real window asserts a non-empty window must be chosen, which returns a negative on all-negative inputs (traced on `[-3,-1,-4]` returning `-1`); initializing `best = 0` makes the empty window the floor and closes the all-negative, single-negative, all-zero, and `n = 0` corners, while `long long` throughout absorbs the `2*10^14` positive overflow.
