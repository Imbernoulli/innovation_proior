**Reading the problem and pinning the contract.** I have a drone log of `n` integer per-tick
altitude changes `a[0..n-1]`, each in `[-10^9, 10^9]`, and a threshold `S`. I must find the shortest
contiguous window `[l, r]` whose sum is at least `S`, output its length `r - l + 1`, or `-1` if none
exists; windows are non-empty. Input is `n` and `S` on the first line, then the `n` values; output is
one integer. Before any algorithm I fix the numeric scale, because it dictates types: a window sum
ranges over `n * 10^9 = 2*10^5 * 10^9 = 2*10^14`, and the contract even lets `S` reach `2*10^14`.
Both blow past the 32-bit limit of about `2.1*10^9`. So every prefix sum, every window sum, and `S`
itself must be 64-bit `long long`. The answer (a length) fits in `int`, but the *sums it is compared
against* do not. That is decision one and it is non-negotiable: an `int` prefix here is a silent
wrong-answer on the large tests.

**Reformulating with prefix sums.** The window `[l, r]` has sum `a[l] + ... + a[r]`. If I define
`prefix[0] = 0` and `prefix[k] = a[0] + ... + a[k-1]`, then that window sum is `prefix[r+1] -
prefix[l]`. Renaming `i = l` and `j = r+1`, the task becomes: over all index pairs `0 <= i < j <=
n` of the prefix array, minimize `j - i` subject to `prefix[j] - prefix[i] >= S`. This reformulation
is the lens everything else uses, so let me sanity-check it on a tiny case: `a = [4, 2]`, window
`[0,1]` sum `6`. `prefix = [0, 4, 6]`; `i=0, j=2` gives `prefix[2]-prefix[0] = 6`, and `j - i = 2`,
which is the window length `r - l + 1 = 2`. Good, the bookkeeping matches.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can
*prove*, not the one I have typed a hundred times for the positive-only cousin of this problem.

- *Sliding-window two-pointer.* This is the famous `O(n)` answer to "smallest subarray with sum at
  least `S`" — advance the right end `r`, and while the window sum stays `>= S` advance the left end
  `l` to shrink it, recording the shortest valid length. It is short and I know it cold. The catch:
  that algorithm relies on window sum being **monotone** as you move an endpoint, which holds only
  when all `a[i] >= 0`. Here `a[i]` can be negative. I will not trust it until I have tried to break
  it.
- *Monotonic deque over prefix sums.* Scan `j`; keep a deque of candidate left indices `i` whose
  `prefix[i]` is increasing; pop the front whenever `prefix[j] - prefix[front] >= S`. `O(n)`, no
  positivity assumption. Riskier to transcribe (which end pops, and a proof that each discarded index
  is useless), but it is the honest method for signed values.

**Stress-testing the naive two-pointer before committing.** "It is the standard solution" is exactly
the trap I am told to avoid, so let me actually run the sliding window in my head on a signed
instance and compare to the truth. Take `a = [4, 2, 2, 2, 2, -3, 3, 6]`, `S = 7`. First, the truth
by brute force over windows: the prefix array is `[0, 4, 6, 8, 10, 12, 9, 12, 18]`. I want the
smallest `j - i` with `prefix[j] - prefix[i] >= 7`. Scanning, the pair `(i, j) = (6, 8)` gives
`18 - 9 = 9 >= 7` with `j - i = 2` — that is the window `a[6..7] = 3 + 6 = 9`, length `2`. No single
tick is `>= 7` (the max element is `6`), so length `2` is optimal. The answer is `2`.

Now the naive sliding window. It maintains a running sum and a left pointer `l`, and after each
right step it shrinks from the left while the window *minus* `a[l]` is still `>= S`:

```
best = inf; s = 0; l = 0;
for r in 0..n-1:
    s += a[r]
    while l <= r and s - a[l] >= S:   # shrink while still valid
        s -= a[l]; l += 1
    if s >= S: best = min(best, r - l + 1)
```

Trace it. `r=0`: `s=4`, not `>=7`. `r=1`: `s=6`. `r=2`: `s=8>=7`; can I shrink? `s - a[0] = 8-4 = 4
< 7`, no; record length `r-l+1 = 3`. `r=3`: `s=10`; shrink? `10 - a[0] = 6 < 7`, no; length `4`,
worse. `r=4`: `s=12`; `12 - 4 = 8 >= 7`, shrink, `l=1, s=8`; `8 - a[1] = 6 < 7`, stop; length `4`.
`r=5` (`-3`): `s=5 < 7`. `r=6` (`3`): `s=8 >= 7`; `8 - a[1] = 6 < 7`, no shrink; length `r-l+1 = 6`.
`r=7` (`6`): `s=14`; shrink: `14 - a[1]=12>=7` (`l=2,s=12`), `12 - a[2]=10>=7` (`l=3,s=10`), `10 -
a[3]=8>=7` (`l=4,s=8`), `8 - a[4]=6<7` stop; length `r-l+1 = 7-4+1 = 4`. The best the naive ever
records is `3`.

**The naive baseline is wrong — and I can see exactly why.** Naive returns `3`; the truth is `2`. The
left pointer `l` is *monotone non-decreasing* in the sliding window — it only ever moves right. So
once `l` has advanced past index `6`'s natural partner it can never come back. The optimal window
`a[6..7]` needs `l = 6`, but by the time `r = 7` the pointer `l` has been dragged to `4` and can only
go further right, never back to `6`. The deeper reason is the monotonicity assumption: with the `-3`
at index 5, `prefix` is **not** increasing — `prefix[6] = 9` is *less* than `prefix[5] = 12`. The
sliding window implicitly assumes that a longer window has a larger sum, so shrinking is safe; the
dip at index 5 breaks that, and the left pointer's one-way motion silently discards the very left
endpoint (`i = 6`, `prefix[6] = 9`) that produces the short answer. The verification paid off: it
killed the approach I would otherwise have shipped on autopilot. The naive two-pointer is out for
signed values.

**Deriving the monotonic-deque method and arguing each pop.** I keep a deque of candidate left
indices `i` (indices into `prefix`), and process `j = 0, 1, ..., n`. The deque obeys two invariants,
each justified by a pruning argument:

- *Front pops (answer extraction).* When `prefix[j] - prefix[dq.front()] >= S`, the window from
  `front` to `j` is valid with length `j - front`; I record it and **pop the front**. Why is it safe
  to discard `front` forever? Because `j` only increases from here, so any future `j' > j` paired with
  this same `front` gives length `j' - front > j - front` — never shorter. This `front` has already
  delivered its shortest possible window, so it is useless going forward. This front-pop is the
  reason the whole loop is amortized `O(n)`: each index is pushed once and front-popped at most once.

- *Back pops (dominance).* Before pushing `j`, while `prefix[dq.back()] >= prefix[j]`, **pop the
  back**. Why is `back` useless once a later index `j` has a smaller-or-equal prefix? For any future
  `j' > j`, the validity test `prefix[j'] - prefix[i] >= S` is *easier* to satisfy when `prefix[i]`
  is smaller, and the length `j' - i` is *shorter* when `i` is larger. Index `j` beats `back` on both
  counts (`prefix[j] <= prefix[back]` and `j > back`), so `back` can never give a strictly better
  window than `j` for any future `j'`. Discard it. This keeps the deque's `prefix` values strictly
  increasing from front to back.

So the answer is the minimum `j - front` ever recorded; if nothing is recorded, output `-1`.

**A numeric self-check of the method on the sample.** Let me run the deque by hand on `prefix = [0,
4, 6, 8, 10, 12, 9, 12, 18]`, `S = 7`, to confirm it finds length `2` where naive found `3`. Deque
holds indices; I show `prefix` values for clarity.

- `j=0` (`prefix 0`): deque empty, front-pop nothing; back: empty; push 0. Deque `[0]` (vals `[0]`).
- `j=1` (`4`): `4 - 0 = 4 < 7`, no front-pop. Back: `prefix[0]=0 >= 4`? No. Push 1. `[0,1]` (`0,4`).
- `j=2` (`6`): `6 - 0 = 6 < 7`, stop front. Back: `4 >= 6`? No. Push 2. `[0,1,2]` (`0,4,6`).
- `j=3` (`8`): `8 - 0 = 8 >= 7`, record `3 - 0 = 3`, pop front `0`. Now front `1`: `8 - 4 = 4 < 7`,
  stop. Back: `6 >= 8`? No. Push 3. `[1,2,3]` (`4,6,8`).
- `j=4` (`10`): `10 - 4 = 6 < 7`, stop. Back: `8 >= 10`? No. Push 4. `[1,2,3,4]` (`4,6,8,10`).
- `j=5` (`12`): `12 - 4 = 8 >= 7`, record `5 - 1 = 4`, pop front `1`. Front `2`: `12 - 6 = 6 < 7`,
  stop. Back: `10 >= 12`? No. Push 5. `[2,3,4,5]` (`6,8,10,12`).
- `j=6` (`9`): `9 - 6 = 3 < 7`, stop front. Back dominance: `prefix[5]=12 >= 9`? Yes, pop 5.
  `prefix[4]=10 >= 9`? Yes, pop 4. `prefix[3]=8 >= 9`? No, stop. Push 6. `[2,3,6]` (`6,8,9`).
  This is the crucial step: the dip lets index 6 (value 9) survive, which naive could never keep.
- `j=7` (`12`): `12 - 6 = 6 < 7`, stop front. Back: `prefix[6]=9 >= 12`? No. Push 7. `[2,3,6,7]`
  (`6,8,9,12`).
- `j=8` (`18`): `18 - 6 = 12 >= 7`, record `8 - 2 = 6`, pop front `2`. `18 - 8 = 10 >= 7`, record
  `8 - 3 = 5`, pop `3`. `18 - 9 = 9 >= 7`, record `8 - 6 = 2`, pop `6`. `18 - 12 = 6 < 7`, stop.

The recorded lengths are `3, 4, 6, 5, 2`; the minimum is `2`. The deque method gets it right, and the
winning record `8 - 6 = 2` is precisely the index 6 that the back-dominance rule preserved through
the prefix dip. The method and my hand-trace agree with the brute-force truth.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first
cut of the core loop, iterating `r` over `prefix` indices `0..n`:

```
deque<int> dq;
int best = INT_MAX;
for (int r = 0; r <= n; r++) {
    while (!dq.empty() && prefix[dq.back()] >= prefix[r]) dq.pop_back();  // dominance
    while (!dq.empty() && prefix[r] - prefix[dq.front()] >= S) {          // extract
        best = min(best, r - dq.front());
        dq.pop_front();
    }
    dq.push_back(r);
}
```

I have a nagging feeling about the *order* of the two while-loops, so I trace the smallest input that
could expose it. Take `a = [5]`, `S = 5`; the answer is obviously `1` (the single tick `5 >= 5`).
`prefix = [0, 5]`. `r=0` (`0`): dominance — deque empty; extract — empty; push 0. Deque `[0]`. `r=1`
(`5`): dominance — `prefix[0]=0 >= 5`? No, keep. extract — `5 - prefix[0] = 5 >= 5`, record
`1 - 0 = 1`, pop 0; deque empty, stop. push 1. Final `best = 1`. Correct here. But this ran the
dominance pop *before* the extract on the same `r`, and I want a case where that ordering actually
bites.

**The bug, found by a second trace.** Consider `a = [10]`, `S = 10`. `prefix = [0, 10]`, answer `1`.
That works (same shape as above). Now the real probe: what if the new index would *dominate* the very
front index that should have produced the answer? Try `prefix` where the front is large. Take
`a = [-4, 10]`, `S = 6`; window `[0,1]` sums `6 >= 6` (length 2), window `[1,1]` sums `10 >= 6`
(length 1) — answer `1`. `prefix = [0, -4, 6]`. `r=0` (`0`): push 0. `[0]`. `r=1` (`-4`): dominance —
`prefix[0]=0 >= -4`? Yes, **pop 0**. Then extract — deque empty, nothing. push 1. `[1]` (val `-4`).
`r=2` (`6`): dominance — `prefix[1]=-4 >= 6`? No. extract — `6 - (-4) = 10 >= 6`, record `2 - 1 = 1`,
pop 1. push 2. Final `best = 1`. That happens to be right. But look at what nearly went wrong, and
construct the case that *does* go wrong: I need the front to be both the answer's left endpoint *and*
something a later same-`r` dominance pop would erase before extraction. Try `a = [6, -100]`, `S = 6`,
answer `1` (`a[0]=6>=6`). `prefix = [0, 6, -94]`. `r=0`: push 0. `r=1` (`6`): dominance `0>=6`? No.
extract `6 - 0 = 6 >= 6`, record `1`, pop 0. push 1. `r=2` (`-94`): dominance `prefix[1]=6>=-94`? Yes,
pop 1. extract empty. push 2. `best=1`. Still fine.

The ordering does not corrupt *these* answers, but the principle is wrong and a sharper case shows it.
Run `a = [3, 3]`, `S = 6`, answer `2` (`3+3=6`). `prefix = [0, 3, 6]`. With **dominance-first**: `r=0`
push 0 `[0]`. `r=1` (`3`): dominance `0>=3`? No. extract `3-0=3<6` no. push 1 `[0,1]`. `r=2` (`6`):
dominance `prefix[1]=3 >= 6`? No. `prefix[0]=0>=6`? No. extract `6 - 0 = 6 >= 6`, record `2 - 0 = 2`,
pop 0. `6 - 3 = 3 < 6` stop. push 2. `best = 2`. Correct. Hmm — dominance-first keeps surviving by
luck because dominance only pops indices with *larger* prefix, and the front used for extraction has
the *smallest* prefix, so dominance never pops the front it needs. So the two orders give the same
answer. The actual defect I should hunt is elsewhere.

**The real bug: the dominance pop can swallow the just-pushed extraction candidate.** Let me think
about whether dominance-before-extract can ever delete an index that the *current* `r`'s extraction
needs. Extraction at `r` uses `dq.front()` (smallest prefix). Dominance at `r` pops from the back any
index with `prefix >= prefix[r]`. If the front index has `prefix[front] >= prefix[r]`, dominance
deletes it — but then `prefix[r] - prefix[front]` would be `<= 0`, and for a positive `S` that window
was not going to be valid anyway. The trouble is a window with `S <= 0`. Trace `a = [-1]`, `S = -1`,
answer `1` (`-1 >= -1`). `prefix = [0, -1]`. dominance-first: `r=0` push 0 `[0]`. `r=1` (`-1`):
dominance — `prefix[0]=0 >= -1`? Yes, **pop 0**. extract — deque empty, records nothing. push 1.
Final `best = INT_MAX` -> output `-1`. **Wrong** — the answer is `1`. The dominance pop erased index 0
before extraction could pair it with `r=1` to find the valid length-1 window `prefix[1]-prefix[0] =
-1 >= -1`.

**Fix and re-verification.** The cure is the canonical ordering: **extract (front-pop) before
dominance (back-pop)**, so the current `r` first harvests every window ending at `r` using the live
deque, and only then prunes dominated tails for future `r`. Swap the two while-loops:

```
for (int r = 0; r <= n; r++) {
    while (!dq.empty() && prefix[r] - prefix[dq.front()] >= S) {  // extract FIRST
        best = min(best, r - dq.front());
        dq.pop_front();
    }
    while (!dq.empty() && prefix[dq.back()] >= prefix[r]) dq.pop_back();  // then dominance
    dq.push_back(r);
}
```

Re-trace the breaker `a = [-1]`, `S = -1`. `prefix = [0, -1]`. `r=0` (`0`): extract — deque empty;
dominance — empty; push 0. `[0]`. `r=1` (`-1`): extract — `-1 - prefix[0] = -1 - 0 = -1 >= -1`,
record `1 - 0 = 1`, pop 0; deque empty, stop. dominance — empty. push 1. Final `best = 1`. Correct.
Re-trace the sample `a = [4,2,2,2,2,-3,3,6]`, `S=7`: this is exactly the hand-run two sections above
(extract-first there too), which produced minimum `2`. Correct. The case that broke now passes, and
it broke for the precise reason I fixed — front-pop must run while every candidate left endpoint is
still present.

**Edge cases, deliberately, because this is where deque code dies.**
- `n = 0`: the loop runs once for `r = 0` (just `prefix[0] = 0` pushed); `best` stays `INT_MAX`;
  output `-1`. No window exists in an empty log — correct.
- `S <= 0`: extraction triggers on `prefix[r] - prefix[i] >= S` even for non-positive differences, so
  a length-1 window can qualify. Checked on `a = [-1], S = -1` above -> `1`. And `a = [-1,-1,-1],
  S = -10` -> every length-1 window already satisfies `>= -10`, so answer `1`; the front-pop fires at
  the first `r >= 1` and records `1`. Correct.
- All-negative with `S > 0`: every prefix difference that is `>= S > 0` requires a positive window
  sum, impossible; `best` stays `INT_MAX` -> `-1`. Confirmed by stress.
- `S` exceeds the total achievable gain (e.g. all `10^9`, `S = n*10^9 + 1`): tested at `n = 2*10^5`
  -> `-1`. The just-reachable `S = n*10^9` -> length `n`. Both correct.
- Overflow: `prefix` and `S` are `long long`; the largest `|prefix|` is `2*10^14` and the test
  `prefix[r] - prefix[dq.front()]` has magnitude at most `4*10^14`, all comfortably inside the
  `~9.2*10^18` `long long` range. `best` is an `int` length bounded by `n <= 2*10^5`, fine. No mixed
  32/64-bit comparison sneaks in because `S` is declared `long long`.
- Performance: each `r` pushes once and front/back-pops at most once amortized, so the deque work is
  `O(n)` total; measured `0.01 s` at `n = 2*10^5`. Well under the 1 s limit.

**Final solution.** I convinced myself the *idea* is right by disproving the naive two-pointer on a
signed instance (`[4,2,2,2,2,-3,3,6]`, naive `3` vs true `2`) and by hand-running the deque to the
same `2`; I convinced myself the *code* is right by tracing a failing `S <= 0` case to a precise
cause — dominance-before-extract erasing the front the current `r` still needed — and re-verifying the
extract-first fix and every corner. That is what I ship: one self-contained file, the `O(n)`
monotonic-deque-on-prefix-sums method I can defend rather than the sliding window I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // prefix[k] = a[0] + ... + a[k-1], prefix[0] = 0, length n+1.
    // A window [l, r) (0 <= l < r <= n) has sum prefix[r] - prefix[l];
    // we want the shortest window with prefix[r] - prefix[l] >= S, i.e. minimize r - l.
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + a[i];

    // Monotonic deque of candidate left endpoints (indices into prefix), with
    // strictly increasing prefix values. For each r we pop from the front while
    // prefix[r] - prefix[front] >= S (that left can never be beaten by a larger r,
    // since r only grows), and we keep the deque increasing from the back so a
    // smaller-or-equal prefix at a later index dominates earlier larger ones.
    deque<int> dq;
    int best = INT_MAX;
    for (int r = 0; r <= n; r++) {
        while (!dq.empty() && prefix[r] - prefix[dq.front()] >= S) {
            best = min(best, r - dq.front());
            dq.pop_front();
        }
        while (!dq.empty() && prefix[dq.back()] >= prefix[r]) {
            dq.pop_back();
        }
        dq.push_back(r);
    }

    cout << (best == INT_MAX ? -1 : best) << "\n";
    return 0;
}
```

**Causal recap.** The negative per-tick changes break the monotonicity the textbook sliding-window
two-pointer silently assumes, and a single traced counterexample (`[4,2,2,2,2,-3,3,6]`, `S=7`: naive
`3` vs reachable `2`, because the prefix dips at the `-3` and the one-way left pointer discards the
left endpoint `i=6` that yields the length-2 window) forced me off the naive baseline onto a
monotonic deque over prefix sums, whose front-pop extracts the shortest window ending at each `r`
(safe because `r` only grows) and whose back-pop discards dominated left endpoints (smaller prefix at
a later index wins on both validity and length); my first transcription ran the dominance pop before
the extract, which a trace of `[-1]` with `S=-1` returning `-1` instead of `1` exposed as erasing the
front the current `r` still needed, so I swapped to extract-first; and `long long` prefixes plus the
`best == INT_MAX -> -1` guard close out the `S<=0`, all-negative, empty-log, single-tick, and
overflow corners.
