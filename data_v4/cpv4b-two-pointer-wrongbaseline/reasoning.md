The word that decides this problem is *net*: each tick's `a[i]` is a net altitude change and can be
negative, so "shortest contiguous window with sum at least `S`" is the **signed** version of a
problem whose famous `O(n)` sliding-window solution assumes all values are non-negative. That is the
trap. Before touching an algorithm I fix the numeric
scale, because it sets the types: a window sum ranges up to `n * 10^9 = 2*10^5 * 10^9 = 2*10^14`, and
`S` alone can reach `2*10^14` — both far past the 32-bit ceiling of about `2.1*10^9`. So every prefix
sum, every window sum, and `S` must be 64-bit `long long`; only the answer, a length bounded by `n`,
fits in `int`. An `int` prefix here is a silent wrong-answer on the large tests, not a crash.

**Prefix-sum reformulation.** Set `prefix[0] = 0`, `prefix[k] = a[0] + ... + a[k-1]`. Then the window
`[l, r]` has sum `prefix[r+1] - prefix[l]`. Renaming `i = l`, `j = r+1`, the task becomes: over index
pairs `0 <= i < j <= n`, minimize `j - i` subject to `prefix[j] - prefix[i] >= S`. This prefix view is
the lens for everything below.

**Two candidate methods, and only one survives contact.** The sliding-window two-pointer — advance
`r`, and while the window sum stays `>= S` advance `l` to shrink it — is the `O(n)` textbook answer,
but it relies on window sum being monotone as an endpoint moves, which holds only
when all `a[i] >= 0`. The alternative is a monotonic deque over prefix sums, which makes no positivity
assumption but needs a real argument for its pop rules. Since the whole premise of this problem is
signed values, I do not get to trust the two-pointer on reputation — I have to try to break it.

**Breaking the naive two-pointer.** Take the sample `a = [4, 2, 2, 2, 2, -3, 3, 6]`, `S = 7`. The
truth first, by prefix scan: `prefix = [0, 4, 6, 8, 10, 12, 9, 12, 18]`; the pair `(i, j) = (6, 8)`
gives `18 - 9 = 9 >= 7` with `j - i = 2` — the window `a[6..7] = 3 + 6 = 9`, length `2`. No single
tick reaches `7` (max element `6`), so `2` is optimal. Now the naive window, which after each right
step shrinks from the left while the window minus `a[l]` is still `>= S`:

Trace it. `r=0`: `s=4`, not `>=7`. `r=1`: `s=6`. `r=2`: `s=8>=7`; can I shrink? `s - a[0] = 8-4 = 4
< 7`, no; record length `r-l+1 = 3`. `r=3`: `s=10`; shrink? `10 - a[0] = 6 < 7`, no; length `4`,
worse. `r=4`: `s=12`; `12 - 4 = 8 >= 7`, shrink, `l=1, s=8`; `8 - a[1] = 6 < 7`, stop; length `4`.
`r=5` (`-3`): `s=5 < 7`. `r=6` (`3`): `s=8 >= 7`; `8 - a[1] = 6 < 7`, no shrink; length `r-l+1 = 6`.
`r=7` (`6`): `s=14`; shrink: `14 - a[1]=12>=7` (`l=2,s=12`), `12 - a[2]=10>=7` (`l=3,s=10`), `10 -
a[3]=8>=7` (`l=4,s=8`), `8 - a[4]=6<7` stop; length `r-l+1 = 7-4+1 = 4`. The best the naive ever
records is `3`.

```
best = inf; s = 0; l = 0;
for r in 0..n-1:
    s += a[r]
    while l <= r and s - a[l] >= S: s -= a[l]; l += 1
    if s >= S: best = min(best, r - l + 1)
```

Tracing it: `r=2` gives `s=8`, no shrink (`8-4=4<7`), length `3`; by `r=4` the left pointer has been
dragged to `l=1`; at `r=6` (`s=8`) it will not shrink (`8-a[1]=6<7`) and records length `6`; at `r=7`
it shrinks to `l=4` and records `4`. The best it ever records is `3`. That is wrong — the truth is
`2`. The reason is exactly the monotonicity failure I feared: with the `-3` at index 5, `prefix` dips
(`prefix[6] = 9 < prefix[5] = 12`), so a longer window no longer means a larger sum. The left pointer
moves only rightward and can never come back to `l = 6`, the endpoint that produces the length-2
window. So the naive baseline is genuinely incorrect here, not merely suboptimal. It is out.

**The monotonic deque, with each pop justified.** I keep a deque of candidate left indices `i` (into
`prefix`) and process `j = 0, 1, ..., n`, maintaining strictly increasing `prefix` values front to
back. Two rules:

- *Front-pop (extract).* When `prefix[j] - prefix[front] >= S`, the window `front..j` is valid with
  length `j - front`; record it and pop the front. Safe to discard `front` forever because `j` only
  increases, so any later `j' > j` paired with this same `front` gives a strictly longer window. Each
  index front-pops at most once, which is what makes the loop amortized `O(n)`.
- *Back-pop (dominance).* Before pushing `j`, while `prefix[back] >= prefix[j]`, pop the back. For any
  future `j'`, a smaller `prefix[i]` is easier to clear `S` and a larger `i` gives a shorter window;
  index `j` beats `back` on both (`prefix[j] <= prefix[back]`, `j > back`), so `back` can never yield
  a strictly better window than `j`. Discard it.

The answer is the minimum `j - front` ever recorded, or `-1` if nothing is recorded. Running this on
the sample prefix `[0, 4, 6, 8, 10, 12, 9, 12, 18]`, the mechanism that matters shows up at `j=6`
(`prefix 9`): back-dominance pops indices 5 (`12`) and 4 (`10`) because both exceed `9`, so index 6
survives in the deque — precisely the left endpoint the naive pointer could never keep. Later at
`j=8` (`18`), extraction records `18 - prefix[6] = 9 >= 7` at length `8 - 6 = 2`. The deque gets the
`2` that the sliding window missed.

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

**A transcription bug: which loop runs first.** The trap in coding this up is the
order of the two while-loops within one `j`. My first cut ran the dominance back-pop before the
extract front-pop. The question is whether dominance can ever delete an index the *current* `j`'s
extraction still needs. Extraction reads `front`, the smallest prefix; dominance pops back indices
with `prefix >= prefix[j]`. These collide only if `prefix[front] >= prefix[j]` — but then
`prefix[j] - prefix[front] <= 0`, which for a positive `S` was never going to be a valid window, so
nothing is lost. The exposure is `S <= 0`. Concretely, `a = [-1]`, `S = -1`, whose answer is `1`
(`-1 >= -1`): `prefix = [0, -1]`. Dominance-first, at `j=1`, pops index 0 (`prefix[0] = 0 >= -1`)
*before* extraction can pair it, leaving the deque empty; nothing is recorded and the program prints
`-1`. That is the wrong answer, and it is wrong for the reason I just isolated.

The fix is the canonical ordering: extract before dominance, so the current `j` first harvests every
window ending at it while all candidate left endpoints are still present, and only then prunes
dominated tails for future `j`. Re-running `a = [-1]`, `S = -1` with extract-first: at `j=1`,
`-1 - prefix[0] = -1 >= -1` records `1`, then dominance runs on the emptied deque — output `1`,
correct.

**Edge corners.** `n = 0`: the loop runs once for `j = 0` (only `prefix[0]` pushed), nothing recorded,
output `-1`. `S <= 0`: extraction fires on non-positive differences too, so a length-1 window can
qualify — `a = [-1], S = -1` gives `1`. All-negative with `S > 0` needs a positive window sum, so
`-1`; `S` above the total achievable gain is `-1`, exactly the total is length `n`. The overflow guard
is already spent up front: with `prefix` and `S` as `long long` the compared difference tops out
around `4*10^14`, well inside the `~9.2*10^18` range, and the `long long` `S` blocks any mixed 32/64
comparison. Each index pushes once and pops at most once, so `O(n)` — trivially inside 1 s at
`n = 2*10^5`.

I ship the `O(n)` monotonic deque over prefix sums, with extract-first ordering and `long long` sums,
guarding `best == INT_MAX -> -1`. The full program is the single self-contained C++17 file in the
answer.
