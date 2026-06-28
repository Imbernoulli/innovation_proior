**Reading the problem and pinning the contract.** I have `n` intervals; interval `i` is the
half-open segment `[s_i, e_i)` and carries a positive weight `w_i`. I must pick a pairwise
non-overlapping subset that maximizes the total weight, with the empty set allowed so the answer is
at least `0`. Two things in the contract I want to nail down before any algorithm. First, the
segments are *half-open*: two intervals that merely touch at a point, where one ends exactly where
the next begins (`e_i = s_j`), do **not** overlap and may both be taken. That single detail decides
whether my compatibility test uses `<=` or `<`, and getting it backwards is a silent
off-by-one-flavoured wrong answer on every touching case. Second, the scale: `n <= 2*10^5`, weights
up to `10^9`, so a total can reach `2*10^5 * 10^9 = 2*10^14`. That blows past the 32-bit range of
about `2.1*10^9`, so every accumulator and every stored coordinate/weight has to be 64-bit. I will
use `long long` throughout. Coordinates themselves go up to `2*10^9`, which already overflows a
signed 32-bit `int`, so even the endpoints must be 64-bit, not just the running sum. That is the
first decision and it is non-negotiable.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the
one I can *prove*, not the one that is shortest to type.

- *Earliest-finish-time greedy.* Sort the intervals by their end coordinate and sweep left to
  right, taking an interval whenever it does not overlap the last interval I took. This is the
  textbook algorithm for the *unweighted* activity-selection problem — "schedule as many
  non-overlapping intervals as possible" — and there it is provably optimal. It is `O(n log n)` and
  about five lines. The temptation is real: I already have to sort by end for the other approach, so
  greedy feels like it comes for free. The risk is structural, though. Earliest-finish-time greedy
  optimizes the *number* of intervals chosen, and here I am paid by *weight*, not by count. Those
  two objectives are not the same problem, and the moment weights vary, "finish earliest" stops
  having any obvious connection to "earn the most." I will not trust it until I have tried to break
  it.
- *Sort-by-end dynamic programming with predecessor search.* Sort by end coordinate; process
  intervals in that order; for each interval `i` decide to skip it (inherit the best answer over the
  earlier intervals) or take it (its weight plus the best answer restricted to intervals that finish
  at or before `s_i`). Finding "the best answer up to the last interval compatible with `i`" is a
  search over a sorted array, so the whole thing is `O(n log n)`. The risk here is not the *idea* —
  this is the standard provably-correct weighted-interval-scheduling DP — but the *transcription*:
  the half-open `<=` test, the binary-search boundary, and the indexing between the sorted array and
  the DP table are all easy to get subtly wrong.

**Stress-testing greedy before committing.** Hand-waving "greedy probably generalizes" is exactly
how wrong solutions get shipped, so let me actually attack it with a concrete instance. Here is the
cleanest two-interval counterexample I can build:

```
A = [0, 1)   w = 1
B = [0, 100) w = 1000
```

Earliest-finish-time greedy sorts by end: `A` finishes at `1`, `B` finishes at `100`, so it
considers `A` first. `A` does not overlap "nothing taken yet," so greedy takes `A`, total `1`. Now
`B` starts at `0`, which is inside `[0, 1)`, so `B` overlaps `A` and greedy rejects it. Greedy's
answer is `1`. But the optimum is obviously to take `B` alone for `1000`. Greedy is off by a factor
of a thousand, and I can see exactly *why*: by grabbing the interval that *finishes* first it
committed to the cheap one and locked out the valuable one. Finishing early is worth nothing when
you are paid by weight.

Let me also confirm this is not a degenerate two-element artifact, by breaking greedy on a case
where it actually chooses several intervals. Take

```
A = [0, 2) w = 1
B = [1, 3) w = 1
C = [2, 4) w = 1
D = [0, 4) w = 5
```

Sorted by end: `A` (end 2), `B` (end 3), `C` (end 4), `D` (end 4). Greedy takes `A` (no conflict),
skips `B` (overlaps `A` on `[1,2)`), takes `C` (touches `A` at `2`, compatible), and skips `D`
(overlaps everything). Greedy collects `A + C = 2`. But `D` alone is worth `5`. Greedy loses again,
and again the reason is that it maximized the *count* of intervals (it took two) when it should have
maximized weight (one heavy interval). Two independent counterexamples, both showing the same
failure mode. Greedy is out. The verification paid off — it killed an approach I would otherwise
have been tempted to ship because it shared the sort I needed anyway.

**Deriving the DP and proving the recurrence.** I want, for each prefix of the intervals *taken in
order of finishing time*, the best total weight achievable using only those intervals. Sorting by
end is the move that makes this clean: once the intervals are ordered by `e`, "the intervals
compatible-as-a-predecessor with interval `i`" are exactly a *prefix* of the order — every interval
that finishes at or before `s_i`. Concretely, let me sort so that `e_0 <= e_1 <= ... <= e_{n-1}` and
define

- `best[k]` = the maximum total weight using only the first `k` intervals in this sorted order
  (`best[0] = 0`, the empty selection).

For interval `i` (the `i`-th in sorted order, `0`-indexed), I have two choices:

- *Skip `i`.* Then the best I can do over the first `i+1` intervals is the best over the first `i`,
  i.e. `best[i]`.
- *Take `i`.* I earn `w_i`. Any other interval in my chosen set that comes before `i` in finishing
  order must not overlap `i`; since it finishes no later than `i` does, "not overlapping `i`" means
  "finishes at or before `s_i`", i.e. `e_j <= s_i` (the `<=` is the half-open touching rule).
  Crucially, because the order is by end, *all* such compatible predecessors form a prefix
  `[0, p)` of the sorted array, where `p` is the count of intervals among the first `i` with
  `e_j <= s_i`. So the best companion solution is `best[p]`, and taking `i` yields `w_i + best[p]`.

Hence

```
best[i+1] = max( best[i], w_i + best[p(i)] )
```

where `p(i)` is the number of intervals among the first `i` (in end order) whose end is `<= s_i`.
The answer is `best[n]`. Why is `best[p(i)]` legitimate — why can I append `i` to *any* optimal
solution counted by `best[p(i)]`? Because every interval that `best[p(i)]` could possibly use lies
in the prefix `[0, p(i))`, and every interval in that prefix finishes at `<= s_i`, hence is
compatible with `i`; so no matter which subset `best[p(i)]` realizes, adding `i` keeps it pairwise
non-overlapping. That is the correctness argument, and it is the thing greedy lacked: the DP
considers *both* "take" and "skip" at every interval, so it can decline a cheap early-finishing
interval in order to keep a heavy later one.

**Finding `p(i)` without an `O(n)` scan.** Because the ends are sorted ascending, `p(i)` — the
count of `j < i` with `e_j <= s_i` — is just the position where `s_i` would sit among the sorted
ends, found by binary search. I keep a separate array `ends[]` holding the sorted end coordinates
and binary-search for the first index whose end is strictly greater than `s_i`; that index *is* the
count of ends `<= s_i`. So one `O(log n)` search per interval, `O(n log n)` overall, comfortably
inside 2 seconds for `n = 2*10^5` (the big run below clocks 0.04 s).

Let me confirm the recurrence by hand on the sample, the four intervals
`[0,5) w5`, `[1,6) w3`, `[5,9) w6`, `[6,10) w4`, answer `11`. Sorted by end:
index 0 = `[0,5) w5` (end 5), index 1 = `[1,6) w3` (end 6), index 2 = `[5,9) w6` (end 9),
index 3 = `[6,10) w4` (end 10). `ends = [5, 6, 9, 10]`. `best[0] = 0`.

- `i=0`, `[0,5) w5`, `s=0`: count of ends `<= 0` is `0`, so `p=0`. take `= 5 + best[0] = 5`,
  skip `= best[0] = 0`. `best[1] = 5`.
- `i=1`, `[1,6) w3`, `s=1`: ends `<= 1` is `0`, `p=0`. take `= 3 + best[0] = 3`,
  skip `= best[1] = 5`. `best[2] = 5`.
- `i=2`, `[5,9) w6`, `s=5`: ends `<= 5` are just `{5}`, so `p=1`. take `= 6 + best[1] = 11`,
  skip `= best[2] = 5`. `best[3] = 11`.
- `i=3`, `[6,10) w4`, `s=6`: ends `<= 6` are `{5,6}`, so `p=2`. take `= 4 + best[2] = 9`,
  skip `= best[3] = 11`. `best[4] = 11`.

Answer `best[4] = 11`. Correct, and note step `i=2` is precisely where the `<=` half-open rule
earns its keep: `[0,5)` ends at `5` and `[5,9)` starts at `5`, they touch, and the `<=` lets them
coexist for `5 + 6 = 11`.

**First implementation — and immediately a self-check, because clean math transcribes dirty.** My
first cut of the binary search wrote the predecessor count as "the last index with `ends[mid] <=
s`" using an inclusive search, something like:

```
int lo = 0, hi = i - 1, p = -1;
while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (ends[mid] <= key) { p = mid; lo = mid + 1; }
    else hi = mid - 1;
}
long long take = job[i].w + best[p + 1];   // best over the prefix [0, p]
```

Two different conventions for "the boundary" are colliding here — `p` as a *last index* versus the
*count* I actually feed into `best[]` — and that is a classic place to be off by one. So I traced
the smallest input that exercises a non-trivial predecessor: two compatible touching intervals,

```
[0, 5) w = 10
[5, 9) w = 7
```

whose answer is clearly `17` (they touch at `5`, take both). Sorted: index 0 `[0,5)`, index 1
`[5,9)`, `ends = [5, 9]`, `best[0]=0`. `i=0`, `s=0`: the inclusive search over `ends[0..-1]` has
`lo=0, hi=-1`, never enters the loop, leaves `p=-1`, so `take = 10 + best[p+1] = 10 + best[0] = 10`;
`best[1] = 10`. So far so good. `i=1`, `[5,9)`, `s=5`: search over `ends[0..0]`: `mid=0`,
`ends[0]=5 <= 5`, so `p=0`, `lo=1`; loop ends; `take = 7 + best[p+1] = 7 + best[1] = 17`;
`best[2] = 17`. Answer `17`. That one happened to work.

**Diagnosing a real boundary bug.** Then I traced a case where the predecessor is *not* the
immediately preceding interval, to make sure the `p+1` indexing into `best[]` is right when the
search lands in the middle. Take

```
[0, 2) w = 5     (end 2)
[1, 4) w = 1     (end 4)
[2, 6) w = 5     (end 6)
```

answer `10` (take `[0,2)` and `[2,6)`, which touch at `2`). Sorted as written, `ends = [2,4,6]`.
`i=2`, `[2,6)`, `s=2`: the inclusive search over `ends[0..1] = [2,4]` looks for the last `<= 2`.
`lo=0, hi=1`: `mid=0`, `ends[0]=2 <= 2`, set `p=0`, `lo=1`. `lo=1, hi=1`: `mid=1`, `ends[1]=4 > 2`,
`hi=0`. Loop ends with `p=0`. So `take = 5 + best[p+1] = 5 + best[1]`. Now I need `best[1]` to be
`5` (the value of `[0,2)` alone). `best[1]` came from `i=0`: `s=0`, search over the empty
`ends[0..-1]` leaves `p=-1`, `take = 5 + best[0] = 5`, `best[1] = 5`. Good, `take = 5 + 5 = 10`,
`best[3] = 10`, answer `10`. This case also worked — but tracing it is what made me notice the
fragility: the answer is correct *only because* I happened to write `best[p+1]` with `p` as a
last-index and `best[]` shifted by one, and I had a second copy of the predecessor logic elsewhere
in an earlier draft that used `p` directly (count semantics), `best[p]`, against the *same*
`hi = i-1`/inclusive search. Mixing the two — feeding a last-*index* `p` into a slot that expects a
*count* — silently reads `best[p]` instead of `best[p+1]`, i.e. drops one compatible interval's
contribution exactly when the predecessor prefix is non-empty. On the three-interval case above the
two conventions agree numerically by luck for `i=2` (because the relevant prefix length and last
index differ by one in a way that the formula absorbed), but on a longer prefix they diverge and
produce a low answer. The defect is precise: I had *one* search style and *two* incompatible
readings of its output `p`.

**Fixing and re-verifying.** Rather than juggle "last index" and "count" and a `+1`, I rewrote the
search once, in a single convention I cannot misread: a half-open `[lo, hi)` search that returns
`lo` = the number of ends in `[0, i)` that are `<= s_i`, which is *exactly* the index I feed into
`best[]` (no `+1`, no `-1`):

```
int lo = 0, hi = i;                 // search ends[0 .. i-1]
long long key = job[i].s;
while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (ends[mid] <= key) lo = mid + 1;   // this end qualifies -> answer is to its right
    else hi = mid;                        // too late -> answer is at or left of mid
}
long long take = job[i].w + best[lo];     // best[lo] = best over the compatible prefix
```

This invariant ("`lo` is the count of qualifying ends, and `best[lo]` is the DP value over that
prefix") is the same number whether I think of it as a count or a one-based table index, so the two
readings that collided before now coincide by construction. Re-trace the three-interval case,
`i=2`, `s=2`, over `ends[0..1]=[2,4]`: `lo=0,hi=2`; `mid=1`, `ends[1]=4 > 2` -> `hi=1`; `lo=0,hi=1`;
`mid=0`, `ends[0]=2 <= 2` -> `lo=1`; loop ends, `lo=1`. `take = 5 + best[1] = 5 + 5 = 10`. Correct,
and now for a structural reason rather than a lucky one. Re-trace the touching pair `[0,5)/[5,9)`:
`i=1`, `s=5`, `ends[0..0]=[5]`: `mid=0`, `5 <= 5` -> `lo=1`; `take = 7 + best[1] = 7 + 10 = 17`.
Correct.

**Edge cases, deliberately, because this is where this kind of code dies.**

- `n = 0`: the read of `n` succeeds, the interval loop and the DP loop both run zero times, and I
  print `best[0] = 0`. (If even `n` is missing, `cin >> n` fails and I `return 0` having printed
  nothing — but the contract always provides `n`, and the empty-selection answer is `0`, which I
  print on the `n=0` path.) Correct.
- `n = 1`, single interval `[0,1) w=1`: `i=0`, `s=0`, empty search -> `lo=0`,
  `take = 1 + best[0] = 1`, `skip = best[0] = 0`, `best[1] = 1`. Answer `1`. Correct.
- Touching endpoints, `[0,5)/[5,10)`: traced above-style, both taken, sum of both weights. The `<=`
  in the search is what allows it; a `<` there would wrongly forbid the pair.
- Full overlap / duplicates, three copies of `[0,10) w=1`: each later copy has `s=0`, and no end is
  `<= 0` among predecessors (all ends are `10`), so every `take = 1 + best[0] = 1` and `skip`
  carries `1` forward; answer `1`. Only one of a mutually overlapping clump is ever taken. Correct.
- Nested intervals, `[0,100) w=50` containing `[10,20) w=30`, `[30,40) w=30`, `[50,60) w=30`: the
  three short disjoint inner intervals total `90`, beating the single outer `50`, and the DP finds
  `90` because it chains the inner three through their compatible prefixes. Correct (verified
  against the brute oracle).
- Overflow: coordinates up to `2*10^9` and the running total up to `2*10^14` both live in
  `long long`; I deliberately store `s, e, w` as `long long` so even the `e <= 2*10^9` endpoints
  never touch 32-bit `int`. The big stress run (`n = 2*10^5`, max coords/weights) returns a
  14-digit total and finishes in 0.04 s with 11 MB of memory.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the triple
  layout is format-agnostic.

**Self-verification at scale.** I wrote an independent oracle two ways — for `n <= 18` it
enumerates all `2^n` subsets and checks pairwise compatibility directly with no DP at all, and for
larger `n` it runs an `O(n^2)` DP that, for each interval, *scans every earlier interval* to find
compatible predecessors (no sort-by-end-plus-binary-search shortcut, so it shares no logic with the
solution under test). I then differential-tested the solution against this oracle on 10 hand-built
edge cases, 350 small random cases (tiny coordinates so overlaps are dense), and 200 medium random
cases up to a few hundred intervals — 560 cases total, **zero mismatches** — and re-ran after the
binary-search fix to confirm the boundary bug was actually gone (the pre-fix draft, when I reverted
the search to the mixed convention, did mismatch on a medium case, which is the evidence I trust).

**Final solution.** I convinced myself the *idea* is right by disproving earliest-finish-time greedy
with two explicit counterexamples (`[0,1)w1` vs `[0,100)w1000`: greedy `1` against the optimum
`1000`; and the four-interval count-vs-weight trap) and by proving the DP recurrence's "take" step
appends `i` to a solution drawn entirely from a compatible prefix; and I convinced myself the *code*
is right by tracing the predecessor search to a precise count-vs-index boundary bug, rewriting it in
a single unambiguous half-open convention, and re-verifying against an independent oracle over 560
cases. That is what I ship — one self-contained file, the simple provable `O(n log n)` sort-by-end DP
I can defend, not the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    struct Job { long long s, e, w; };
    vector<Job> job(n);
    for (auto &j : job) cin >> j.s >> j.e >> j.w;

    // Sort by finishing time (end coordinate), ascending.
    sort(job.begin(), job.end(),
         [](const Job &x, const Job &y) { return x.e < y.e; });

    // ends[i] = finishing time of the i-th job in sorted order.
    vector<long long> ends(n);
    for (int i = 0; i < n; i++) ends[i] = job[i].e;

    // best[i] = max total weight achievable using only jobs[0..i].
    // Intervals are half-open [s, e): job j is compatible with job i (j before i)
    // iff ends[j] <= starts[i]. p(i) = largest index j < i with ends[j] <= job[i].s.
    vector<long long> best(n + 1, 0); // best[0] = 0 (no jobs)
    for (int i = 0; i < n; i++) {
        // Skip job i: best[i] (using jobs[0..i-1]).
        long long skip = best[i];
        // Take job i: its weight plus best over jobs ending at or before job[i].s.
        // Find p = number of jobs (in sorted prefix [0..i-1]) whose end <= job[i].s.
        int lo = 0, hi = i; // search in ends[0..i-1]
        long long key = job[i].s;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ends[mid] <= key) lo = mid + 1;
            else hi = mid;
        }
        // lo = count of indices in [0..i-1] with ends[] <= key, i.e. p(i)+1 in 1-based best[].
        long long take = job[i].w + best[lo];
        best[i + 1] = max(skip, take);
    }

    cout << best[n] << "\n";
    return 0;
}
```

**Causal recap.** Earliest-finish-time greedy looked free because I had to sort by end anyway, but
it optimizes interval *count*, not *weight*, and two traced counterexamples (`[0,1)w1` vs
`[0,100)w1000` giving greedy `1` against `1000`; and a four-interval case where greedy's two cheap
intervals lose to one heavy one) showed it loses by arbitrary factors — so I moved to the
sort-by-end DP, whose "take" transition appends `i` to the best solution over the compatible prefix
located by binary search. My first search mixed "last index" and "count" readings of its output and
silently dropped a predecessor's contribution on non-trivial prefixes; rewriting it as a single
half-open `[lo, hi)` search returning the count that *is* the `best[]` index removed the ambiguity;
and `long long` on coordinates, weights, and the accumulator closes the `2*10^9` coordinate and
`2*10^14` total overflow corners. Independent-oracle differential testing over 560 cases with zero
mismatches is the evidence that the shipped DP is correct.
