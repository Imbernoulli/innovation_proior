Two details in this weighted-interval problem decide correctness before any algorithm does, and
both are quiet enough to slip a wrong submission through. The segments are half-open: `[s_i, e_i)`
occupies every `x` with `s_i <= x < e_i`, so two intervals that merely touch — one ending exactly
where the next begins, `e_i = s_j` — do **not** overlap and may both be chosen. That single fact
fixes whether my compatibility test is `<=` or `<`, and flipping it is a silent wrong answer on
every touching case. The second is scale: `n` up to `2*10^5` with weights up to `10^9`, so a total
can reach `2*10^5 * 10^9 = 2*10^14`. That is five orders of magnitude past the signed-32-bit ceiling
of about `2.147*10^9`, so the accumulator and anything summed into it must be 64-bit. The
coordinates themselves top out at `2*10^9`, which actually still sits just *under* that `2.147*10^9`
ceiling — they would fit in `int` — but I store `s, e, w` uniformly as `long long` so no mixed-type
comparison or sum can surprise me, and the weight total needs the width regardless.

Two routes fit an `O(n log n)` budget. Earliest-finish-time greedy — sort by end, sweep, take any
interval that does not conflict with the last one taken — is provably optimal for the *unweighted*
activity-selection problem, and it is tempting here because I have to sort by end for the
alternative anyway, so it feels free. But it maximizes the *count* of intervals chosen, and here I
am paid by weight. Before trusting that those two objectives coincide, I try to break it.

The cleanest break is two intervals: `A = [0,1) w=1` and `B = [0,100) w=1000`. Sorted by end,
greedy meets `A` first, takes it (total `1`), then finds `B` starting at `0` inside `[0,1)` and
rejects it — answer `1`, against the obvious optimum of `B` alone for `1000`. Off by a factor of a
thousand, because grabbing the earliest *finisher* locked in the cheap interval. To be sure this is
not a two-element fluke, a case where greedy actually selects several: `A=[0,2) w1`, `B=[1,3) w1`,
`C=[2,4) w1`, `D=[0,4) w5`. Sorted by end, greedy takes `A`, skips `B` (overlaps `A`), takes `C`
(touches `A` at `2`, compatible), skips `D` — collecting `A+C = 2` when `D` alone is worth `5`. Same
failure: it maximized count over weight. Greedy is out.

So I need a rule that can *decline* a cheap early finisher to keep a heavy later one. Sort by end,
`e_0 <= e_1 <= ... <= e_{n-1}`, and process in that order. The payoff of sorting by end is that the
predecessors compatible with interval `i` — those finishing at or before `s_i` — form a *prefix* of
the sorted order. Let `best[k]` be the maximum weight using only the first `k` intervals, with
`best[0] = 0`. For interval `i` I either skip it, inheriting `best[i]`, or take it for `w_i` plus
the best solution restricted to intervals finishing `<= s_i`. Because everything finishing `<= s_i`
is a prefix `[0, p(i))`, where `p(i)` counts the earlier intervals with `e_j <= s_i` (the `<=` is
the half-open rule), that restricted best is exactly `best[p(i)]`, and appending `i` to whatever
subset realizes it stays pairwise non-overlapping — every interval it could use finishes `<= s_i`,
hence is compatible with `i`. So

```
best[i+1] = max( best[i], w_i + best[p(i)] )
```

and the answer is `best[n]`. This is the guarantee greedy lacked: "take" and "skip" are both on the
table at every step, so the DP can pass on a cheap interval to preserve a heavier compatible one.
Since the ends are sorted ascending, `p(i)` is a binary search — the number of ends `<= s_i` — for
one `O(log n)` search per interval and `O(n log n)` overall, comfortably inside the 2-second limit.

The one place this transcribes badly is that search boundary. There are two natural things to
compute — the last *index* `j` with `ends[j] <= s_i`, or the *count* of such ends — and they differ
by one, while `best[]` is indexed by prefix length. Feed a last-index into a slot that wants a count
and I silently read `best[p]` instead of `best[p+1]`, dropping one predecessor's contribution
exactly when the compatible prefix is non-empty; short traces can even mask it, since on a
length-one prefix the two conventions happen to agree. I dodge the whole trap by writing the search
in a single convention that cannot be misread — a half-open `[lo, hi)` search returning `lo` = the
count of qualifying ends, which *is* the index into `best[]`, no `+1` or `-1` fixup:

```cpp
int lo = 0, hi = i;                 // search ends[0 .. i-1]
long long key = job[i].s;
while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (ends[mid] <= key) lo = mid + 1;   // this end qualifies -> answer lies to its right
    else hi = mid;                        // too late -> answer at or left of mid
}
long long take = job[i].w + best[lo];     // best[lo] = best over the compatible prefix
```

On the sample `[0,5)w5, [1,6)w3, [5,9)w6, [6,10)w4` (ends `[5,6,9,10]` after the sort) this gives
`best = [0, 5, 5, 11, 11]`. The decisive step is `i=2`, `[5,9)` with `s=5`: the search counts the
one end `<= 5` (the `5` from `[0,5)`), so `take = 6 + best[1] = 11` beats `skip = best[2] = 5`. That
is exactly where the half-open `<=` earns its keep — `[0,5)` ends at `5`, `[5,9)` starts at `5`,
they touch, and the `<=` lets them coexist for `5 + 6 = 11`. A `<` there would forbid the pair and
report the wrong answer. Final `best[4] = 11`, matching the expected output.

A few corners the recurrence still has to survive. `n = 0`: the read of `n` succeeds, both loops run
zero times, and I print `best[0] = 0`. The nested case is the interesting one — `[0,100) w50`
around disjoint `[10,20) w30`, `[30,40) w30`, `[50,60) w30` — where the DP must prefer the inner
chain (`90`) to the single outer interval (`50`); it does, chaining the three inner intervals
through their compatible prefixes. Mutually overlapping duplicates (three copies of `[0,10) w1`)
each find no predecessor ending `<= 0`, so every "take" is `1 + best[0]` and only one copy is ever
counted. And the overflow corner lands where predicted: the max-scale run returns a 14-digit total
and finishes in 0.04 s in 11 MB, so the `long long` accumulator was necessary and the width is
otherwise free.

For evidence beyond hand-traces I built an independent oracle — exhaustive over all `2^n` subsets
for `n <= 18`, and an `O(n^2)` scan-every-earlier-interval DP for larger `n`, sharing none of the
sort-by-end-plus-binary-search logic with the solution — and differential-tested against it over 10
hand-built edge cases, 350 dense small-coordinate randoms, and 200 medium randoms up to a few
hundred intervals: 560 cases, zero mismatches. Reverting the predecessor search to the mixed
index/count convention does produce a mismatch on a medium case, which is the concrete evidence that
the clean half-open convention is the one that matters, not just cosmetics.

What I ship is the one file I can defend: read `n` and the `n` triples from stdin, sort by end,
run the sort-by-end DP with `long long` throughout and the predecessor found by that single
half-open search, and print `best[n]`. The full program is in the answer.
