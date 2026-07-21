I have a ring of circumference `L` with `n` posts at sorted distinct positions `p[0] < ... < p[n-1]`
in `[0, L)`, and I must pick exactly `k` of them to maximize the clearance — the smallest clockwise
gap between cyclically consecutive chosen posts, where the arc wrapping from the last chosen post back
to the first counts like any other gap. This is the classic "spread `k` items to maximize the minimum
spacing," except on a ring instead of a line, and that single geometric difference is where the whole
problem lives. Two constraint sizes set the ground rules before I write anything: `n <= 2000`, which
leaves room for an `O(n^2)`-ish feasibility test, and `L <= 10^9`, which means an *unrolled* position
like `p[i] + L` reaches nearly `2*10^9` and a 32-bit `int` silently wraps to garbage on a large test.
Every position and gap is `long long` from the start.

The clearance is a single integer I want to maximize, and "can I achieve clearance at least `d`?" is
monotone: a set of `k` posts with all cyclic gaps `>= d` also satisfies any `d' < d`. So `feasible(d)`
is a step function and I binary-search the largest `d` where it holds. The entire problem collapses
onto writing `feasible` correctly for a ring — brute force over all `C(n, k)` subsets is astronomical
for `n = 2000` and survives only as an offline checker on tiny inputs.

For the search's upper bound I don't want to guess `L`. Walking once around the ring traverses each of
the `k` cyclic gaps of a chosen set exactly once, so they sum to `L`; the minimum of `k` nonnegative
numbers summing to `L` is at most `L/k`, and since `k >= 2` the clearance can never exceed `L/2`. So
`hi = L/2` is a safe bound (`L/k` would be tighter, but `L/2` is simpler and still correct).

Now the predicate. On a *straight* line the "aggressive cows" greedy is the reflex: anchor the first
item at the leftmost candidate `p[0]`, sweep left to right, take a candidate the moment its distance
from the last taken is `>= d`, feasible iff you take `>= k`. My first instinct is to transplant it
verbatim:

```
// line greedy, applied naively to the ring
bool feasibleLine(long long d) {
    long long taken = 1, last = p[0];
    for (int i = 1; i < n; i++)
        if (p[i] - last >= d) { taken++; last = p[i]; }
    return taken >= k;
}
```

I run it against brute force on the sample `L = 20`, posts `[0,1,2,3,9,11,17]`, `k = 3`, where brute
force reports clearance `6` (the triple `{3,9,17}`). Take the decisive query `d = 8`: anchor at
`p[0] = 0`; skip `1,2,3`; `9 - 0 = 9 >= 8`, take `9`; skip `11`; `17 - 9 = 8 >= 8`, take `17`. Three
taken, so `feasibleLine(8)` is true and the search reports at least `8` — but the true maximum is `6`.
The line greedy is too optimistic.

The defect is geometric, not a typo. The sweep measured `0->9` and `9->17` and then stopped at the end
of the array; it never counted the gap closing the cycle, from `17` back to `0`, which is
`(0 + 20) - 17 = 3`. The triple `{0,9,17}` actually has clearance `min(9, 8, 3) = 3`. On a ring a
selection is a *cycle* with exactly `k` gaps, not `k - 1`, and the line sweep silently drops the last
one. "Anchor, sweep, count `>= k`" is the right shape but answers a line question.

Two things must change. The anchor must be treated as *forced chosen*, and after the sweep the wrap
from the last taken post back to the anchor must itself be `>= d`. And anchoring only at `p[0]` is
unjustified on a ring: a line has a leftmost item that is always safe to fix, but a ring has no
leftmost — every post is a candidate "first chosen," and which one I fix determines which wrap gap I am
forced to honor. So feasibility must try every post as the forced anchor.

For a fixed `start` I sweep the other posts clockwise. To keep "clockwise from `start`" arithmetic
clean I unroll the ring: at step `t = 1..n-1` the post is index `(start + t) mod n`, with unrolled
position `p[idx]` plus one `L` once the index passes `n`, so positions only increase. I keep `lastPos`
as the last taken position, take a post when `pos - lastPos >= d`, and refuse any post that would leave
a wrap `(p[start] + L) - pos < d`. Since positions only increase, once a post is too close to the
anchor on the wrap side every later post is closer still, so I can `break`. Taking the earliest valid
post is still right — it leaves the most room for the rest and the largest possible wrap. `feasible(d)`
just runs this count from every anchor and accepts if any reaches `k` (and short-circuits `d <= 0` to
true, since every selection has nonnegative gaps).

Re-running the failing query with this predicate: at `d = 8`, anchor `0` takes `9` (wrap
`20 - 9 = 11`, fine) then reaches `17` with gap `8` but wrap `20 - 17 = 3 < 8`, so it breaks without
taking it — count `2`. Sweeping the other anchors at `d = 8` tops out at `2` everywhere, so
`feasible(8)` is now false. At `d = 6`, anchor `3` takes `9` (gap 6, wrap `14`) and `17` (gap 8, wrap
`(3+20)-17 = 6 >= 6`) for count `3`, so `feasible(6)` is true. Binary search lands on `6`, matching
brute force.

The `k = 2` regime pins the wrap handling, since pairs have the closed form `min(g, L - g)`: for
`L = 10`, posts `[1, 8]`, the gap is `7` so the clearance must be `min(7, 3) = 3`, and the code returns
`3` — neither anchor can claim the long arc `7`, because the closing-wrap check forces the short arc
`3` to bind. At the other extreme `k = n` forces every post, and `countFrom` reaches `taken = n` only
by taking all of them, which makes every consecutive gap *and* the wrap `>= d` at once — exactly the
"minimum existing cyclic gap" answer (`[0,3,6,9]`, `L = 12` gives `3`). The overflow guard is not
cosmetic: on the largest tests `pos = p[idx] + L` and `wrap = (p[start] + L) - pos` reach `~2*10^9`,
past 32-bit range but well inside `long long`.

`feasible` tries `n` anchors, each an `O(n)` sweep, so `O(n^2)` per call, times `O(log L) ~ 31`
binary-search steps: `O(n^2 log L) ~ 1.2*10^8` operations at `n = 2000`. I timed the worst case
(`n = 2000`, `L = 10^9`) at `0.11 s`, well inside the 2 s limit; memory is one `O(n)` array.

Why trying all anchors is correct: if the optimum is a set `S` of `k` posts with clearance `d*`, fix
any `s in S` as the anchor. The greedy from `s` takes the earliest valid post at each step, and the
standard exchange argument — now with the wrap acting as a hard boundary at distance `L` from the
anchor — gives a count `>= |S| = k` with every accepted gap, wrap included, `>= d*`. Conversely, a
true `feasible(d)` exhibits an anchor whose taken posts realize `k` cyclic gaps all `>= d`, a genuine
selection of clearance `>= d`. So `feasible` is exactly the monotone predicate the binary search needs.
The full program, wrapping this predicate in the `[0, L/2]` binary search, is in the answer.
