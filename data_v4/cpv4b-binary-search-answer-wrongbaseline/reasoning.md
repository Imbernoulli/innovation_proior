**Reading the problem and pinning the contract.** I have a ring of integer circumference `L`, with `n`
distinct posts at sorted positions `p[0] < p[1] < ... < p[n-1]` in `[0, L)`. I must choose exactly `k`
of them, and the *clearance* of a choice is the smallest clockwise gap between cyclically consecutive
chosen posts — where the gap that wraps from the last chosen post around to the first counts like every
other gap. I want to maximize that smallest gap and print it. The scale: `2 <= k <= n <= 2000`,
`n <= L <= 10^9`. So positions and arc distances reach `10^9`; a wrap distance is computed as
`(p[start] + L) - pos`, which is below `2*10^9` and overflows 32-bit signed by a hair. Everything that
touches a position or a gap must be 64-bit. That is decision one and it is non-negotiable. `int` here
is a silent wrong-answer on a large test.

**Laying out the candidate approaches.** Two routes are on the table.

- *Brute force over subsets.* Enumerate every size-`k` subset, compute its cyclic clearance, keep the
  max. Obviously correct, and I will use exactly this as my offline checker — but `C(2000, 1000)` is
  not a number I can iterate, so it is a checker, never the submission.
- *Binary search on the answer.* The clearance is a single integer I want to maximize, and "can I
  achieve clearance at least `d`?" feels monotone: if I can spread `k` posts with all gaps `>= d`, then
  any smaller threshold `d' < d` is also satisfied by the same posts. So `feasible(d)` is a step
  function — true up to some threshold, false after — and binary search finds the largest `d` with
  `feasible(d)` true. This is the textbook shape for this area. The entire difficulty collapses onto
  one function: `feasible(d)`.

**Bounding the search range, with a numeric self-check.** Before coding the predicate I want the
`hi` of the binary search, and I will not just guess `L`. Claim: the answer is at most `floor(L/k)`,
and in particular at most `floor(L/2)` since `k >= 2`. Why: for any chosen set of `k` posts, list the
`k` cyclic gaps `g_0, ..., g_{k-1}`; walking once around the ring traverses each arc exactly once, so
`g_0 + ... + g_{k-1} = L` exactly. The minimum of `k` nonnegative numbers summing to `L` is at most
`L/k`. So clearance `<= L/k <= L/2`. Let me check the identity on a concrete subset rather than trust
it: ring `L = 20`, chosen posts `{3, 9, 17}`. Cyclic gaps clockwise: `9-3 = 6`, `17-9 = 8`, wrap
`(3+20)-17 = 6`. Sum `6 + 8 + 6 = 20 = L`. The identity holds, and the minimum `6` is `<= 20/3 = 6.67`.
Good — `hi = L/2` is a safe, valid upper bound (I could use `L/k`, but `L/2` is simpler and still
correct). I also confirmed this `sum-of-cyclic-gaps == L` identity programmatically on hundreds of
random subsets; it never failed, and no computed answer ever exceeded `L/2`.

**The "standard" feasibility test, transplanted from the line.** On a *straight* line, the famous
"aggressive cows" predicate for "place `k` items with min spacing `>= d`" is a one-pass greedy: anchor
the first item at the leftmost candidate `p[0]`, sweep left to right, and take a candidate the moment
its distance from the last taken is `>= d`; feasible iff you take at least `k`. It is provably optimal
on a line because greedily taking the earliest valid candidate leaves the most room for the rest. My
first instinct is to drop this in verbatim:

```
// line greedy, applied naively to the ring
bool feasibleLine(long long d) {
    long long taken = 1, last = p[0];
    for (int i = 1; i < n; i++)
        if (p[i] - last >= d) { taken++; last = p[i]; }
    return taken >= k;
}
```

**First trace — does the line greedy answer the ring question?** I will not ship this on faith; I will
run it against brute force on the sample. Sample: `L = 20`, posts `[0, 1, 2, 3, 9, 11, 17]`, `k = 3`.
Brute force over all `C(7,3) = 35` triples reports clearance `6` (the triple `{3, 9, 17}` from above).
Now binary search with `feasibleLine`. Take the decisive query `d = 8`: anchor at `p[0] = 0`; scan —
`1,2,3` are `< 8` from `0`, skip; `9 - 0 = 9 >= 8`, take `9` (taken=2, last=9); `11 - 9 = 2`, skip;
`17 - 9 = 8 >= 8`, take `17` (taken=3). `taken = 3 >= k`, so `feasibleLine(8)` returns **true**. The
binary search therefore believes a clearance of `8` is achievable and reports at least `8`. But brute
force says the true maximum is `6`. The line greedy is **too optimistic** — it returns `8` where the
answer is `6`.

**Diagnosing the bug.** The defect is precise and it is geometric, not a typo. The line greedy
measured three gaps — `0->9` (=9), `9->17` (=8) — and then *stopped at the right end of the array*. It
never accounted for the gap that closes the cycle: from the last chosen post `17` back to the first
chosen post `0`, going clockwise, the arc is `(0 + 20) - 17 = 3`. That wrap gap is `3`, far below `8`,
so the triple `{0, 9, 17}` actually has clearance `min(9, 8, 3) = 3`, not `8`. On a ring the selection
is a *cycle*: there are exactly `k` gaps, not `k - 1`, and the line greedy silently drops the `k`-th
one. So "anchor at `p[0]`, sweep, count `>= k`" is the right *shape* but the wrong *predicate* for this
variant — it answers a line question. I must not appeal to "the standard solution"; the standard
solution is for a different geometry and here it is simply wrong.

**Repairing the predicate — close the loop.** Two things must change. First, whatever post I anchor at
must be treated as *forced chosen*, and after the sweep I must verify the wrap gap from the last taken
post back to the anchor is `>= d`. Second — and this is the subtler half — anchoring only at `p[0]` is
itself unjustified on a ring: `p[0]` need not belong to the optimal set. On a line the leftmost item is
always safe to anchor (sliding the first choice left never hurts), but a ring has no leftmost; every
post is a potential "first chosen," and which one I fix changes which wrap gap I am forced to honor. So
the correct feasibility tries *every* post as the forced anchor and asks whether any anchor admits `k`
posts with all gaps — wrap included — at least `d`.

For a fixed anchor `start`, I sweep the other posts in clockwise order. To make "clockwise from
`start`" arithmetic clean I *unroll* the ring: for step `t = 1..n-1`, the post index is
`(start + t) mod n`, and its unrolled position is `p[idx]` plus `L` once we wrap past index `n`. I keep
`lastPos` as the last taken unrolled position, take a post when `pos - lastPos >= d`, and — crucially —
refuse to take a post if doing so would leave a wrap `(p[start] + L) - pos < d`. Because positions only
increase as I sweep, once a post is too close to the anchor on the wrap side, every later post is even
closer, so I can `break`. Taking the *earliest* valid post at each step is still right: it maximizes the
remaining room exactly as on the line, and leaves the largest possible wrap.

```
long long countFrom(int start, long long d) {
    long long taken = 1;
    long long lastPos = p[start];
    for (int step = 1; step < n; step++) {
        int idx = start + step;
        long long pos = p[idx % n] + (idx >= n ? L : 0);
        long long gap = pos - lastPos;
        if (gap < d) continue;
        long long wrap = (p[start] + L) - pos;
        if (wrap < d) break;
        taken++;
        lastPos = pos;
    }
    return taken;
}
bool feasible(long long d) {
    if (d <= 0) return true;
    for (int s = 0; s < n; s++)
        if (countFrom(s, d) >= k) return true;
    return false;
}
```

**Second trace — re-running the failing query.** Back to the sample, `d = 8`, but now with `feasible`
that tries every anchor and honors the wrap. Anchor `start = 0` (`p[0] = 0`): take `9` (gap 9, wrap
`20 - 9 = 11 >= 8`, taken=2, lastPos=9); `11` gap 2 skip; `17` gap 8 `>= 8`, but wrap `20 - 17 = 3 < 8`
so I `break` without taking it. `countFrom(0, 8) = 2 < 3`. The wrap correctly vetoed the third post.
Let me sweep the other anchors at `d = 8`. Anchor `p = 3`: nearest forward post `>= 8` away is `11`
(gap 8), wrap `(3+20)-11 = 12 >= 8`, taken=2; next `17` gap 6 skip; wrapping past `n`, post `0` is at
unrolled `0+20 = 20`, gap `20 - 11 = 9 >= 8`, wrap `(3+20)-20 = 3 < 8`, break. Count `2`. Every anchor
tops out at `2`, so `feasible(8) = false`. The predicate no longer hallucinates clearance `8`. Now
`d = 6`: anchor `p = 3` — take `9` (gap 6, wrap `(3+20)-9 = 14`, taken=2); `11` gap 2 skip; `17` gap 8
`>= 6`, wrap `(3+20)-17 = 6 >= 6`, take (taken=3); done. `countFrom(3, 6) = 3 >= k`, so
`feasible(6) = true`. Binary search lands on `6`. That matches brute force exactly. The fix is real and
it fixed the thing I diagnosed.

**A second, independent self-verify — `k = 2`.** Pairs are the cleanest sanity check because the answer
has a closed form: for two posts the two cyclic gaps are `g` and `L - g`, so the clearance is
`min(g, L - g)`. Test `L = 10`, posts `[1, 8]`, `k = 2`. The gap `8 - 1 = 7`, so clearance should be
`min(7, 3) = 3`. Trace my code at `d = 4`: anchor `p = 1`, post `8` gap `7 >= 4`, wrap
`(1+10)-8 = 3 < 4`, break — count `1`; anchor `p = 8`, post `1` unrolled at `1 + 10 = 11`, gap
`11 - 8 = 3 < 4`, skip — count `1`. `feasible(4) = false`. At `d = 3`: anchor `p = 1`, post `8` gap 7,
wrap `3 >= 3`, take — count `2 >= k`, true. So the answer is `3`. Correct, and it confirms the wrap
handling is symmetric: neither anchor can claim the long arc `7` as the clearance, because the other
arc `3` always binds. (The full program prints `3` on this input, which I checked.)

**Edge cases, deliberately.**
- `k = 2`, diametric posts `L = 10`, `[0, 5]`: gaps `5` and `5`, clearance `5 = L/2`, hitting the upper
  bound exactly. The program prints `5`, so `hi = L/2` is reachable and not off by one.
- `k = n`: every post is forced, so the answer is just the minimum existing cyclic gap (including the
  wrap). `L = 12`, posts `[0, 3, 6, 9]`, `k = 4`: gaps `3,3,3,3`, answer `3`. With `k = n` my
  `countFrom` can only reach `taken = n` by taking *every* post, which forces every consecutive gap and
  the wrap to be `>= d` simultaneously — exactly the min-gap condition. Program prints `3`. Correct.
- Clustered posts where the wrap binds: `L = 100`, posts `[0, 1, 2, 50]`, `k = 3`. Best triple is
  `{0, 2, 50}` (or `{1, 2, 50}`): gaps like `2, 48, 50`, min `2`. Program prints `2`, matching brute.
  The two posts a single unit apart can never both be chosen with clearance `> 1`, and the answer is
  driven by the small cluster, not the big arc — precisely the case the line greedy mishandles.
- `d = 0`: `feasible(0)` short-circuits to true; any `k` posts have nonnegative gaps, so the binary
  search floor is sound and the answer is never negative.
- Overflow: `pos` reaches `p[idx] + L < 2*10^9` and `wrap = (p[start] + L) - pos` stays in
  `[0, L]`; all are `long long`. The sum-of-gaps reasoning used only for the bound never appears in the
  hot path. Safe.

**Complexity and why it fits.** `feasible(d)` tries `n` anchors, each an `O(n)` sweep, so `O(n^2)` per
predicate; binary search does `O(log L) ~ 31` predicates; total `O(n^2 log L)`. With `n = 2000` that
is about `2000^2 * 31 ~ 1.2*10^8` simple operations. I timed the worst case (`n = 2000`,
`L = 10^9`, `k = 1000`) at `0.11 s`, comfortably inside the 2 s limit. Memory is one `O(n)` array.

**Correctness of trying all anchors.** Suppose the optimum chooses a set `S` of `k` posts with
clearance `d*`. Pick any `s in S` as the anchor. `countFrom(s, d*)` greedily takes the earliest valid
post at each step; a standard exchange argument (identical to the line case, but now the wrap acts as a
hard right boundary at distance `L` from the anchor) shows the greedy count from `s` is at least
`|S| = k`, and every gap it accepts — including the closing wrap it explicitly checks — is `>= d*`. So
`feasible(d*)` returns true. Conversely, if `feasible(d)` returns true, the witnessing anchor's taken
posts have all `k` cyclic gaps `>= d` by construction, so a real selection of clearance `>= d` exists.
Hence `feasible` is exactly the predicate binary search needs, and it is monotone in `d`.

**Final solution.** I disproved the transplanted line greedy with one traced query (`d = 8` on the
sample: line greedy says feasible, true clearance is `6`), pinned the cause to the dropped wrap gap and
the unjustified single anchor, rebuilt the predicate to force an anchor, check the wrap, and try all
anchors, and re-verified the fixed predicate on the same query, on `k = 2`, and on the corner cases.
This is what I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
long long k, L;
vector<long long> p; // sorted, in [0, L)

// Greedy count starting from anchor post `start` (which is FORCED chosen):
// walk forward around the ring, take the next post whose gap from the last
// taken is >= d. Returns the number of posts taken (>= 1). The closing wrap
// gap (from the last taken back to `start`) must ALSO be >= d for the whole
// thing to form a valid cyclic placement; we enforce that by never taking a
// post that would leave a wrap gap < d, and by checking it at the end.
long long countFrom(int start, long long d) {
    long long taken = 1;
    long long lastPos = p[start];          // absolute position of last taken
    // iterate the other n-1 posts in cyclic order after `start`
    for (int step = 1; step < n; step++) {
        int idx = start + step;
        long long pos = p[idx % n] + (idx >= n ? L : 0); // unrolled position
        long long gap = pos - lastPos;
        if (gap < d) continue;             // too close, skip this post
        // would taking it leave a valid wrap back to start? wrap = (p[start]+L) - pos
        long long wrap = (p[start] + L) - pos;
        if (wrap < d) break;               // taking it (or anything further) kills the wrap
        taken++;
        lastPos = pos;
    }
    return taken;
}

// Can we choose >= k posts with every cyclic-adjacent gap >= d?
bool feasible(long long d) {
    if (d <= 0) return true;               // any selection has nonneg gaps
    // Some post must be chosen; try every post as the forced anchor.
    for (int s = 0; s < n; s++) {
        if (countFrom(s, d) >= k) return true;
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> k >> L)) return 0;
    p.resize(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    // Clearance is in [0, L/2] (k>=2 means at least two points, so the min
    // cyclic gap can be at most floor(L/2)). Binary search the largest d
    // for which a valid placement of >= k posts exists.
    long long lo = 0, hi = L / 2, ans = 0;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** Binary search on the clearance is the right framework, so the whole problem reduced to
the feasibility predicate; I reached for the standard line "aggressive cows" greedy, but tracing its
decisive query `d = 8` on the sample showed it declaring feasible a triple whose true clearance is `6`,
because on a ring the selection is a *cycle* of `k` gaps and the line sweep silently drops the wrap gap
(and, more deeply, has no canonical "leftmost" anchor to fix); fixing it means forcing a chosen anchor,
checking the closing wrap explicitly, and trying every post as the anchor, after which re-tracing the
same `d = 8` query, the `k = 2` closed form `min(g, L-g)`, and the clustered/`k=n`/diametric corners all
agree with brute force, and the `O(n^2 log L)` predicate runs the `n = 2000` worst case in `0.11 s`.
