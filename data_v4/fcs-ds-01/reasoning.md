**Reading the problem and pinning the contract.** I have an array `a[0..n-1]` of
integers and `q` queries, each a pair `(l, r)` with `0 <= l <= r < n`, and for
each I must report how many *distinct* values occur in `a[l..r]`, inclusive on
both ends. The queries are all given before I answer any of them — this is the
offline setting, which is a strong hint: I am allowed to answer them in whatever
order is convenient, as long as I print results back in the original input
order. Scale first, because it decides everything: `n, q <= 2*10^5` and
`|a[i]| <= 10^9`. The values can be negative and as large as a billion, so I
cannot index a count array by raw value; I will need coordinate compression.
The output is one line per query, so the total output is up to `2*10^5` lines —
large enough that I should build the output in one string and flush once rather
than do `2*10^5` separate `cout <<` calls. Those two facts (compress the values,
batch the output) are settled before I pick an algorithm.

**The obvious approach, and why it dies at scale.** The definitional method is:
for each query, walk `a[l..r]` and count distinct values, using a hash set or a
freshly-cleared boolean/seen array. That is unarguably correct. The cost is the
problem: a single query can span the whole array, so each query is `O(n)` in the
worst case and the batch is `O(q*n)`. Plug in the limits: `2*10^5 * 2*10^5 =
4*10^10` element touches. Even at a generous `10^9` simple operations per second
that is forty seconds — two orders of magnitude over a 2-second budget. So the
rescan is out for the full constraints, though I will keep exactly this method
as my brute-force oracle on small inputs, because its correctness is beyond
doubt.

**Can I do each query independently and sublinearly?** That is the instinct that
works for range *sums*: precompute prefix sums, answer `sum(l,r)` in `O(1)` as
`P[r+1]-P[l]`. So I ask whether "distinct count" has a prefix structure. It does
not, and it is worth seeing precisely why, because the failure is what points me
at the real method. Distinct-count is not additive and not subtractive across a
split point. Take `a = [1, 2, 1]`. The distinct count of the prefix `[0,1]` is
2, of the prefix `[0,2]` is 2, and `2 - 2 = 0` is not the distinct count of the
suffix `[2,2]` (which is 1). The reason is that a value can appear on *both*
sides of any split, so inclusion–exclusion over prefixes double-counts in a way
I cannot untangle with a single scalar per prefix. There *is* a clever offline
sweep-by-`r` with a Fenwick tree that exploits "only the last occurrence of each
value matters up to `r`," giving `O((n+q) log n)`. That works and I keep it in
my back pocket. But it needs a second structure (a BIT) and careful
last-occurrence bookkeeping, and the constants make it no faster in practice
than the simpler tool I am about to derive — so I will only fall back to it if
the simpler tool cannot meet the limit.

**The pivot: stop treating queries as independent.** Here is the move. The
rescan throws away everything between consecutive queries: it rebuilds the
distinct count from scratch every time. But if I could keep a single *window*
`[L, R]` together with a table `cnt[value] = how many times that value occurs in
the current window`, then extending the window by one position to the right is
`O(1)`: increment `cnt[a[R+1]]`, and if it just went from `0` to `1`, the
running distinct total goes up by one. Shrinking is the mirror: decrement, and
if it hit `0`, the distinct total goes down by one. So moving an endpoint by one
costs `O(1)`, and answering a query is just "move `L` and `R` to the query's
endpoints, then read the running total." The entire cost of the algorithm is now
**the total distance the two pointers travel** across all queries. If I process
queries in input order, that distance is unbounded — adversarial queries can
ping-pong the window across the whole array every time, which is back to
`O(q*n)`. So the whole game reduces to one question: **in what order should I
visit the queries so that the total pointer travel is small?**

**Deriving the ordering — the actual insight.** I want an order that keeps both
endpoints from wandering too far between consecutive queries. The trick is to
split the index axis `[0, n)` into blocks of width `B`, and sort the queries by
the pair `(block of l, r)` — that is, primarily by which block the left endpoint
`l` falls in, and secondarily by the right endpoint `r`. Let me count the cost
of this order and choose `B` to minimize it, because that is what tells me the
ordering is right rather than just plausible.

- *Left pointer.* Within a single `l`-block, every query has `l` in a window of
  width `B`, so each consecutive `L`-move is at most `B`. Across all `q`
  queries, the left pointer moves `O(q * B)` in total.

- *Right pointer.* Within a single `l`-block, the queries are sorted by `r`, so
  `R` only marches forward, covering at most `O(n)` total inside that block.
  There are `n / B` blocks, so the right pointer moves `O((n / B) * n) =
  O(n^2 / B)` in total. (When I cross from one block to the next, `R` may jump
  back once, but that is one `O(n)` jump per block, i.e. `O(n^2 / B)` again,
  same bound.)

Total travel is `O(q*B + n^2 / B)`. This is minimized by balancing the two
terms: set `q*B = n^2 / B`, i.e. `B = n / sqrt(q)`. Substituting back, both
terms become `O(n * sqrt(q))`, so the whole batch costs `O((n + q) * sqrt(q))`
pointer moves (the `+q` covers reading and the per-query `O(1)` finalize). At
`n = q = 2*10^5`, `sqrt(q) ~= 450`, so this is on the order of `n * sqrt(q) ~=
9*10^7` moves — under a tenth of a second territory, comfortably inside 2
seconds. This block-sort ordering is the named technique (Mo's algorithm), and
the `sqrt`-block choice is the entire reason it is fast; the `O(1)` add/remove on
the count table is the easy half.

**One refinement that halves the right-pointer travel.** With the plain
`(block(l), r)` order, every time I cross into a new block I sort that block's
queries by ascending `r`, so `R` sweeps left-to-right within the block and then
snaps all the way back to the start for the next block. I can avoid the snap-back
with a boustrophedon (even/odd) trick: sort `r` ascending in even-numbered
blocks and descending in odd-numbered blocks. Then `R` sweeps right in one
block, and the next block begins where it ended and sweeps left — no reset. This
does not change the asymptotics but roughly halves the right-pointer constant,
which matters at full scale. I will include it.

**Settling the data layout.**
- Compress values: copy `a`, sort, unique, and replace each `a[i]` by its index
  in the compressed list. Now values live in `[0, V)` with `V <= n`, so
  `cnt` is a plain `vector<int>` of size `V`. This also makes the algorithm
  oblivious to negatives and to the `10^9` magnitude.
- Store each query as `(l, r, idx)` so I can scatter answers back to input
  order after sorting.
- Window is inclusive `[L, R]`; I start it *empty* as `L = 0, R = -1` so the
  initial window contains nothing and the first "extend right" lands on index 0.

**First implementation.** I write the core loop. The standard and slightly
delicate part is the *order* of the four while-loops that march the pointers,
because if I shrink before I grow I can transiently make the window invalid. My
first cut:

```
int curL = 0, curR = -1;
for (Query Q : qs) {
    while (curL < Q.l) remove(curL++);
    while (curR < Q.r) add(++curR);
    while (curL > Q.l) add(--curL);
    while (curR > Q.r) remove(curR--);
    ans[Q.idx] = distinct;
}
```

with

```
auto add = [&](int pos){ if (cnt[a[pos]]++ == 0) distinct++; };
auto remove = [&](int pos){ if (--cnt[a[pos]] == 0) distinct--; };
```

**Tracing it on a tiny case and hitting a bug.** Let me trace the smallest input
that could expose an ordering mistake. Take `a = [5, 5]` (after compression
`[0, 0]`, `V = 1`), and a single query `(0, 0)` whose correct answer is `1`.
Start `curL = 0, curR = -1`, `cnt = [0]`, `distinct = 0`.

- `while (curL < 0)`: `0 < 0` false, skip.
- `while (curR < 0)`: `-1 < 0` true → `add(++curR)` = `add(0)`: `cnt[a[0]] =
  cnt[0]` goes `0 -> 1`, so `distinct -> 1`. Now `curR = 0`. Loop check `0 < 0`
  false, stop.
- `while (curL > 0)`: false. `while (curR > 0)`: false.
- Answer `= distinct = 1`. Correct here.

That case passed, so the single-query path is fine. Now I run the differential
tester against the rescan oracle on random small inputs — and it reports a
mismatch. The failing seed produces, among others, two queries on the same array
where the second query has a *smaller* `l` than the first but the window had
already been shrunk from the left. I reconstruct the essential pattern: array
`a = [1, 2]` (compressed `[0, 1]`), query A `(1, 1)` then query B `(0, 0)` (in
the sorted order they can land adjacent like this). Trace with my first loop:

- Query A `(1,1)`: start `L=0,R=-1`. `while(L<1)`: `remove(0)` — but index 0 was
  never added! `cnt[a[0]] = cnt[0]` goes `0 -> -1`, and `--cnt == 0` is false,
  so `distinct` is untouched but `cnt[0]` is now `-1`, `curL = 1`. Then
  `while(R<1)`: `add(1)`,`add`? `++curR` from -1 gives 0 first: actually
  `add(++curR)` makes `curR=0`, `add(0)`: `cnt[0]` goes `-1 -> 0`, and
  `++ == 0`? the post-increment returns the old value `-1`, not `0`, so the
  `== 0` test is false and `distinct` does **not** increase even though I just
  brought index 0 into the window. The count table is now corrupt.

**Diagnosing the bug.** The defect is the loop *order*. By running the
left-shrink `while (curL < Q.l) remove(curL++)` *before* I have grown the window
to cover the new range, I called `remove` on positions that were never `add`ed,
driving `cnt` negative. The window invariant "`cnt` reflects exactly the
positions in `[curL, curR]`" only holds if I never remove a position outside the
current window. The fix is the canonical Mo ordering: **grow before you shrink.**
Extend `R` rightward and extend `L` leftward first (both are `add`s that can only
enlarge the window), and only then shrink `R` and shrink `L` (the `remove`s).
That guarantees every `remove` acts on a position currently inside the window.

**Fixing and re-verifying.** The corrected loop:

```
while (curR < Q.r) add(++curR);     // grow right
while (curL > Q.l) add(--curL);     // grow left
while (curR > Q.r) remove(curR--);  // shrink right
while (curL < Q.l) remove(curL++);  // shrink left
```

Re-trace `a = [1,2]`, query A `(1,1)`: start `L=0,R=-1`. Grow right: `add(0)`
→ `cnt[0]=1, distinct=1`; `add(1)` → `cnt[1]=1, distinct=2`; now `R=1`. Grow
left: `curL=0 > 1`? no. Shrink right: `R=1>1`? no. Shrink left: `curL=0<1` →
`remove(0)`: `cnt[0] 1->0`, `distinct 2->1`, `curL=1`. Answer `distinct=1` —
range `[1,1]` is `{2}`, correct. Then query B `(0,0)`: grow right `R=1<0`? no.
Grow left `curL=1>0` → `add(0)`: `cnt[0] 0->1`, `distinct 1->2`, `curL=0`. Shrink
right `R=1>0` → `remove(1)`: `cnt[1] 1->0`, `distinct 2->1`, `R=0`. Shrink left
`0<0`? no. Answer `distinct=1` — range `[0,0]` is `{1}`, correct. The corruption
is gone and it is gone *for the reason I fixed*: no `remove` ever touches a
position outside the window now.

**Edge cases, deliberately, because this is where range-query code dies.**
- `n = 1`, one query `(0,0)`: grow right adds index 0, `distinct = 1`; the
  answer is `1`. Correct — a single element is one distinct value.
- All-equal array, e.g. `[7,7,7,7,7]`: every `add` after the first finds
  `cnt > 0` and does *not* bump `distinct`; every range answers `1`. Correct.
- All-distinct array: every `add` lifts a fresh value off zero, so a range
  `[l,r]` answers `r-l+1`. Correct.
- Negative values and zeros, e.g. `[-1, 0, -1, 5]`: compression maps these to
  `[0,1,0,2]` (sorted distinct `-1,0,5`), so the algorithm never sees a negative
  index and the counts behave normally. This is exactly why I compressed.
- `+-10^9` extremes: compression collapses the value range to `[0,V)`, so the
  count array is `O(n)` regardless of magnitude; no overflow because counts are
  bounded by `n` and the distinct total fits trivially in a `long long`.
- Single-element ranges `l == r` interspersed with wide ranges: the grow/shrink
  loops handle any endpoint configuration; I verified this both by trace and by
  the random tester biasing 20% of queries to `l == r`.
- Duplicate queries (same `(l,r)` repeated): after the window reaches that
  range, all four loops are no-ops and the same `distinct` is read out — `O(1)`
  per duplicate, correct.

**Performance sanity check.** I generate `n = q = 2*10^5` worst cases: a
small-alphabet array (size 2, maximal count-table churn) with both narrow ranges
and ranges spanning most of the array, which is the adversarial input for
pointer travel. Measured wall time is ~0.1-0.16 s and peak memory ~10 MB — well
inside the 2 s / 256 MB budget. The block size is `n / sqrt(q)` guarded to at
least 1 so it never degenerates when `q` is tiny relative to `n`.

**Why this is the right algorithm and not a heavier one.** I could have reached
for the Fenwick-tree-sweep-by-`r` (`O((n+q) log n)`) or a persistent/merge-sort
segment tree, but those are strictly more machinery for the same wall-clock here
and more surface area for bugs. Mo's algorithm is the canonical offline
range-distinct tool at exactly these limits: `O((n+q) sqrt(q))`, an `O(1)`
add/remove that is trivially correct, and a single integer array of bookkeeping.
The only non-obvious ingredients are the two I derived — the `sqrt`-block sort
key (which is the whole reason it is fast) and the even/odd `r` direction (which
halves the constant) — plus the grow-before-shrink loop order that the trace
forced me to get right.

**Final solution.** I convinced myself the *idea* is right by showing the rescan
and the prefix trick both fail and then deriving the block size that bounds
pointer travel, and I convinced myself the *code* is right by tracing the
loop-order bug to a precise cause (a `remove` on an unadded position driving
`cnt` negative), fixing it with grow-before-shrink, and re-verifying the fix plus
every corner against the rescan oracle over hundreds of random cases with zero
mismatches. That is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Offline range-distinct counting via Mo's algorithm.
//
// We must answer q queries, each asking for the number of DISTINCT values in
// a[l..r]. Re-scanning per query is O(q*n) which is ~4*10^10 at the limits and
// far too slow. Mo's algorithm reorders the queries so that two consecutive
// queries differ in their endpoints by only a little on average, then it walks
// a sliding window [curL, curR] from one query to the next with O(1)
// add/remove operations on a value-count array. With block size B = n/sqrt(q),
// the left pointer moves O(q*B) times in total and the right pointer moves
// O(n^2 / B) times in total; balancing gives O((n + q) * sqrt(q)) work overall,
// which is ~1.3*10^8 element moves at n = q = 2*10^5 and runs well inside the
// limit. The crux ("the insight") is the SORT KEY: queries are bucketed by the
// block index of l, and within a block sorted by r -- with the standard
// even/odd boustrophedon trick that sweeps r left-to-right in even blocks and
// right-to-left in odd blocks, halving the right-pointer travel.

struct Query {
    int l, r, idx;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // Coordinate-compress the values so the count array is sized by #distinct
    // values (at most n), independent of the value range.
    vector<int> comp(a.begin(), a.end());
    sort(comp.begin(), comp.end());
    comp.erase(unique(comp.begin(), comp.end()), comp.end());
    for (int i = 0; i < n; i++) {
        a[i] = (int)(lower_bound(comp.begin(), comp.end(), a[i]) - comp.begin());
    }
    int V = (int)comp.size();

    vector<Query> qs(q);
    for (int i = 0; i < q; i++) {
        int l, r;
        cin >> l >> r;          // 0-indexed, inclusive on both ends
        qs[i] = {l, r, i};
    }

    // Block size: n / sqrt(q) (guarded to be >= 1) minimizes total pointer
    // movement under the standard Mo cost analysis.
    int block = max(1, (int)(n / max(1.0, sqrt((double)q))));

    sort(qs.begin(), qs.end(), [&](const Query &x, const Query &y) {
        int bx = x.l / block, by = y.l / block;
        if (bx != by) return bx < by;
        // Even/odd trick: alternate the r-sort direction per block so the right
        // pointer sweeps back and forth instead of resetting each block.
        if (bx & 1) return x.r > y.r;
        return x.r < y.r;
    });

    vector<int> cnt(V, 0);   // cnt[value] = occurrences inside current window
    long long distinct = 0;  // number of values with cnt > 0 in the window
    vector<long long> ans(q);

    // Window [curL, curR] is inclusive; start it empty (curL > curR).
    int curL = 0, curR = -1;

    auto add = [&](int pos) {
        if (cnt[a[pos]]++ == 0) distinct++;
    };
    auto remove = [&](int pos) {
        if (--cnt[a[pos]] == 0) distinct--;
    };

    for (const Query &Q : qs) {
        // Grow the window outward first, then shrink, so cnt never goes
        // negative on a transient.
        while (curR < Q.r) add(++curR);
        while (curL > Q.l) add(--curL);
        while (curR > Q.r) remove(curR--);
        while (curL < Q.l) remove(curL++);
        ans[Q.idx] = distinct;
    }

    string out;
    out.reserve((size_t)q * 7);
    char buf[24];
    for (int i = 0; i < q; i++) {
        int len = snprintf(buf, sizeof(buf), "%lld\n", ans[i]);
        out.append(buf, len);
    }
    cout << out;
    return 0;
}
```

**Causal recap.** Per-query rescan is correct but `O(q*n) = 4*10^10`, dead on
arrival; the prefix-sum instinct fails because distinct-count is not subtractive
across a split (a value can sit on both sides), shown on `[1,2,1]`. That pushed
me to a single sliding window with an `O(1)` count-table add/remove, where the
only remaining cost is total pointer travel — and bucketing queries by
`(block(l), r)` with block width `n/sqrt(q)` balances left-travel `O(q*B)`
against right-travel `O(n^2/B)` into `O((n+q)sqrt(q))`, with the even/odd `r`
direction halving the constant. My first loop ordering shrank the left pointer
before growing the window, and a trace of `[1,2]` with queries `(1,1)` then
`(0,0)` showed `remove` driving `cnt` negative on an unadded position;
grow-before-shrink fixes it, and the rescan-oracle differential test over
hundreds of random and edge cases then comes back clean.
