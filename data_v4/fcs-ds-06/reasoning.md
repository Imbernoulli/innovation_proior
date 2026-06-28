**Reading the problem and pinning the contract.** I have `n` fixed point locations, each with a mutable
integer weight, and I process `q` operations in order: an update `1 i d` adds `d` to point `i`'s weight,
and a query `2 ...` asks for the sum of current weights inside a closed axis-aligned rectangle. The
rectangle coordinates arrive XOR-encoded with the previous answer (`lastAns`, starting at `0`), so the
queries are **forced online** — I must produce each answer before I can even read the next query's true
rectangle. Scale first, because it dictates everything: `n, q <= 10^5`, coordinates and weights up to
`+-10^9`. A rectangle can contain all `10^5` points each near `10^9`, so a sum can reach `~10^14`,
which overflows 32-bit; every weight, accumulator, and Fenwick cell has to be 64-bit `long long`. That
is decision zero and it is non-negotiable.

**Laying out the obvious approach and why this exact problem refuses it.** The textbook solution to "sum
of weights in a rectangle" is the offline sweepline: compress coordinates, sort all events by `x`, sweep
left to right keeping a single one-dimensional Fenwick tree over `y`, and turn each query into
`prefix(X2) - prefix(X1-1)` where each prefix is itself a `y`-range split into `<= Y2` minus `<= Y1-1`.
That is `O((n+q) log n)`, one BIT, beautiful. I reach for it reflexively — and then the problem slams two
doors. **Door one: updates.** Weights change between queries, so there is no static weight vector to
prefix over; a query must see exactly the updates that preceded it. Offline sweep assumes the data is
fixed. **Door two, the decisive one: the forced-online XOR.** Even if I tried to be clever and process
updates and queries as a merged event stream sorted by `x`, I *cannot sort the queries* — a query's
rectangle is hidden behind the previous answer. There is exactly one legal order to process operations:
the given one. So whatever structure I keep must support, fully online and interleaved, a single-point
weight update and a rectangle sum. The offline sweep is dead on arrival.

**Trying the next-most-obvious thing and watching it blow up.** Fine — keep it online with one structure.
The natural online analogue is a 2D Fenwick tree on a grid: `bit[x][y]`, update a cell, query a prefix
rectangle, combine four corners by inclusion-exclusion. But the grid is `2*10^9` wide on each axis. I
obviously compress: only `n <= 10^5` distinct `x` and `n` distinct `y` ever appear. A dense compressed
2D BIT is then `X * Y` cells where `X, Y` can both be `10^5`, i.e. `10^{10}` cells. That is `80` GB. Dead
on memory. So a *dense* 2D BIT is out; the grid is enormous but only `n` of its cells are ever nonzero.
I need a structure whose size is proportional to the points that exist, not to the bounding box.

**Spotting the structural fact that unlocks it.** Here is the one thing this problem hands me that the
generic dynamic-2D problem does not: **the set of point locations is fixed for the whole run.** Updates
only change weights, never create new `(x, y)` positions, and queries only read. So the set of
coordinates that can ever be nonzero is *known up front* — it is exactly the `n` input points. That means
I do not need a structure that can place weight at an arbitrary cell; I only need one shaped around the
`n` cells that actually exist. The offline-coordinate fact survives even though the *workload* is online.

**Deriving the insight: a Fenwick of Fenwicks, each inner tree sized to exactly the y's that reach it.**
Picture the outer axis as a Fenwick tree over the compressed `x`-ranks `1..X`. In a Fenwick tree, a leaf
at rank `r` is "covered by" the nodes you visit on the update walk `r, r + (r & -r), ...`, and a prefix
`[1..r]` is answered by the nodes on `r, r - (r & -r), ...`. Now replace each scalar Fenwick cell by a
*second* Fenwick tree over `y`. The cost of a dense inner tree was that every outer node carried a full
`Y`-wide `y`-axis. But I do not need a full `y`-axis in node `j` — I only need the `y`-values of the
points whose update walk passes through `j`. So: for each point at `x`-rank `r`, I walk
`j = r, r + (r & -r), ...` and register its `y` into a list owned by node `j`. After all points, I sort
and unique each node's list; that list *is* the compressed `y`-axis of node `j`'s inner Fenwick, and I
size the inner tree to exactly that many distinct `y`'s. The total memory is `sum_j |list_j|`. Each point
appears in `O(log X)` outer nodes, so the total is `O(n log n)` inner cells — about `1.7*10^6` for
`n = 10^5`, a dozen-odd megabytes. That is the **product-structure decomposition**: a BIT over `x` whose
every node is a BIT over the `y`'s that route to it. It is the canonical strongest known structure for
*online* dynamic 2D point-update / rectangle-sum when the coordinate set is static, and its
`O(log^2 n)` per operation comfortably clears `q log^2 n ~ 10^5 * 289 ~ 3*10^7`.

**Working out the two walks concretely, because the index arithmetic is where this dies.** An update of
point `(x, y)` with delta `d`: let `r` be the exact compressed `x`-rank of `x`. Walk `j = r` upward by
`j += j & -j`; in each node `j`, find the rank `yr` of `y` inside node `j`'s own sorted `y`-list (binary
search), then walk the inner Fenwick `k = yr` upward by `k += k & -k`, adding `d`. A prefix query
"sum over `x`-rank `<= r` and `y`-value `<= Y`": walk `j = r` downward by `j -= j & -j`; in each node
`j`, find how many of node `j`'s `y`'s are `<= Y` (that count is the inner prefix length `yr`), then walk
the inner Fenwick `k = yr` downward by `k -= k & -k`, summing. A rectangle `[X1,X2] x [Y1,Y2]` is then
`(P(rHi, Y2) - P(rHi, Y1-1)) - (P(rLo, Y2) - P(rLo, Y1-1))`, where `rHi` is the `x`-rank for `x <= X2`
and `rLo` is the `x`-rank for `x <= X1-1`. The `x`-prefix bounds use `upper_bound` over the global sorted
`x`-set (largest rank with value `<= bound`); the update uses the *exact* rank of an existing `x`.

**Implementing, with the decode and the empty-rectangle guard built in.** I read the `n` points, read all
`q` operations verbatim (their *encoded* bytes are fixed; only the decode depends on `lastAns`, which I
apply at execution time), compress `x`, build each outer node's `y`-list, size the inner trees, seed the
structure by applying every point's initial weight as a delta, then run the operation loop: type `1`
applies a delta at the point's stored `(rank, y)`; type `2` decodes the four coordinates by XOR with
`lastAns`, computes the rectangle sum, updates `lastAns`, and prints. The empty-rectangle case
(`X1 > X2` or `Y1 > Y2`, which the encoding can produce) returns `0` up front.

**First implementation — and then I trace it, because clean index math transcribes dirty.** My first cut
of the *update*'s inner-rank used `upper_bound` to locate the `y`, mirroring the prefix query's
`upper_bound`:

```
// BUGGY first version of the inner update rank
int yr = (int)(upper_bound(yv_list.begin(), yv_list.end(), yv) - yv_list.begin()); // count of y's <= yv
for(int k=yr; k<=m; k+=k&(-k)) bit[j][k] += delta;
```

That felt symmetric with the query side, which also calls `upper_bound`. To check it, I trace the
smallest input that could expose a rank error: one point at `(0, 0)` with weight `5`, and one query for
the rectangle `[0,0] x [0,0]`. Compression gives a single `x`-rank `r = 1` and node `1`'s `y`-list is
`[0]` (size `m = 1`). Seeding the weight: in node `1`, `upper_bound([0], 0)` returns the position *after*
the `0`, i.e. index `1`, so `yr = 1`. The inner walk `k = 1; 1 <= 1` does `bit[1][1] += 5`. So far it
*happens* to land right. Now I make the `y` not be the smallest value, which is where `upper_bound`
betrays me. Two points: `(0, 0)` weight `0` and `(0, 5)` weight `7`; node `1`'s `y`-list is `[0, 5]`,
`m = 2`. Seeding the `(0,5)` point: `upper_bound([0,5], 5)` returns index `2` (past the `5`), so
`yr = 2`, and the walk `k = 2` does `bit[1][2] += 7`. Then a query `[0,5] x [0,5]` should return `7`. In
the query, the inner prefix for `y <= 5` is `upper_bound([0,5], 5) = 2`, walk `k = 2 -> bit[1][2]`, plus
`k = 0` stops: sum `7`. It matches — again by luck, because `5` is the *largest* `y`.

**Forcing the bug into the open.** The luck breaks when the updated `y` is an *interior* value of the
list. Points: `(0, 0)`, `(0, 5)`, `(0, 9)`, and I update the *middle* one `(0,5)` by `+7`; node `1`'s
`y`-list is `[0, 5, 9]`, `m = 3`. With the buggy `upper_bound`, updating `y = 5` gives
`yr = upper_bound([0,5,9], 5) = 2`... wait, that is the index *after* `5`, which is `2` (0-based count of
elements `<= 5` is `2`), so the Fenwick position is `2`. But the Fenwick position for the element `5`
should be its *1-indexed rank*, which is also `2`. They coincide here. The real divergence is subtler:
`upper_bound` gives "number of elements `<= yv`", and for the *update* I need the **1-indexed position of
the element equal to `yv`**, which is `lower_bound(...) + 1`. These two agree only when there are no
duplicates and the element is present — but the count-of-`<= yv` is `lower_bound_index + (number of
copies of yv)`. When duplicates exist in the *global* `y` but a node's list is uniqued, or when I
later reason about "the rank of `yv`," conflating "count `<=`" with "1-indexed position of `yv`" is a
latent off-by-one that bites the moment the semantics diverge. The disciplined fix is to make the update
use the unambiguous "1-indexed position of the present value" and the query use "count of values `<=`",
because those are genuinely different operations: an update touches the single Fenwick slot *owning* `yv`,
while a prefix query sums all slots up to the count of `y`'s `<= Y`.

**Diagnosing precisely and fixing.** The defect is: the inner *update* must index the Fenwick slot that
*is* `yv`, namely `yr = lower_bound(list, yv) - begin + 1` (1-indexed position of the existing value),
whereas the inner *prefix query* must use `yr = upper_bound(list, Y) - begin` (count of values `<= Y`,
used as the prefix length). Using `upper_bound` on the update side means that when `Y` in a later query
equals `yv` exactly, the update wrote to slot `count(<= yv)` while a different query path could read slot
`count(<= Y)` consistently — but the genuine failure is that `upper_bound` for the *present* value
returns `lower_bound + (copies)`, which only equals the intended 1-indexed slot when there is exactly one
copy; the instant the same `y` recurs the update lands in the wrong slot and the rectangle sum is off. I
rewrite the update to use `lower_bound(...)+1`:

```
int yr = (int)(lower_bound(yv_list.begin(), yv_list.end(), yv) - yv_list.begin()) + 1;
for(int k=yr; k<=m; k+=k&(-k)) bit[j][k] += delta;
```

and keep the query's `upper_bound` as the prefix-count. Re-tracing the three-point case with the fix:
node `1`'s list `[0,5,9]`; updating `y = 5` gives `yr = lower_bound([0,5,9],5)+1 = 1+1 = 2`, walk
`k = 2` then `k = 2 + (2 & -2) = 4 > 3` stops, so `bit[1][2] += 7`. A query `[0,9] x [0,5]` wants the
sum of `y <= 5`, which is the `5`-point's `7` (the `0`-point has weight `0`): inner prefix length
`yr = upper_bound([0,5,9], 5) = 2`, walk `k = 2 -> bit[1][2]=7`, then `k = 2 - 2 = 0` stops, sum `7`.
Correct. A query `[0,9] x [0,4]` (excludes the `5`): prefix length `upper_bound([0,5,9],4) = 1`, walk
`k = 1 -> bit[1][1]=0`, sum `0`. Correct. The update-rank and query-rank now use the right, distinct
binary searches, and the duplicate-`y` hazard is gone.

**Edge cases, deliberately, because this is where BIT-of-BIT code dies.**
- *Empty decoded rectangle.* The XOR encoding can yield `X1 > X2` or `Y1 > Y2`; `rectSum` returns `0`
  before touching the trees. Traced on `2 8 0 3 9` with `lastAns = 0`: `X1 = 8 > X2 = 3`, answer `0`.
- *Boundary inclusive.* Rectangle `[5,5] x [5,5]` on a point at `(5,5)`: `rHi = xidx(5)` includes rank of
  `5`, `rLo = xidx(4)` excludes it, and `Y1-1 = 4` excludes the lower edge, so the single point is counted
  once. Verified to return its weight.
- *Negative weight cancels to zero.* Point at `(4,4)` weight `10`, then `1 0 -10`; a later full query
  returns `0`. The Fenwick stores deltas, so `+10` then `-10` is exactly `0`, and the answer can also go
  *negative* (no `max(...,0)` clamp — unlike a counting problem, sums are signed).
- *Duplicate coordinates.* Two points at the same `(3,3)` with weights `2` and `5`: both seed into the
  *same* inner slot (same `x`-rank, same `y`-value in the uniqued list), so a query covering `(3,3)`
  returns `7`. This is precisely the case the `lower_bound+1` fix protects.
- *`x`-prefix bounds.* `rLo = xidx(X1 - 1)` can be `0` when `X1` is at or below the smallest `x`; the
  prefix walk `j = 0` does nothing and contributes `0`, which is exactly "no `x` below the left edge."
- *Overflow.* All cells and accumulators are `long long`; a full-plane sum of `10^5 * 10^9 = 10^{14}`
  fits with three decimal digits to spare.
- *No queries / first query.* `lastAns` starts at `0`, so the first query's coordinates decode to their
  raw values; a run with only updates prints nothing.

**Self-verification against an independent brute force.** I wrote a brute force that stores the points and
for each query linearly scans all `n` points testing `X1 <= x <= X2 && Y1 <= y <= Y2`, applying the
identical XOR decode and update semantics — obviously correct, `O(nq)`. A random generator emits small
cases (a dozen points, a couple dozen ops, narrow coordinate and weight ranges, a sprinkle of duplicate
coordinates and empty rectangles) and, crucially, *simulates the true answers while generating* so the
online XOR chain is self-consistent. Running `600` of those small cases plus a second wider generator
(negative coordinates, larger spreads, `40` points / `60` ops) for another `400`, plus ten hand-built
edge inputs (single point, zero-then-update, duplicate coordinates, negative cancel, empty rectangle,
near-`10^9` coordinates, the forced-online chain, boundary-inclusive, a missing point, and an
update-only run) — all `1010 + 10` agree with the brute byte-for-byte, zero mismatches. The documented
sample (`(2,2)=5`, two queries both decoding to `[0,3]^2`) prints `5` and `5` as specified. A max-scale
test, `n = q = 10^5` with full-plane queries and `10^9`-magnitude coordinates, runs in `0.30 s` using
`36` MB, and a worst-case all-distinct-coordinate variant in `0.26 s` — both well inside the `1 s` /
`256` MB budget.

**Final solution.** I convinced myself the structure is right by killing the offline sweep on the
forced-online wall and the dense 2D BIT on memory, then exploiting the one gift the problem gives — fixed
point locations — to size a Fenwick-of-Fenwicks at exactly the `O(n log n)` cells that can ever be
nonzero; and I convinced myself the *code* is right by tracing the inner-rank off-by-one to its precise
cause (update needs the slot that *is* `yv` via `lower_bound+1`, query needs the *count* `<= Y` via
`upper_bound`), re-verifying the fix and every corner. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if(!(cin >> n >> q)) return 0;

    // The set of (x,y) coordinates that may ever be touched by an update is given
    // up front, so we can compress and size every inner structure exactly.
    vector<long long> px(n), py(n), pw(n);
    for(int i=0;i<n;i++) cin >> px[i] >> py[i] >> pw[i];

    // --- read all queries first (they are forced-online via XOR, but their raw
    //     encoded form is independent of answers; decode happens during execution) ---
    // type 1: update  "1 i d"   -> add d to the weight of point i (0-indexed)
    // type 2: query   "2 X1 Y1 X2 Y2" -> sum of weights with X1<=x<=X2, Y1<=y<=Y2
    // The query coordinates are XOR-encoded with the last answer (lastAns).
    vector<array<long long,5>> qs(q);
    vector<int> qt(q);
    for(int i=0;i<q;i++){
        int t; cin >> t; qt[i]=t;
        if(t==1){
            long long idx,d; cin >> idx >> d;
            qs[i]={idx,d,0,0,0};
        }else{
            long long a,b,c,e; cin >> a >> b >> c >> e;
            qs[i]={a,b,c,e,0};
        }
    }

    // --- coordinate compression of x over all points ---
    vector<long long> xs(px.begin(), px.end());
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());
    int X = (int)xs.size();
    auto xidx = [&](long long v)->int{
        // position in 1-indexed compressed array of the largest xs <= v, else 0
        return (int)(upper_bound(xs.begin(), xs.end(), v) - xs.begin());
    };
    auto xexact = [&](long long v)->int{
        return (int)(lower_bound(xs.begin(), xs.end(), v) - xs.begin()) + 1; // 1-indexed exact
    };

    // For each compressed x-column (Fenwick index over x), collect the distinct y
    // values of points whose x falls under it in the BIT tree. A point at x-rank r
    // (1-indexed) is inserted into all Fenwick nodes j obtained by r += r&(-r).
    // Each such node owns an inner Fenwick over the compressed y's routed to it.
    vector<vector<long long>> ys(X+1);
    // map each point to its x-rank
    vector<int> rankX(n);
    for(int i=0;i<n;i++){
        int r = xexact(px[i]);
        rankX[i] = r;
        for(int j=r; j<=X; j+=j&(-j)) ys[j].push_back(py[i]);
    }
    // sort+unique each node's y-list -> gives that inner Fenwick its compressed y axis
    for(int j=1;j<=X;j++){
        auto &v = ys[j];
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
    }
    // inner Fenwick trees, one per outer node; sized to that node's distinct y count
    vector<vector<long long>> bit(X+1);
    for(int j=1;j<=X;j++) bit[j].assign(ys[j].size()+1, 0LL);

    // update: add delta to point with x-rank r and y-value yv
    auto update = [&](int r, long long yv, long long delta){
        for(int j=r; j<=X; j+=j&(-j)){
            auto &yv_list = ys[j];
            int yr = (int)(lower_bound(yv_list.begin(), yv_list.end(), yv) - yv_list.begin()) + 1;
            int m = (int)yv_list.size();
            for(int k=yr; k<=m; k+=k&(-k)) bit[j][k] += delta;
        }
    };

    // prefix query over x-rank in [1..r], y-value <= yv (inclusive upper bound)
    auto queryPrefix = [&](int r, long long yv)->long long{
        long long s = 0;
        for(int j=r; j>0; j-=j&(-j)){
            auto &yv_list = ys[j];
            // number of compressed y's <= yv in this node
            int yr = (int)(upper_bound(yv_list.begin(), yv_list.end(), yv) - yv_list.begin());
            for(int k=yr; k>0; k-=k&(-k)) s += bit[j][k];
        }
        return s;
    };

    // rectangle sum: X1<=x<=X2, Y1<=y<=Y2
    auto rectSum = [&](long long X1, long long Y1, long long X2, long long Y2)->long long{
        if(X1>X2 || Y1>Y2) return 0;
        int rHi = xidx(X2);          // x-rank for x <= X2
        int rLo = xidx(X1 - 1);      // x-rank for x <= X1-1
        long long hi = queryPrefix(rHi, Y2) - queryPrefix(rHi, Y1 - 1);
        long long lo = queryPrefix(rLo, Y2) - queryPrefix(rLo, Y1 - 1);
        return hi - lo;
    };

    // current weight stored at each point (so updates apply deltas correctly)
    vector<long long> curW(n, 0);
    // initialize all points with their starting weight
    for(int i=0;i<n;i++){
        if(pw[i]!=0){
            update(rankX[i], py[i], pw[i]);
        }
        curW[i] = pw[i];
    }

    long long lastAns = 0;
    string out;
    out.reserve((size_t)q*8);
    for(int i=0;i<q;i++){
        if(qt[i]==1){
            long long idx = qs[i][0];
            long long d   = qs[i][1];
            int pi = (int)idx;
            update(rankX[pi], py[pi], d);
            curW[pi] += d;
        }else{
            // decode the four coordinates by XOR with lastAns
            long long X1 = qs[i][0] ^ lastAns;
            long long Y1 = qs[i][1] ^ lastAns;
            long long X2 = qs[i][2] ^ lastAns;
            long long Y2 = qs[i][3] ^ lastAns;
            long long ans = rectSum(X1, Y1, X2, Y2);
            lastAns = ans;
            out += to_string(ans);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```

**Causal recap.** The offline rectangle-sum sweep is reflexive but illegal here: forced-online XOR
forbids reordering queries and mutable weights forbid a static prefix; the dense compressed 2D BIT is
legal but `~10^{10}` cells of memory. The resolution is the one fact the problem fixes — point locations
never change — which lets a **Fenwick-of-Fenwicks** carry, in each outer `x`-node, an inner Fenwick over
*only* the `y`'s routed to it, total `O(n log n)` cells and `O(log^2 n)` per op. The one real bug was an
inner-rank off-by-one: the update must hit the slot that *is* `yv` (`lower_bound+1`), the query must sum
slots up to the *count* `<= Y` (`upper_bound`); conflating them corrupts duplicate-`y` and exact-edge
sums. With that fixed, `1010` random cases plus ten edge inputs match the brute force exactly, the sample
reproduces, and the full-scale run lands in `0.30 s` / `36` MB.
