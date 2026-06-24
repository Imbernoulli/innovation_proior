**Reading the problem and pinning the contract.** I have `n` points with integer coordinates, and I must count the triangles whose three corners are points of the set and which are *axis-aligned right* — one right angle whose two legs are parallel to the axes, i.e. one vertical leg and one horizontal leg. A triangle is identified by its set of three corner points, so I count each triple once. Input is `n` then `n` pairs `x y`; I print one integer. Two facts I want fixed before choosing an algorithm. First, coordinates run to `10^9` in magnitude, so coordinates themselves fit in 64-bit but I will never multiply two *coordinates*, only *counts*. Second, and this drives the data type: `n <= 2*10^5`, and the count can be enormous. A full `k x k` grid has `k^2` points and, as I will derive, about `k^2(k-1)^2` triangles; with `k^2 ~ 2*10^5` that is roughly `(2*10^5)(447-1)^2 ~ 4*10^10`, which overflows signed 32-bit (`~2.1*10^9`). So the accumulator is `long long`. An `int` here is a silent wrong-answer on the grid tests.

**A subtlety the statement forces me to respect: duplicate points.** The input may repeat a point, but coincident points are *one location* and a triangle needs three *distinct positions*. So before any counting I must collapse exact duplicates. If I forget this, a column that physically holds two distinct points but appears in the input as `(0,0),(0,0),(0,5)` will look like it has three points sharing `x=0`, and I will count phantom triangles whose "vertical leg" has zero length. I will come back to this — it is exactly the kind of dedup that is easy to get subtly wrong.

**Laying out the candidate approaches.** Two routes, and I want the one whose counting I can *prove* counts each triangle exactly once.

- *Per-pair / per-edge counting.* For every pair of points sharing a column I have a vertical segment; for every pair sharing a row I have a horizontal segment; somehow glue compatible segments into triangles. The danger is structural: a single triangle can be reached through more than one pair, so this approach tends to over-count and needs a correction factor I have to get exactly right.
- *Per-anchor counting.* Observe that an axis-aligned right triangle has a *unique* right-angle vertex `P`: from `P` one leg goes vertically (a second corner sharing `P`'s x) and the other horizontally (a third corner sharing `P`'s y). The other two corners are not right angles. So every triangle is owned by exactly one corner. If, for each point `P`, I count `(#points in P's column other than P) * (#points in P's row other than P)`, and sum over all `P`, I count each triangle exactly once — no division. This is `O(n log n)` with maps and, crucially, has a clean uniqueness argument.

The per-anchor route is the one with a provable no-double-count guarantee, so I lean there. But "provable" only if I get the exclude-self subtraction and the dedup right, and I have been burned by exactly those. Let me first try the per-pair idea anyway, precisely to *see* it double-count, so I know what I am avoiding.

**Stress-testing the per-pair idea before committing — and watching it double-count.** Suppose I try: count `V` = number of vertical segments (pairs sharing a column) and `H` = number of horizontal segments (pairs sharing a row), then claim the answer is some product. Take the worked sample, the 5 points `(0,0),(0,2),(0,3),(2,0),(3,0)`. Column `x=0` holds `(0,0),(0,2),(0,3)` — that is `C(3,2)=3` vertical segments. Row `y=0` holds `(0,0),(2,0),(3,0)` — `C(3,2)=3` horizontal segments. A naive "multiply totals" gives `3*3=9`, but the true answer is `4`. The discrepancy is glaring: a vertical segment like `(0,2)-(0,3)` does not even pass through any point on row `y=0`, so it cannot be a leg of any right triangle here; and the per-pair view has no clean way to insist that the vertical and horizontal legs *share their right-angle endpoint*. To fix per-pair I would need, per pair, to know how many points sit at the corner — which is just the per-anchor count in disguise, now wrapped in an error-prone correction. The experiment did its job: it showed the per-pair route over-counts (9 vs 4) and that the honest quantity is "legs that meet at a common vertex". Per-pair is out; I commit to per-anchor.

**Deriving the per-anchor formula and sanity-checking it.** Fix a point `P=(px,py)`. A triangle with its right angle at `P` is determined by choosing the vertical-leg endpoint — any other point with `x=px` — and the horizontal-leg endpoint — any other point with `y=py`. These choices are independent, so the number of triangles anchored at `P` is `col(px) * row(py)` where `col(px)` = number of points in column `px` *excluding P*, and `row(py)` similarly. The full answer is `sum over P of col-minus-self(px) * row-minus-self(py)`. Uniqueness of the right-angle vertex means no triangle is counted twice, so this sum is exact with no correction.

Let me verify on the sample `(0,0),(0,2),(0,3),(2,0),(3,0)`. Column counts: `x=0` holds 3 points, `x=2` holds 1, `x=3` holds 1. Row counts: `y=0` holds 3, `y=2` holds 1, `y=3` holds 1. Now per point, with "minus self":
- `(0,0)`: col `x=0` -> `3-1=2`; row `y=0` -> `3-1=2`; contributes `2*2=4`.
- `(0,2)`: col `x=0` -> `2`; row `y=2` -> `1-1=0`; contributes `0`.
- `(0,3)`: col -> `2`; row `y=3` -> `0`; contributes `0`.
- `(2,0)`: col `x=2` -> `0`; row `y=0` -> `2`; contributes `0`.
- `(3,0)`: col -> `0`; row -> `2`; contributes `0`.
Total `4`. Matches the stated answer, and the four triangles are exactly `{(0,0),(0,2),(2,0)}`, `{(0,0),(0,2),(3,0)}`, `{(0,0),(0,3),(2,0)}`, `{(0,0),(0,3),(3,0)}` — all anchored at `(0,0)`. The formula is right.

**First implementation — and immediately a trace, because clean formulas transcribe dirty.** My first cut, written quickly:

```
map<long long,long long> col, row;
for (auto &p : raw) { col[p.first]++; row[p.second]++; }
long long answer = 0;
for (auto &p : raw) {
    long long up   = col[p.first] - 1;
    long long side = row[p.second] - 1;
    answer += up * side;
}
```

I count occurrences of each x and each y over the *raw* input, then sum `(col-1)*(row-1)` over the raw points. It looks like a faithful transcription. But the duplicate subtlety is nagging at me, so I trace the smallest input that can expose it. Take three input points `(0,0),(0,0),(5,0)` — but here only two *distinct* positions exist, so there are zero triangles (a triangle needs three distinct corners). What does the code say? Building maps over raw points: `col[0]=2` (two points with x=0), `col[5]=1`; `row[0]=3` (all three have y=0). Now summing over the three raw points:
- first `(0,0)`: `up = col[0]-1 = 1`, `side = row[0]-1 = 2`, adds `1*2 = 2`.
- second `(0,0)`: same, adds `2`.
- `(5,0)`: `up = col[5]-1 = 0`, adds `0`.
Total `4`. The true answer is `0`.

**Diagnosing the bug.** The code returns `4` for an input with only two distinct points — pure phantom. The defect is the duplicate handling, on two fronts at once. First, summing over *raw* points double-counts the anchor: the two coincident `(0,0)` entries each contribute a full term, so even a legitimate anchor would be tallied as many times as it is repeated. Second, the column/row counts are inflated by duplicates: `col[0]=2` and `row[0]=3` treat the two coincident `(0,0)` as two distinct column members, so the "vertical leg" can be drawn to a point that is actually `P` itself — a zero-length, illegal leg. Both errors stem from never collapsing coincident points. The fix is not a patch on the arithmetic; it is to dedup first.

**Fixing: dedup, then count over distinct positions.** I collapse exact duplicates into a set of distinct points, build `col`/`row` over those distinct points, and sum over the distinct points only:

```
set<pair<long long,long long>> seen;
vector<pair<long long,long long>> pts;
for (auto &p : raw) if (seen.insert(p).second) pts.push_back(p);
map<long long,long long> col, row;
for (auto &p : pts) { col[p.first]++; row[p.second]++; }
long long answer = 0;
for (auto &p : pts) {
    answer += (col[p.first]-1) * (row[p.second]-1);
}
```

Re-trace `(0,0),(0,0),(5,0)`: dedup gives distinct `pts = {(0,0),(5,0)}`. `col[0]=1, col[5]=1`; `row[0]=2`. Sum over the two distinct points: `(0,0)` -> `(1-1)*(2-1)=0*1=0`; `(5,0)` -> `(1-1)*(2-1)=0`. Total `0`. Correct — no triangle from two positions. Re-trace the worked sample (no duplicates, so dedup is a no-op): still `4`, as derived. The case that broke now passes, and it broke for the precise reason I fixed (duplicates inflating counts and the per-raw-point sum), which is the evidence I trust.

**A second debug episode: the exclude-self off-by-one.** I want to be sure the `-1` ("exclude `P` itself") is genuinely needed and correctly placed, because dropping it is the classic off-by-one in this template. Suppose I had written `up = col[p.first]` and `side = row[p.second]` without the `-1`. Trace the single point `(0,0)` with no others — answer must be `0` (you cannot make a triangle from one point). With the `-1`: `col[0]=1`, `up=0`, contributes `0`. Without the `-1`: `up = col[0] = 1`, `side = row[0] = 1`, contributes `1` — a phantom triangle whose two "other" corners are `P` itself, twice. So the `-1` is exactly what forbids using `P` as its own leg endpoint; it is load-bearing, not cosmetic. Now a sharper check that the `-1` belongs on *each* factor and not just once. Take `(0,0),(0,1),(1,0)` (an L). `col[0]=2, col[1]=1`; `row[0]=2, row[1]=1`. Per point: `(0,0)` -> `(2-1)*(2-1)=1`; `(0,1)` -> `(2-1)*(1-1)=0`; `(1,0)` -> `(1-1)*(2-1)=0`. Total `1` — the one right triangle anchored at `(0,0)`. If I had subtracted `1` from only one factor (say only the column), `(0,0)` would give `(2-1)*2 = 2`, double the truth. So both factors need the exclude-self, and the trace pins it.

**Edge cases, deliberately, because this is where counting code dies.**
- `n = 0`: no points read, `pts` empty, the sum loop never runs, `answer = 0`. The empty set has no triangles — correct. (The `if (!(cin >> n)) return 0;` also covers truly empty stdin.)
- Fewer than three distinct points, e.g. `n=2` `(0,0),(0,5)`: `col[0]=2, row[0]=1, row[5]=1`. `(0,0)` -> `(2-1)*(1-1)=0`; `(0,5)` -> `(2-1)*(1-1)=0`. Total `0` — you cannot form a triangle from two points, and indeed every term has a zero factor because no point shares both a column *and* a separate row partner. Correct.
- All points in one column, e.g. `(0,0),(0,1),(0,2)`: every `row[...] = 1`, so each `side = 0`; answer `0`. Three collinear points form no triangle at all — correct.
- Heavy duplicates: e.g. `(1,1),(1,1),(1,4),(5,1),(1,4)` collapses to distinct `{(1,1),(1,4),(5,1)}`. `col[1]=2, col[5]=1`; `row[1]=2, row[4]=1`. `(1,1)` -> `(2-1)*(2-1)=1`; `(1,4)` -> `(2-1)*(1-1)=0`; `(5,1)` -> `(1-1)*(2-1)=0`. Total `1`, the triangle `{(1,1),(1,4),(5,1)}` anchored at `(1,1)`. Correct, and the duplicates did not inflate it.
- Overflow: the accumulator is `long long`. On a `447 x 447` grid the answer is `447^2 * 446^2 ~ 3.97*10^10`, well within 64-bit and well past 32-bit — so `long long` is mandatory and sufficient. Coordinates are only ever used as map keys, never multiplied, so coordinate magnitude (`10^9`) cannot overflow anything.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so parsing is format-agnostic.

**Final solution.** I convinced myself the *idea* is right by disproving per-pair (it gave `9` vs the true `4` on the sample, because legs need not meet) and by deriving the per-anchor product from the uniqueness of the right-angle vertex, then hand-checking it on the sample. I convinced myself the *code* is right by tracing two failing variants to precise causes — duplicates inflating counts and the per-raw-point sum (fixed by dedup-then-count-over-distinct), and the exclude-self off-by-one on each factor (fixed by `-1` on both) — and re-verifying the fixes and the corners. That is what I ship: one self-contained file, the `O(n log n)` per-anchor count I can defend rather than the per-pair scheme I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> xs(n), ys(n);
    // dedup identical points: a multiset of coincident points is one location
    set<pair<long long,long long>> seen;
    vector<pair<long long,long long>> pts;
    for (int i = 0; i < n; i++) {
        long long x, y;
        cin >> x >> y;
        if (seen.insert({x, y}).second) pts.push_back({x, y});
    }
    int m = (int)pts.size();

    // colCount[x] = number of DISTINCT points with that x; rowCount[y] similarly
    map<long long, long long> colCount, rowCount;
    for (auto &p : pts) {
        colCount[p.first]++;
        rowCount[p.second]++;
    }

    // For each point P, right triangles with the right angle AT P and legs
    // parallel to the axes = (#points sharing P's column, other than P)
    //                       * (#points sharing P's row, other than P).
    // Each such triangle has a unique right-angle vertex, so summing over P
    // counts every triangle exactly once -- no division, no double counting.
    long long answer = 0;
    for (auto &p : pts) {
        long long up = colCount[p.first] - 1;  // exclude P itself
        long long side = rowCount[p.second] - 1;
        answer += up * side;
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The per-pair counting scheme looked plausible but over-counted on the sample (`9` vs the true `4`, because a vertical and a horizontal segment need not meet at a shared right-angle vertex), so I moved to per-anchor counting, whose correctness rests on the unique right-angle vertex of every axis-aligned right triangle: summing `(col-1)*(row-1)` over points counts each triangle exactly once. My first transcription counted over the *raw* input and built column/row counts including duplicates, which a trace of `(0,0),(0,0),(5,0)` exposed by returning `4` phantom triangles for two distinct positions; collapsing coincident points first and summing over distinct points fixes it, and a separate trace pinned the exclude-self `-1` to *both* factors (one point -> `0`, the L-shape -> `1`); with `long long` accumulators the empty, sub-three-point, collinear, duplicate-heavy, and grid-overflow corners all close out.
