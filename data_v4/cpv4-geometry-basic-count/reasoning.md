I have `n` integer points and I need to count axis-aligned right triangles: triples of the points that form a triangle with one right angle whose two legs are a vertical and a horizontal segment. A triangle is its set of three corners, so I count each triple once. Two things about the constraints fix the shape of the solution before I pick an algorithm. Coordinates run to `10^9`, but I will only ever use them to *group* points — never multiply two coordinates — so their magnitude is harmless. The count itself is the danger. A full `k x k` grid packs `k^2` points and produces about `k^2 (k-1)^2` triangles, and with `k^2 ~ 2*10^5` (so `k ~ 447`) that is roughly `447^2 * 446^2 ~ 4*10^10` — well past signed 32-bit (`~2.1*10^9`). The accumulator has to be `long long`; an `int` is a silent wrong answer on exactly the grid tests. The other trap the statement plants is duplicates: input points may repeat, but coincident points are one location and a triangle needs three *distinct* positions. If I count over raw input, a column that the input writes as `(0,0),(0,0),(0,5)` looks like three points sharing `x=0`, and I invent triangles whose vertical leg has zero length. So I must collapse exact duplicates before counting.

There are two ways to count, and I want the one whose no-double-count I can actually prove. The per-pair route enumerates vertical segments (pairs sharing a column) and horizontal segments (pairs sharing a row) and tries to glue compatible ones into triangles; a single triangle is reachable through more than one pairing, so it over-counts and needs a correction I would have to get exactly right. The per-anchor route uses a structural fact: an axis-aligned right triangle has a *unique* right-angle vertex `P`, from which one leg runs vertically (a corner sharing `P`'s x) and the other horizontally (a corner sharing `P`'s y). The other two corners are not right angles, so every triangle is owned by exactly one of its corners. Counting `(#points in P's column other than P) * (#points in P's row other than P)` for each `P` and summing counts each triangle once, with no division.

The per-pair scheme over-counts, and the sample shows by how much. On `(0,0),(0,2),(0,3),(2,0),(3,0)`, column `x=0` holds `C(3,2)=3` vertical pairs and row `y=0` holds `C(3,2)=3` horizontal pairs, so "multiply the totals" gives `9` — but the answer is `4`. The gap is exactly that a vertical pair like `(0,2)-(0,3)` shares no vertex with row `y=0`, so most pairings never meet at a corner. Repairing per-pair means re-imposing "do these two legs share an endpoint," which is the per-anchor count wearing a correction factor. So I commit to per-anchor.

The formula: fix `P=(px,py)`; a triangle with its right angle at `P` is determined by choosing the vertical-leg endpoint (any other point with `x=px`) and the horizontal-leg endpoint (any other point with `y=py`) independently, so `P` anchors `col(px) * row(py)` triangles where each count excludes `P` itself. The answer is `sum_P (col(px)-1)*(row(py)-1)`. On the sample only `(0,0)` sits in both a populated column (`col=3`) and a populated row (`row=3`), giving `(3-1)*(3-1)=4`; every other point has a lone factor of one and contributes zero. Total `4`.

Now the transcription, where the duplicate trap actually bites. The naive version builds the column/row maps over the raw points and sums over the raw points:

```
map<long long,long long> col, row;
for (auto &p : raw) { col[p.first]++; row[p.second]++; }
for (auto &p : raw) answer += (col[p.first]-1) * (row[p.second]-1);
```

This looks faithful, but trace `(0,0),(0,0),(5,0)` — only two distinct positions, so the true answer is `0`. Over raw input, `col[0]=2`, `col[5]=1`, `row[0]=3`. Summing over the three raw points: each `(0,0)` contributes `(2-1)*(3-1)=2`, and `(5,0)` contributes `0`, for a total of `4` — pure phantom. Two errors compound: summing over raw points tallies the same anchor once per repeat, and the duplicate-inflated `col`/`row` counts let a "leg" be drawn to a copy of `P` itself. Both vanish once I collapse coincident points first and count over distinct positions only:

```
set<pair<long long,long long>> seen;
vector<pair<long long,long long>> pts;
for (auto &p : raw) if (seen.insert(p).second) pts.push_back(p);
```

with `col`/`row` and the sum then built over `pts`. Re-tracing `(0,0),(0,0),(5,0)`: `pts = {(0,0),(5,0)}`, `col[0]=1`, `col[5]=1`, `row[0]=2`, and both distinct points have a zero factor, so the total is `0`. The sample, having no duplicates, is unchanged at `4`.

The exclude-self `-1` is load-bearing on *both* factors, not one. A single point `(0,0)` with no others must give `0`; with the `-1` it does (`(1-1)*(1-1)`), but without it each factor is `1` and it invents a triangle whose two "other" corners are `P` twice. And the `-1` has to sit on each factor independently: for the L-shape `(0,0),(0,1),(1,0)`, `col[0]=2`, `row[0]=2`, so `(0,0)` gives `(2-1)*(2-1)=1` — the one real triangle. Subtracting only from the column would give `(2-1)*2=2`, double the truth.

The degenerate corners all reduce to a zero factor: `n=0` runs the sum loop zero times (and `if (!(cin >> n)) return 0;` guards empty stdin), and fewer than three distinct points or an all-collinear set leaves every term with one empty leg, so `0`. So the program dedups to distinct positions, builds ordered column/row maps, and sums `(col-1)*(row-1)` over the distinct points in `O(n log n)`; the full module is in the answer.
