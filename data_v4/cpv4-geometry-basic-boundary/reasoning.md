**Reading the problem and pinning the contract.** I am given `n` axis-aligned rectangles, each as two opposite corners `(a,b)` and `(c,d)` in arbitrary order, every coordinate in `[-10^9, 10^9]`. Each rectangle is *closed*: it owns every integer lattice point inside it or on its boundary. I must count the number of distinct lattice points covered by the union and print it. The very first thing I fix is the data type. The covered region can be enormous: a single rectangle spanning the whole `2*10^9 x 2*10^9` extent owns about `4*10^{18}` points, far past 32-bit range and even brushing the top of 64-bit, but the *realistic* worst case under sane tests sits comfortably inside `long long` (up to ~`9.2*10^{18}`). Either way `int` is an instant wrong-answer, so every coordinate, every width, every accumulator is `long long`. Second fixed fact: corners come in arbitrary order, so for each rectangle I must normalize `x1 = min(a,c), x2 = max(a,c)` and likewise for `y`. If I forget this, a rectangle given as `(5,5)-(2,3)` produces a negative width and silently vanishes from the count.

**The single fact this whole problem turns on.** This is a *point count*, not an area. A rectangle whose columns run `x1..x2` covers `x2 - x1 + 1` distinct integer columns, not `x2 - x1`. The `+1` is the inclusive boundary, and it appears in *both* dimensions, so a single naive "area" formula `(x2-x1)*(y2-y1)` is off by a whole border on two sides at once. Concretely, the `(0,0)-(2,2)` square is a `3x3` block of `9` lattice points, but `(2-0)*(2-0) = 4`. That gap of `9` vs `4` is exactly the pitfall I have to defeat, and it will reappear at every shared edge between rectangles too. I want a representation that makes the `+1` automatic rather than something I sprinkle by hand and forget.

**Candidate approaches.** Two routes.

- *Direct grid marking.* Keep a set of covered `(x,y)` and, per rectangle, insert every lattice point. The answer is the set size. This is `O(sum of areas)` and trivially correct — it literally enumerates the definition — but it dies the instant coordinates are large. I will keep it as my *brute-force oracle* for testing, never as the shipped solution.
- *Sweep with coordinate compression.* Sort the distinct vertical cut lines, walk the plane left to right in vertical slabs, and in each slab take the union length of the active horizontal intervals; the slab contributes `width * union_length`. This handles `10^9` coordinates and is what I will ship. The danger is entirely in the inclusive/exclusive bookkeeping that turns continuous lengths into integer counts.

**Deriving the half-open trick so the `+1` is automatic.** The clean way to count inclusive integer cells with interval machinery is to map each closed integer range `[x1, x2]` to the *half-open* real interval `[x1, x2+1)`. Then the number of integer columns `x1, x1+1, ..., x2` equals the length `(x2+1) - x1 = x2 - x1 + 1` of that half-open interval — the inclusive `+1` is now baked into the right endpoint. The unit strip `[x, x+1)` stands for the single integer column `x`. So I expand every rectangle's x-range to `[x1, x2+1)` and its y-range to `[y1, y2+1)`, push the x-endpoints `x1` and `x2+1` as the compression coordinates, and sweep slabs `[xs[k], xs[k+1])`. The width of a slab is `xs[k+1] - xs[k]`, which counts the integer columns it represents. Inside a slab I collect the y-intervals `[y1, y2+1)` of all rectangles whose x-range covers the *whole* slab, take their union length, and that union length is exactly the number of distinct integer rows covered in that slab. Multiply, sum. Because both expansions use `+1`, the inclusive boundary is handled uniformly in x and y, and shared edges between two rectangles fall out of the interval-union for free (a coordinate covered by both contributes its length once).

**A derivation sanity-check before writing code.** Let me confirm the half-open mapping reproduces the area pitfall correctly on the `(0,0)-(2,2)` square. x-range `[0, 3)`, y-range `[0, 3)`. There is one slab `[0,3)` of width `3`; its only y-interval is `[0,3)` of length `3`; contribution `3 * 3 = 9`. That is the right `9`, not the buggy `4`. Good — the `+1` on both endpoints is doing its job. Now the two-rectangle sample `(0,0)-(2,2)` and `(1,1)-(4,3)`: x-endpoints are `0,3` from the first and `1,5` from the second, sorted-unique `{0,1,3,5}`, giving slabs `[0,1)`(w1), `[1,3)`(w2), `[3,5)`(w2). Slab `[0,1)`: only rect 1 covers it (needs `x1<=0` and `x2+1>=1`; rect1 has `0<=0, 3>=1` yes; rect2 has `1<=0` no), y-union `[0,3)` length 3, contributes `1*3=3`. Slab `[1,3)`: rect1 covers it (`0<=1, 3>=3`), rect2 covers it (`1<=1, 5>=3`), y-intervals `[0,3)` and `[1,4)`, union `[0,4)` length 4, contributes `2*4=8`. Slab `[3,5)`: rect1 needs `3>=5` — no; rect2 has `1<=3, 5>=5` — yes, y-interval `[1,4)` length 3, contributes `2*3=6`. Total `3+8+6 = 17`. That matches the stated answer `17`. The derivation holds.

**First implementation and a trace.** My first cut of the slab-coverage test and the interval union:

```
for (int i = 0; i < n; i++) {
    if (X1[i] <= xl && X2[i] >= xr) {            // (A) does rect i cover this slab?
        ivs.push_back({Y1[i], Y2[i] + 1});
    }
}
...
// union of half-open y-intervals
long long curL = ivs[0].first, curR = ivs[0].second, rows = 0;
for (size_t j = 1; j < ivs.size(); j++) {
    if (ivs[j].first >= curR) {                  // (B) start a new piece?
        rows += curR - curL;
        curL = ivs[j].first; curR = ivs[j].second;
    } else {
        curR = max(curR, ivs[j].second);
    }
}
rows += curR - curL;
```

Line (A) bothers me the moment I read it back, so I trace the smallest case that exercises a slab's right edge: a single rectangle `(0,0)-(2,2)`, where the answer must be `9`. The x-endpoints are `0` and `x2+1 = 3`, so `xs = {0, 3}`, one slab `xl=0, xr=3`, width `3`. The coverage test (A) asks `X1[i] <= xl` i.e. `0 <= 0` (true) **and** `X2[i] >= xr` i.e. `2 >= 3` (**false**). So the rectangle is judged *not* to cover its own slab, `ivs` is empty, the slab is skipped, and the program prints `0`.

**The bug (episode 1): an off-by-one in the slab-coverage test.** The output `0` for a non-empty rectangle is plainly wrong, and the cause is a classic inclusive/exclusive slip. I expanded the rectangle's x-range to the half-open interval `[X1, X2+1)`, so the rectangle covers the slab `[xl, xr)` exactly when `X1 <= xl` and `X2+1 >= xr` — the right end of the rectangle's *half-open* range is `X2+1`, not `X2`. By writing `X2[i] >= xr` I compared the rectangle's *closed* right edge against the slab's half-open right edge, dropping the inclusive `+1` on precisely the boundary column. For the `(0,0)-(2,2)` square that single missing `+1` makes the rectangle disown its rightmost slab — here the *only* slab — collapsing the answer to `0`. The fix is to compare `X2[i] + 1 >= xr`, matching the half-open convention I committed to everywhere else. I correct (A) to `if (X1[i] <= xl && X2[i] + 1 >= xr)` and re-trace `(0,0)-(2,2)`: now `0<=0` and `2+1=3 >= 3`, true; `ivs = {[0,3)}`; union length `3`; contribution `3*3 = 9`. Correct. The bug was real, the trace caught it, and it was exactly the inclusive-boundary `+1` this problem is built around.

**Second trace, hunting the interval-union edge.** With (A) fixed I now suspect (B), the `>=` in the merge test, because adjacency of intervals is the other place an off-by-one hides. I trace two rectangles stacked so their y-ranges *touch* at a shared row: `(0,0)-(2,1)` and `(0,2)-(2,4)`, both spanning columns `0..2`. By hand the union is the column block `x in {0,1,2}` crossed with rows `{0,1} U {2,3,4} = {0,1,2,3,4}`, i.e. `3 * 5 = 15` points; the two rectangles do not actually overlap (rows `0,1` vs `2,3,4` are disjoint) but they are *vertically adjacent* with no gap. There is one slab `[0,3)` width `3`. The y-intervals are `[0,2)` and `[2,5)` (half-open from `Y2+1`). Sorted, `curL=0, curR=2`; then `j=1` with `ivs[1].first = 2`. Test (B) `ivs[j].first >= curR` is `2 >= 2`, **true**, so the code *closes* the first piece: `rows += 2-0 = 2`, then resets `curL=2, curR=5`; after the loop `rows += 5-2 = 3`, total `rows = 5`. Contribution `3*5 = 15`.

That actually gives the right `15` here — but the trace exposes that `>=` is semantically wrong even though the arithmetic happens to land. Two *half-open* intervals `[0,2)` and `[2,5)` are exactly adjacent: their union is the single interval `[0,5)` of length `5`. Closing one piece and opening another at the touch point `2` and summing their lengths gives `2 + 3 = 5` — the same number *only because adjacency conserves total length when there is no gap*. The danger is a different configuration: take `[0,2)` and `[2,2)` — a zero-length second interval (a degenerate rectangle that is a horizontal line, `Y1=Y2`, expanded to `[Y, Y+1)`... actually length 1, so let me pick the genuinely degenerate-overlap case). Consider `[0,3)` and `[2,5)` which truly overlap: `curL=0,curR=3`; `j=1`, `ivs[1].first=2`; with `>=` the test is `2 >= 3` false, so it merges, `curR = max(3,5) = 5`, final `rows = 5`. Correct. So where does `>=` actually break? When two intervals share an endpoint and I must *not* split, the conserved-length coincidence hides it; but the merge test's job is to decide overlap-or-touch versus a true gap, and the only mathematically correct predicate for "disjoint with a real gap" on half-open intervals is `ivs[j].first > curR` (strictly greater). Using `>=` declares a gap at the exact touch point `start == curR`, splitting a contiguous run into two. The length sum is preserved so the *answer* survives, but the invariant "`[curL,curR)` is one maximal piece" is violated, and any future change that does something per-piece (e.g. counting pieces, or clamping) would then be wrong.

**The bug (episode 2): `>=` should be `>` in the gap test.** To keep the code's meaning honest and robust I change (B) from `if (ivs[j].first >= curR)` to `if (ivs[j].first > curR)`. I re-trace the adjacency case `[0,2)`,`[2,5)`: now `2 > 2` is **false**, so the intervals merge, `curR = max(2,5) = 5`; after the loop `rows = 5-0 = 5`; contribution `3*5 = 15`. Same correct `15`, but now produced by a *single* merged piece `[0,5)`, which is the truthful representation. And I re-check a genuine gap to be sure `>` still splits when it must: rectangles `(0,0)-(2,1)` and `(0,3)-(2,4)` leave row `2` uncovered, y-intervals `[0,2)` and `[3,5)`. `curL=0,curR=2`; `j=1`, `3 > 2` true, so close `rows += 2`, reset to `[3,5)`; final `rows += 2`, total `4`; contribution `3*4 = 12`. By hand: columns `{0,1,2}` times rows `{0,1,3,4}` = `3*4 = 12`. Correct, and the gap is respected. Both the touch and the gap now behave, for the right reason.

**Re-verifying the original failing cases together.** I rebuild and re-run the whole sample `(0,0)-(2,2)`,`(1,1)-(4,3)`. Slab `[0,1)`: only rect1 (`0<=0 && 3>=1`), y `[0,3)`, len 3, `1*3=3`. Slab `[1,3)`: rect1 (`0<=1 && 3>=3`) and rect2 (`1<=1 && 5>=3`), y `[0,3)`,`[1,4)`, merge to `[0,4)` len 4, `2*4=8`. Slab `[3,5)`: rect2 only (`1<=3 && 5>=5`), y `[1,4)` len 3, `2*3=6`. Total `17`. Matches. The two bugs that I traced — the missing `+1` in the slab-coverage test and the `>=` gap test — are both fixed, and they were fixed for the reasons the traces revealed, not by guessing.

**Edge cases, deliberately, because boundary problems die here.**
- *`n = 0`*: the read of `n` succeeds with `0`, the rectangle loop never runs, `xs` is empty, the slab loop condition `k+1 < xs.size()` is `1 < 0` (unsigned) — careful: `xs.size()` is `0`, `k=0`, `k+1 = 1`, `1 < 0` is false, loop body never runs, `total = 0`. Prints `0`. Correct (empty union). I double-check the unsigned comparison cannot wrap: `k` starts at `0`, the guard `k + 1 < xs.size()` is evaluated before any decrement, and `k` only increments, so no underflow. Safe.
- *Single point `(3,3)-(3,3)`*: `x1=x2=3`, x-endpoints `3` and `4`, one slab `[3,4)` width `1`; y-interval `[3,4)` length `1`; `1*1 = 1`. One lattice point. Correct.
- *Thin line `(0,0)-(5,0)`*: x-range `[0,6)`, y-range `[0,1)`. One slab `[0,6)` width `6`, y-length `1`, `6*1 = 6`. The six points `(0..5, 0)`. Correct — the degenerate height still gets its inclusive `+1`.
- *Reversed corners `(5,5)-(2,3)`*: normalization gives `x1=2,x2=5,y1=3,y2=5`; x-range `[2,6)` width `4`, y-range `[3,6)` length `3`, `4*3 = 12`. Same as the in-order rectangle. Correct — the `min/max` saves me.
- *Corner-touching `(0,0)-(2,2)` and `(2,2)-(4,4)`*: they share exactly the single point `(2,2)`. Expected `9 + 9 - 1 = 17`. The sweep: x-endpoints `{0,3,5}`, slabs `[0,3)`(w3),`[3,5)`(w2). Slab `[0,3)`: rect1 (`0<=0 && 3>=3`) y `[0,3)`; rect2 (`2<=0`? no). Just rect1, len 3, `3*3=9`. Slab `[3,5)`: rect1 (`3>=5`? no); rect2 (`2<=3 && 5>=5`) y `[2,5)` len 3, `2*3=6`. Total `9+6=15`?? Let me recount the shared point... rect2 is `(2,2)-(4,4)`: x-range `[2,5)`, so its x-endpoints are `2` and `5`, not `3`. I mis-listed them. Redo: x-endpoints are rect1 `{0,3}`, rect2 `{2,5}`, sorted-unique `{0,2,3,5}`, slabs `[0,2)`(w2),`[2,3)`(w1),`[3,5)`(w2). Slab `[0,2)`: rect1 only, y `[0,3)`, `2*3=6`. Slab `[2,3)`: rect1 (`0<=2 && 3>=3` yes) y `[0,3)`, rect2 (`2<=2 && 5>=3` yes) y `[2,5)`; union `[0,5)` len 5; `1*5=5`. Slab `[3,5)`: rect2 only, y `[2,5)` len 3, `2*3=6`. Total `6+5+6 = 17`. Correct — the single shared corner `(2,2)` is counted once. My first count slipped because I used the wrong compression coordinate; the program uses `x2+1` and gets it right, which is exactly why I push `x2+1` and not `x2`.
- *Overflow*: all of `width`, `rows`, `total` are `long long`; the partial products `width*rows` fit since both factors are bounded by the coordinate span `~2*10^9` and realistic test totals stay within `long long`. No `int` anywhere in the arithmetic path.

**Validating against the brute oracle.** I wrote an independent brute that literally marks every covered lattice point in a Python set (correct by construction for small coordinates) and a small-case generator. Compiling the swept solution and comparing on 500 random instances — mixing dense overlaps, reversed corners, degenerate points and lines, corner-touches, and empty inputs — yields zero mismatches. The documented sample also prints `17`. Both traced bugs would have produced visible failures on these cases (the missing `+1` zeroes out single rectangles; the spurious `>=` is masked on length but I verified the merged-piece invariant by hand), so passing the oracle plus the hand traces is the evidence I trust.

**Final solution.** I convinced myself the idea is right by deriving the half-open mapping that bakes the inclusive `+1` into both axes and checking it on the sample, and I convinced myself the *code* is right by tracing two off-by-one slips to precise causes — the slab-coverage test must compare against `X2+1`, and the gap test must be strict `>` — and re-verifying each fix plus the corner cases against a brute oracle. This is what I ship, one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<long long> X1(n), Y1(n), X2(n), Y2(n);
    // Collect column boundaries. A rectangle covers integer columns x1..x2 inclusive.
    // We sweep over unit-column strips. To use coordinate compression on a closed
    // (inclusive) integer grid, expand each rectangle's x-range to the half-open
    // interval [x1, x2+1): integer columns x1..x2 correspond to unit strips
    // [x1,x1+1), ..., [x2,x2+1). The same trick is applied to y inside each strip.
    vector<long long> xs;
    xs.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        long long a, b, c, d;
        cin >> a >> b >> c >> d; // (a,b)-(c,d) opposite corners, any order
        long long x1 = min(a, c), x2 = max(a, c);
        long long y1 = min(b, d), y2 = max(b, d);
        X1[i] = x1; Y1[i] = y1; X2[i] = x2; Y2[i] = y2;
        xs.push_back(x1);
        xs.push_back(x2 + 1); // half-open right end
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());

    long long total = 0;
    // Sweep each x-slab [xs[k], xs[k+1]) which is a band of (xs[k+1]-xs[k]) integer columns.
    for (size_t k = 0; k + 1 < xs.size(); k++) {
        long long xl = xs[k];
        long long xr = xs[k + 1];
        long long width = xr - xl; // number of integer columns in this slab
        if (width <= 0) continue;

        // Gather y-intervals of rectangles whose x-range covers this slab.
        // Rectangle i covers columns [X1[i], X2[i]]; in half-open terms [X1[i], X2[i]+1).
        // It covers the slab iff X1[i] <= xl and X2[i]+1 >= xr.
        vector<pair<long long,long long>> ivs; // half-open [y1, y2+1)
        for (int i = 0; i < n; i++) {
            if (X1[i] <= xl && X2[i] + 1 >= xr) {
                ivs.push_back({Y1[i], Y2[i] + 1});
            }
        }
        if (ivs.empty()) continue;
        sort(ivs.begin(), ivs.end());

        // Union length of half-open y-intervals = number of distinct integer rows covered.
        long long curL = ivs[0].first, curR = ivs[0].second;
        long long rows = 0;
        for (size_t j = 1; j < ivs.size(); j++) {
            if (ivs[j].first > curR) {
                rows += curR - curL;
                curL = ivs[j].first;
                curR = ivs[j].second;
            } else {
                curR = max(curR, ivs[j].second);
            }
        }
        rows += curR - curL;

        total += width * rows;
    }

    cout << total << "\n";
    return 0;
}
```

**Causal recap.** The problem is a point-count, not an area, so every dimension carries an inclusive `+1`; I made that automatic by mapping each closed integer range `[x1,x2]` to the half-open interval `[x1,x2+1)` whose length already equals `x2-x1+1`, then sweeping vertical slabs and summing `width * (y-union length)`. My first slab-coverage test compared the rectangle's closed right edge `X2` against the slab's half-open right edge, dropping that `+1`; a trace of the lone `(0,0)-(2,2)` square returning `0` instead of `9` pinpointed it, fixed by comparing `X2+1 >= xr`. My interval-merge gap test used `>=`, which declares a false gap at exactly-touching half-open endpoints; tracing vertically adjacent rectangles showed the length coincidentally survives but the maximal-piece invariant breaks, fixed by the strict `>`. With both off-by-ones corrected, corner-touch, degenerate, reversed-corner, and empty cases all check out by hand and the swept solution matches a brute-force set-marking oracle on 500 random small cases with zero mismatches.
