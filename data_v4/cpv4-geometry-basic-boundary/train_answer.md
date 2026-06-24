**Problem.** Given `n` closed axis-aligned rectangles (each as two opposite corners with integer coordinates in `[-10^9, 10^9]`, in arbitrary order), count the number of distinct **integer lattice points** covered by their union — every point inside or on the boundary of at least one rectangle. Read the rectangles from stdin, print the count. `0 <= n <= 2*10^5`.

**Key idea — half-open expansion + slab sweep.** This is a point count, not an area, so a rectangle spanning columns `x1..x2` owns `x2 - x1 + 1` columns. Bake that inclusive `+1` into the geometry by mapping each closed integer range `[x1, x2]` to the half-open real interval `[x1, x2+1)`: its length `(x2+1) - x1` *is* the count, and the unit strip `[x, x+1)` is the single integer column `x`. Then coordinate-compress the x-endpoints `{x1, x2+1}`, sweep vertical slabs `[xs[k], xs[k+1])`, and for each slab sum the union length of the half-open y-intervals `[y1, y2+1)` of the rectangles that cover the slab. The slab contributes `width * (y-union length)`, where `width = xs[k+1] - xs[k]` counts the integer columns it represents. Both axes use `+1`, so the inclusive boundary — and every shared edge or corner between rectangles — is handled uniformly and counted exactly once. This is `O(n^2 log n)`, fine for the intended scale, and survives `10^9` coordinates that a literal grid never could.

**Pitfalls (both are off-by-ones at an inclusive/exclusive boundary).**
1. *Slab-coverage test.* A rectangle covers slab `[xl, xr)` iff `X1 <= xl` **and** `X2 + 1 >= xr` — the rectangle's *half-open* right end is `X2+1`, not `X2`. Writing `X2 >= xr` drops the `+1` on the boundary column and makes a single rectangle disown its rightmost slab; a `(0,0)-(2,2)` square then prints `0` instead of `9`. A trace of that lone square exposes it.
2. *Interval-merge gap test.* Two half-open intervals that share an endpoint (`[0,2)` and `[2,5)`) are adjacent and must merge into `[0,5)`. The correct "real gap" predicate is strict `ivs[j].first > curR`; using `>=` declares a false gap at the touch point and splits a contiguous run, breaking the maximal-piece invariant.

**Edge cases.** `n = 0` -> empty `xs`, loop never runs, prints `0`. Single point `(p,p)-(p,p)` -> `1`. Thin line `(0,0)-(5,0)` -> `6` (degenerate height still gets its `+1`). Reversed corners are fixed by `x1=min, x2=max` per axis. Corner-touching rectangles share their single corner point, counted once by the interval union. All accumulators are `long long`; `int` is a silent wrong-answer.

**Complexity.** `O(n^2 log n)` time (per slab, scan all rectangles and sort the active y-intervals), `O(n)` extra space.

**Code.**

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
