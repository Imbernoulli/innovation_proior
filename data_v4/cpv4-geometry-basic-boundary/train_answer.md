**Problem.** Given `n` closed axis-aligned rectangles (each as two opposite corners with integer coordinates in `[-10^9, 10^9]`, in arbitrary order), count the number of distinct **integer lattice points** covered by their union — every point inside or on the boundary of at least one rectangle. Read the rectangles from stdin, print the count. `0 <= n <= 2*10^5`.

**Key idea — half-open expansion + event sweep over a cover-count segment tree.** This is a point count, not an area, so a rectangle spanning columns `x1..x2` owns `x2 - x1 + 1` columns. Bake that inclusive `+1` into the geometry by mapping each closed integer range `[x1, x2]` to the half-open real interval `[x1, x2+1)`: its length `(x2+1) - x1` *is* the count, and the unit strip `[x, x+1)` is the single integer column `x`. Then sweep left to right over events: each rectangle activates its half-open y-interval `[y1, y2+1)` at `x = x1` and deactivates it at `x = x2 + 1`. A segment tree built over the elementary y-segments between consecutive compressed values of `{y1, y2+1}` maintains the union length of the active intervals under these updates: each node keeps a count of the intervals fully covering its segment and a covered length that is the whole segment `ys[hi] - ys[lo]` whenever the count is positive, otherwise the sum of its children's (`0` at a leaf). A deactivation retraces exactly the path of its activation, so counts never go negative and no lazy propagation is needed. Coverage is constant between consecutive event abscissas `x` and `x'`, so each band contributes `(x' - x) * cov[root]` — integer columns times integer rows. Both axes use `+1`, so the inclusive boundary — and every shared edge or corner between rectangles — is handled uniformly and counted exactly once. Sorting plus `2n` tree updates of `O(log n)` each is `O(n log n)`, comfortably inside the 2-second limit at `n = 2*10^5`, and the compression survives `10^9` coordinates that a literal grid never could. The tempting shortcut of rescanning all `n` rectangles inside each of the up-to-`2n` slabs is `O(n^2 log n)` — on the order of `10^11` operations at full scale; it already takes over 3 seconds at `n = 10^4` and is hopeless at `n = 2*10^5`, which is exactly why the active set must be maintained incrementally.

**Pitfalls (both are off-by-ones at an inclusive/exclusive boundary).**
1. *Deactivation abscissa.* A rectangle's half-open right end is `X2 + 1`, not `X2`, so its removal event belongs at `x = X2 + 1`. Deactivating at `X2` drops the `+1` on the boundary column and disowns the rightmost column of every rectangle; a lone `(0,0)-(2,2)` square then prints `6` instead of `9`. A trace of that lone square exposes it.
2. *Half-open leaf range.* The y-interval `[y1, y2+1)` must update the compressed leaf range `[idx(y1), idx(y2+1))` — exclusive on the right. Making the right index inclusive annexes the elementary segment just above the rectangle: two isolated points `(0,0)` and `(0,2)` come out as `3` instead of `2`, the uncovered row between them silently counted. Touching intervals, by contrast, need no special predicate here: two half-open intervals that share an endpoint (`[0,2)` and `[2,5)`) fuse automatically, because coverage is accounted per elementary segment rather than by merging sorted endpoint pairs — the strict-vs-non-strict gap test (`>` vs `>=`) that plagues an endpoint-merge scan never has to be written.

**Edge cases.** `n = 0` -> no events, the program prints `0` and returns before the tree is ever built. Single point `(p,p)-(p,p)` -> `1`. Thin line `(0,0)-(5,0)` -> `6` (degenerate height still gets its `+1`). Reversed corners are fixed by `x1=min, x2=max` per axis. Corner-touching rectangles share their single corner point, counted once by the covered-length accounting. All accumulators are `long long`; `int` is a silent wrong-answer.

**Complexity.** `O(n log n)` time (sort the `2n` events and `2n` y-endpoints, then one `O(log n)` tree update per event), `O(n)` space for the events, the compressed axis, and the tree.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count distinct integer lattice points covered by a union of closed
// axis-aligned rectangles. Closed integer ranges are expanded to half-open
// real intervals: [x1, x2] -> [x1, x2+1) and [y1, y2] -> [y1, y2+1), so the
// "area" of the half-open geometry *is* the lattice-point count.
//
// Event sweep over x: at x = x1 a rectangle activates its y-interval
// [y1, y2+1), at x = x2+1 it deactivates it. A segment tree over the
// elementary y-segments (between consecutive compressed values of
// {y1, y2+1}) maintains the total covered y-length under these updates.

static int m;                       // number of elementary y-segments
static vector<long long> ys;        // compressed y-boundaries, size m+1
static vector<int> cnt;             // cover count per node
static vector<long long> cov;       // covered length per node

static void update(int node, int lo, int hi, int l, int r, int delta) {
    if (r <= lo || hi <= l) return;
    if (l <= lo && hi <= r) {
        cnt[node] += delta;
    } else {
        int mid = (lo + hi) / 2;
        update(2 * node, lo, mid, l, r, delta);
        update(2 * node + 1, mid, hi, l, r, delta);
    }
    if (cnt[node] > 0)
        cov[node] = ys[hi] - ys[lo];          // fully covered at this level
    else
        cov[node] = (hi - lo == 1) ? 0 : cov[2 * node] + cov[2 * node + 1];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    struct Event { long long x; int type; long long yl, yr; }; // [yl, yr) half-open
    vector<Event> ev;
    ev.reserve(2 * n);
    ys.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        long long a, b, c, d;
        cin >> a >> b >> c >> d;               // opposite corners, any order
        long long x1 = min(a, c), x2 = max(a, c);
        long long y1 = min(b, d), y2 = max(b, d);
        ev.push_back({x1, +1, y1, y2 + 1});    // activate at left column
        ev.push_back({x2 + 1, -1, y1, y2 + 1}); // deactivate past right column
        ys.push_back(y1);
        ys.push_back(y2 + 1);                  // half-open upper end
    }
    if (n == 0) { cout << 0 << "\n"; return 0; }

    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());
    m = (int)ys.size() - 1;                    // elementary segments [ys[i], ys[i+1])
    cnt.assign(4 * m, 0);                      // n >= 1 forces m >= 1 (y2+1 > y1)
    cov.assign(4 * m, 0);

    sort(ev.begin(), ev.end(),
         [](const Event& p, const Event& q) { return p.x < q.x; });

    long long total = 0;
    long long prevX = ev[0].x;
    size_t i = 0;
    while (i < ev.size()) {
        long long x = ev[i].x;
        total += (x - prevX) * cov[1];         // columns [prevX, x) x covered rows
        while (i < ev.size() && ev[i].x == x) {
            int l = (int)(lower_bound(ys.begin(), ys.end(), ev[i].yl) - ys.begin());
            int r = (int)(lower_bound(ys.begin(), ys.end(), ev[i].yr) - ys.begin());
            update(1, 0, m, l, r, ev[i].type); // half-open leaf range [l, r)
            i++;
        }
        prevX = x;
    }

    cout << total << "\n";
    return 0;
}
```
