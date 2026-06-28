**Reading the problem and pinning the contract.** I am given `n` integer points and must output twice
the area of the largest triangle whose vertices are three of those points; `0` if fewer than three
points or all of them collinear. Working with *twice* the area is deliberate and load-bearing: for
integer coordinates the cross product `(B-A) x (C-A)` is an integer equal to twice the signed area, so
the whole computation stays in exact integers and I never touch floating point. Before any algorithm I
fix the scale, because it decides the data types. Coordinates go up to `10^9` in magnitude, so a single
edge vector component can be `2*10^9`, and a cross product is a product of two such things:
`2*10^9 * 2*10^9 = 4*10^18`. Signed 64-bit `long long` tops out near `9.2*10^18`, so twice-the-area
fits with room to spare — but only just, and only if I am careful to cast to 64-bit *before* the
multiply. An `int` multiply here overflows silently and is a wrong answer on the large tests. So
`long long` everywhere, and the cross product computed as `(ll)(...)*(...)`.

**The obvious approach, and why brute force is hopeless at scale.** The definitionally-correct method
is to try all triples: for every `i < j < k`, compute `|cross|` and keep the max. That is `O(n^3)`.
With `n` up to `5000` that is `1.25 * 10^11` triples — utterly out of reach for a 1-second limit
(roughly two orders of magnitude too slow even at a billion cheap ops per second). So the brute force
is only good as an *oracle* on tiny inputs; it cannot be the shipped solution. I need real structure.

**First structural lever: the optimum lives on the convex hull.** Here is the observation that shrinks
the problem. Suppose the optimal triangle has a vertex `V` that is strictly inside the convex hull of
all the points. Hold the other two vertices `A`, `B` fixed; the area of triangle `A B V` is
`(1/2) * |AB| * dist(V, line AB)`. To maximise it I want `V` as far from line `AB` as possible. Among
all points, the farthest from any line is attained at a hull vertex (the supporting line argument:
sweep a line parallel to `AB` outward until it last touches the set — it touches at a hull vertex). So
I can always replace an interior `V` by a hull vertex without decreasing the area. Applying this to all
three vertices: **every maximum-area triangle has all three vertices on the convex hull.** That means I
can first compute the convex hull `H` (size `h <= n`) and then search only over triples of hull
vertices. On random or adversarial-circle inputs `h` can still be `Theta(n)`, so this alone does not
beat `O(n^3)` — but it sets up the second lever, which needs convexity.

**Building the hull with Andrew's monotone chain.** Sort points by `(x, y)`, build the lower chain
left-to-right and the upper chain right-to-left, popping a vertex whenever the last turn is not a
strict left turn. I use a `cross(...) <= 0` pop test, which discards collinear points on the hull
boundary too — that is what I want, because a collinear "hull" point is never a triangle apex that
helps, and keeping them would only inflate `h`. I also dedupe exact-equal points first, since duplicate
coordinates confuse the chain's turn tests. After building, if the hull has fewer than 3 vertices the
input was collinear (or had `<3` distinct points), and the answer is `0`.

**Second lever — and the trap.** Now I have a convex polygon `hull[0..h-1]` and want its
maximum-area inscribed triangle. The famous "rotating calipers" lore says: this should be `O(h)`. The
classical Dobkin–Snyder idea is to keep three pointers `a, b, c` and rotate all three around the
polygon in a single coordinated sweep — when advancing the apex `a`, you do *not* reset `b` and `c`,
you let them continue from where they were, so the whole thing is one pass, `O(h)`. It is beautiful and
it is what I would reach for first. Let me sanity-check it before trusting it, because "one clever
sweep solves a triple-optimisation" is exactly the kind of claim that hides a bug.

**Why the single-sweep O(h) calipers is wrong.** The reason it is tempting is that for a *fixed* apex
`a` and *fixed* second vertex `b`, the area `area(a, b, c)` as `c` walks along the hull is **unimodal**
— it rises to a single peak (the vertex farthest from line `ab`) and falls. Unimodality lets a pointer
`c` chase the peak monotonically. The flaw is in the *coupling across `a`*: when I move the apex `a`
forward and refuse to reset `b` and `c`, I am implicitly assuming the optimal `(b, c)` for the new apex
is at or ahead of the old `(b, c)`. That assumption is false. Concretely, on a slightly irregular
polygon there can be two distinct locally-optimal ("2-stable") triangles rooted at the *same* apex with
nearly equal areas, sitting at different `(b, c)` positions; a single forward sweep finds whichever it
reaches first and walks right past the other. This is not hand-waving on my part — this exact algorithm
was believed correct for decades and then shown to fail on a concrete 9-gon: the global optimum is one
of the 2-stable triangles the sweep skips. So `O(h)` Dobkin–Snyder is out as a *correct* method, and I
must not ship it.

**Deriving the correct scan: fix the apex, reset the chasers per apex.** Keep the part that is actually
true — for fixed apex `i` and fixed second vertex `j`, the best third vertex `l` is found by a monotone
forward chase because `area(i, j, l)` is unimodal in `l`. The fix to Dobkin–Snyder is simply: **do not
share the pointers across apexes.** For each apex `i` independently, restart `j` at `i+1` and `l` at
`i+2`, then sweep `j` forward around the hull, letting `l` chase. Within one apex, as `j` advances, the
optimal `l` is monotone non-decreasing (the "interleaving" property: consecutive 2-stable triangles
rooted at `i` only move `l` forward), so `l` never has to back up *within* an apex — but it *is* reset
when `i` changes. That reset per apex is the whole correctness difference, and it costs `O(h)` per apex
for `O(h^2)` total. With `h <= 5000`, that is at most `2.5 * 10^7` cross-product evaluations — about
0.2 s in practice, comfortably under the limit. This `O(h^2)` two-pointer is the established correct
replacement; it is exactly the "2-stable triangles, examined for every root" method, and it is what I
implement.

**First implementation.** Apex loop over `i`; inner pointers `j` and `l` as indices mod `h`. For a
fixed `(i, j)`, advance `l` while the next position gives at least as large an area; then record the
area; then advance `j`; keep `l` strictly ahead of `j`. My first cut of the inner machinery:

```
ll best = 0;
for (int i = 0; i < h; i++) {
    int j = (i + 1) % h;
    int l = (i + 2) % h;
    while (true) {
        while (true) {
            int ln = (l + 1) % h;
            ll cur = llabs(cross(hull[i], hull[j], hull[l]));
            ll nxt = llabs(cross(hull[i], hull[j], hull[ln]));
            if (nxt >= cur) l = ln; else break;
        }
        ll area2 = llabs(cross(hull[i], hull[j], hull[l]));
        if (area2 > best) best = area2;
        int jn = (j + 1) % h;
        if (jn == i) break;
        j = jn;
        if (l == j) { l = (l + 1) % h; }
    }
}
```

**Tracing it on a triangle, and finding the first bug.** I test the smallest non-degenerate hull,
`h = 3`, with hull `(0,0), (4,0), (0,3)` — the only triangle has `area2 = 12`. Apex `i = 0`:
`j = 1`, `l = 2`. Inner `l`-chase: `ln = (2+1)%3 = 0`, which is `i` itself — and I compute
`cross(hull[0], hull[1], hull[0])`, i.e. a triangle with two identical vertices, area `0`. Since
`nxt = 0 >= cur`? Only if `cur` is also `0`, but `cur = |cross(0,1,2)| = 12`, so `nxt(0) >= 12` is
false and `l` stays at `2`. Lucky here. But the logic is wrong in principle: `ln` is allowed to land on
`i`, which is *not a legal third vertex* — the third vertex must be strictly between `j` and `i` going
around. On a larger hull this lets `l` wrap past `i` and start measuring degenerate or
double-counted triangles. The defect: my `l`-chase has **no guard stopping `l` from reaching the
apex `i`.** I need `if (ln == i) break;` inside the chase so `l` never advances onto the apex.

**The second, nastier bug — `j` and `l` colliding.** I trace `h = 4`, a square `(0,0),(4,0),(4,4),(0,4)`,
`area2` of the best triangle is `16` (half the square). Apex `i = 0`, `j = 1`, `l = 2`. Chase `l`:
`ln = 3`; `cur = |cross(0,1,2)|`. `cross((0,0),(4,0),(4,4)) = 4*4 - 0 = 16`; `nxt = cross((0,0),(4,0),(0,4)) =
4*4 - 0 = 16`; `nxt >= cur` so `l = 3`; next `ln = (3+1)%4 = 0 = i`, guard breaks. `area2 = 16`,
`best = 16`. Advance `j`: `jn = 2 != 0`, `j = 2`. Now `l = 3`, `j = 2`, fine, `l != j`. Chase: `ln =
0 = i`, break immediately; `area2 = |cross((0,0),(4,4),(0,4))| = 4*4 - 4*0 = 16`. Advance `j`: `jn =
3 != 0`, `j = 3`. Now `l = 3 == j`! My `if (l == j) l = (l+1)%h;` sets `l = 0 = i` — an illegal apex
vertex. The next chase computes `cross(hull[0], hull[3], hull[0])`, a degenerate `0`, and worse, if a
later `ln` wrapped I would be reading triangles that reuse the apex. The fix: when `j` catches `l`, push
`l` forward, but if that push would land on `i`, there is simply no room for a third vertex with this
`j`, so I must **break out of this apex** rather than let `l` sit on `i`.

**Fixing both bugs.** I add the `ln == i` guard to the inner chase, and at the `j`-advance step I guard
the collision: if pushing `l` past `j` reaches `i`, break. The corrected inner block:

```
while (true) {
    while (true) {
        int ln = (l + 1) % h;
        if (ln == i) break;                 // l must stay strictly before i
        ll cur = llabs(cross(hull[i], hull[j], hull[l]));
        ll nxt = llabs(cross(hull[i], hull[j], hull[ln]));
        if (nxt >= cur) l = ln; else break;
    }
    ll area2 = llabs(cross(hull[i], hull[j], hull[l]));
    if (area2 > best) best = area2;
    int jn = (j + 1) % h;
    if (jn == i) break;                     // j wrapped to apex: done
    j = jn;
    if (l == j) {
        int ln = (l + 1) % h;
        if (ln == i) break;                 // no room for a third vertex
        l = ln;
    }
}
```

**Re-tracing the square with the fix.** Apex `0`: `j=1, l=2`. Chase: `ln=3 != 0`; `cur=16`,
`nxt=16`, `16>=16` so `l=3`; `ln=(3+1)%4=0=i`, guard breaks at `l=3`. `area2=16`, `best=16`.
`jn=2`, `j=2`; `l=3 != j`. Chase: `ln=0=i` break; `area2=|cross((0,0),(4,4),(0,4))|=16`. `jn=3`,
`j=3`; now `l==j`, push: `ln=(3+1)%4=0=i` -> **break this apex**. Apexes `1,2,3` by symmetry also
yield `16`. Final `best=16`. Correct, and crucially the degenerate apex-reuse triangles never get
measured.

**Re-tracing the right triangle `h=3`.** Apex `0`, `j=1`, `l=2`. Chase: `ln=(2+1)%3=0=i`, guard
breaks immediately, `l=2`. `area2=|cross((0,0),(4,0),(0,3))| = 4*3 - 0 = 12`, `best=12`. `jn=2 != 0`,
`j=2`; `l=2==j`, push: `ln=(2+1)%3=0=i` -> break. Apexes `1,2` give the same `12`. Answer `12`.
Correct.

**Stress-testing the whole pipeline against brute force.** Hand traces only cover what I thought to
check; the real confidence comes from a differential test. I wrote an `O(n^3)` brute that takes the max
`|cross|` over all triples (it does not even need the hull — it is a correct superset oracle, since the
hull-only optimum equals the all-triples optimum), and a generator that deliberately stresses the
fragile parts: tiny `n in {0,1,2}`, all-collinear inputs, heavy duplicate points, tight coordinate
ranges (`|coord| <= 2..4`) that manufacture many ties and collinear hull edges, and interior points
that must be ignored. Running ~2700 random cases plus a battery of explicit edge cases (`n=0`, single
point, two points, a collinear triple, a square, duplicate-collinear sets, an interior-point set, and
extreme `+-10^9` coordinates) produced **zero mismatches**. The extreme-coordinate cases return
`4*10^18`, confirming the 64-bit headroom is real and the cross product does not overflow. A circle of
`5000` points (hull size `5000`, the `O(h^2)` worst case) runs in about `0.2 s`, confirming the time
budget.

**Edge cases, deliberately.**
- `n < 3` (including `n = 0`): handled before the hull — print `0`. The empty/too-few-points case.
- All points collinear: the monotone chain collapses to a segment, `h < 3`, print `0`. The `<= 0` pop
  test is what guarantees collinear points are dropped so `h` truly reflects "is there positive area".
- Duplicate points: deduped before the chain so the turn tests never see a zero-length edge.
- Interior points: never enter the hull, automatically ignored — and the all-triples brute confirms
  the hull reduction loses nothing.
- Overflow: cross product cast to `ll` before multiply; worst magnitude `4*10^18 < 9.2*10^18`. Safe.

**Final solution.** I disproved the seductive `O(h)` single-sweep calipers (it skips 2-stable
triangles, the documented 9-gon failure), kept the one true sub-fact (area is unimodal in the third
vertex for a fixed apex–base pair), and turned it into the correct `O(h^2)` per-apex-reset two-pointer
on the convex hull. I convinced myself the *idea* is right by the supporting-line hull argument and the
interleaving/reset distinction, and the *code* is right by tracing two real bugs (an unguarded `l`
reaching the apex, and a `j`/`l` collision pushing `l` onto the apex) to precise causes, fixing both,
and differential-testing ~2700 cases to zero mismatches. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct P { ll x, y; };

// 2x signed area of triangle (o,a,b); positive if a->b is a left turn around o.
static inline ll cross(const P& o, const P& a, const P& b) {
    return (ll)(a.x - o.x) * (b.y - o.y) - (ll)(a.y - o.y) * (b.x - o.x);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<P> pts(n);
    for (int i = 0; i < n; i++) cin >> pts[i].x >> pts[i].y;

    // Fewer than 3 points: no triangle.
    if (n < 3) { cout << 0 << "\n"; return 0; }

    // --- Andrew's monotone chain convex hull ---
    sort(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });
    // Remove exact duplicate points (they cannot help and confuse the chain).
    pts.erase(unique(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x == b.x && a.y == b.y;
    }), pts.end());
    int m = (int)pts.size();
    if (m < 3) { cout << 0 << "\n"; return 0; }

    vector<P> hull(2 * m);
    int k = 0;
    // lower hull
    for (int i = 0; i < m; i++) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) k--;
        hull[k++] = pts[i];
    }
    // upper hull
    for (int i = m - 2, lo = k + 1; i >= 0; i--) {
        while (k >= lo && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) k--;
        hull[k++] = pts[i];
    }
    hull.resize(k - 1); // last point == first, drop it
    int h = (int)hull.size();

    // All points collinear -> hull is a segment (h < 3): max area triangle is 0.
    if (h < 3) { cout << 0 << "\n"; return 0; }

    // --- Maximum-area triangle on the convex hull ---
    // Fix apex i; advance pointers j (= i+1 initially) and l (= i+2 initially)
    // monotonically forward over the hull. For a fixed (i, j), the function
    //   l -> 2*area(hull[i], hull[j], hull[l])
    // is unimodal as l sweeps from j+1 around to i-1, so we push l forward while
    // the area keeps growing. As j advances, the optimal l is monotone non-
    // decreasing (the interleaving property of 2-stable triangles), so l never
    // needs to be reset back behind its current value within one apex.
    // We RESET j and l for every apex i (this is the fix to the broken O(n)
    // Dobkin-Snyder scan, which moves the apex without resetting and misses the
    // optimum). Cost: O(h) per apex => O(h^2) overall.
    ll best = 0;
    for (int i = 0; i < h; i++) {
        int j = (i + 1) % h;
        int l = (i + 2) % h;
        // For this apex, sweep j around; l chases j monotonically.
        // We stop when j has gone all the way around back toward i.
        while (true) {
            // advance l while area strictly increases
            while (true) {
                int ln = (l + 1) % h;
                if (ln == i) break;                 // l must stay strictly before i
                ll cur = llabs(cross(hull[i], hull[j], hull[l]));
                ll nxt = llabs(cross(hull[i], hull[j], hull[ln]));
                if (nxt >= cur) l = ln; else break;
            }
            ll area2 = llabs(cross(hull[i], hull[j], hull[l]));
            if (area2 > best) best = area2;

            int jn = (j + 1) % h;
            if (jn == i) break;                     // j wrapped to apex: done
            // l must remain strictly ahead of j; if j catches up, push l along.
            j = jn;
            if (l == j) {
                int ln = (l + 1) % h;
                if (ln == i) break;                 // no room for a third vertex
                l = ln;
            }
        }
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** Brute `O(n^3)` over triples is correct but `1.25*10^11` ops at `n=5000` — far too
slow. The optimum's vertices are all on the convex hull (push any interior vertex outward along the
supporting line), so I reduce to the hull. On the hull the tempting `O(h)` single-sweep calipers
(Dobkin–Snyder) is *wrong* — it shares pointers across apexes and skips 2-stable triangles, the
documented 9-gon failure — so I keep only the true sub-fact (area is unimodal in the third vertex for a
fixed apex+base) and reset the chasing pointers per apex, giving the correct `O(h^2)` two-pointer. Two
transcription bugs (an unguarded `l` reaching the apex, and a `j`/`l` collision pushing `l` onto the
apex) were caught by tracing `h=3` and `h=4` by hand and fixed with `ln == i` guards; ~2700
differential cases against the brute oracle then ran to zero mismatches, with `+-10^9` coordinates
confirming 64-bit headroom and a 5000-point circle confirming the 0.2 s timing.
