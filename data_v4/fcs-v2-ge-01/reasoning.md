I need twice the area of the largest triangle on `n` integer points, or `0` when there are fewer
than three points or they are all collinear. Two things about this problem decide everything before I
touch an algorithm. First, "twice the area" is not a formatting quirk — it is what keeps the
computation exact: `(B-A) x (C-A)` is an integer equal to twice the signed area, so I stay in integers
and never touch floating point. Second, the arithmetic sits right against the 64-bit ceiling.
Coordinates reach `10^9`, an edge component reaches `2*10^9`, and a cross product multiplies two of
those: `2*10^9 * 2*10^9 = 4*10^18`. Signed `long long` tops out near `9.2*10^18`, so twice-the-area
fits — but only if I cast each factor to 64-bit *before* the multiply. An `int` product overflows
silently and is wrong exactly on the extreme tests. So `long long` throughout, cross product formed as
`(ll)(...)*(...)`, reading `n` then the pairs from stdin and printing a single integer line.

The definition — try every triple `i < j < k` and keep the max `|cross|` — is `O(n^3)`. At `n = 5000`
that is `1.25 * 10^11` triples, two orders of magnitude past a 1 s budget. Good only as an oracle on
tiny inputs; I need structure.

The structure is that the optimum lives on the convex hull. Suppose an optimal triangle has a vertex
`V` strictly inside the hull. Fix the other two vertices `A`, `B`; the area is
`(1/2) * |AB| * dist(V, line AB)`, maximised by pushing `V` as far from line `AB` as possible — and the
farthest point of the set from any line is a hull vertex (sweep a line parallel to `AB` outward until
it last touches the set). So `V` can be moved to a hull vertex without shrinking the area; applying
that to all three vertices, every maximum-area triangle has all three vertices on the hull. I reduce
the `n` points to their hull `H` (size `h`) and search only hull triples. On points near a circle `h`
can still be `Theta(n)`, so the reduction alone does not beat `O(n^3)` — but it hands me a convex
polygon, which the next step needs.

I build `H` with Andrew's monotone chain: sort by `(x, y)`, build the lower and upper chains, popping
whenever the last turn is not a strict left turn. I dedupe exact-equal points first — duplicate
coordinates give zero-length edges that confuse the turn tests — and I use a `cross(...) <= 0` pop test
so collinear boundary points are discarded. A collinear hull point never helps as an apex, and
dropping it makes `h < 3` cleanly mean "no positive-area triangle exists." If `h < 3` after building,
the input was collinear or had fewer than three distinct points, and the answer is `0`.

Now the polygon-side problem: the maximum-area inscribed triangle of a convex polygon. The folklore
says `O(h)` via rotating calipers — the Dobkin–Snyder idea: keep three pointers and rotate all three
around the polygon in one coordinated sweep, and crucially, when the apex advances you do *not* reset
the two chasers, you let them continue from where they sit. It is the first thing I would reach for.
But this exact algorithm was believed correct for decades and then shown to fail on a concrete 9-gon,
so I need to understand *why* before writing a line of it.

The flaw is a coupling I would not have suspected. For a fixed apex `a` and fixed second vertex `b`,
`area(a, b, c)` as `c` walks the hull is unimodal — it climbs to the vertex farthest from line `ab` and
falls — so a single pointer can chase that peak monotonically. That part is sound. The failure is
across `a`: refusing to reset `b, c` when the apex moves assumes the optimal `(b, c)` for the new apex
lies at or ahead of the old one. On a slightly irregular polygon two distinct locally-optimal
("2-stable") triangles can be rooted at the same apex, at different `(b, c)` positions, with nearly
equal areas; the forward sweep locks onto whichever it reaches first and walks straight past the other
— which on the 9-gon is the global optimum. So the `O(h)` sweep is out.

The repair keeps the true sub-fact and drops the false assumption: for each apex `i` *independently*,
restart `j` at `i + 1` and `l` at `i + 2`, then sweep `j` forward with `l` chasing. Within one apex, as
`j` advances the optimal `l` is monotone non-decreasing (consecutive 2-stable triangles only move `l`
forward), so `l` never backs up — but it *is* reset when the apex changes. That per-apex reset is the
entire correctness difference, and it costs `O(h)` per apex, `O(h^2)` total. With `h <= 5000` that is
at most `2.5 * 10^7` cross products, well inside a second.

The delicate part is the index bookkeeping: the third vertex `l` must stay strictly between `j` and the
apex `i` going around, and both pointers move mod `h`. That invariant is exactly the kind a wrap-around
loop violates, so I write the inner machinery and trace it on the smallest hulls to see where it
breaks.

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

On a triangle hull `(0,0), (4,0), (0,3)`, apex `i = 0`, `j = 1`, `l = 2`: the `l`-chase computes
`ln = (2+1)%3 = 0`, which is the apex itself, and `cross(hull[0], hull[1], hull[0])` is a triangle with
a repeated vertex, area `0`. Here `0 >= 12` is false so `l` stays put by luck — but the loop was
*willing* to step `l` onto the apex, and on a larger hull that lets `l` wrap past `i` and start
measuring degenerate triangles. So the `l`-chase needs `if (ln == i) break;`.

The square `(0,0), (4,0), (4,4), (0,4)` exposes the second bug. Apex `0` sweeps correctly to `l = 3`,
`j = 2`, but advancing `j` to `3` makes `l == j`; my `l = (l+1)%h` then sets `l = 0 = i`, the apex
again. The right handling: when `j` catches `l`, push `l` forward, but if that push would land on `i`
there is simply no room for a third vertex with this `j`, so break out of the apex. Both fixes are the
same `ln == i` test in two places:

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

Re-running the square, apex `0` now yields `16` and stops cleanly when the push would hit the apex; the
three symmetric apexes give `16` too, and the degenerate apex-reuse triangles are never measured.

Hand traces only cover what I thought to check, so I settle it with a differential test: an `O(n^3)`
brute over all triples — a correct superset oracle, since it needs no hull and the all-triples max
equals the hull-only max — run against the solution on a generator aimed at the fragile parts: tiny
`n in {0,1,2}`, all-collinear sets, heavy duplicates, coordinate ranges as tight as `|coord| <= 2` to
manufacture ties and collinear edges, and interior points that must be ignored. Several hundred random
cases plus explicit edges (`n = 0, 1, 2`, a collinear triple, the square, duplicate-collinear sets, an
interior-point set, and `+-10^9` corners) produce zero mismatches. The extreme corners return
`4 * 10^18`, confirming the 64-bit headroom is real and the cross product does not overflow. The
`O(h^2)` worst case is a hull of 5000 near-circular points: `~2.5 * 10^7` cross products, which the
operation count puts at a couple of tenths of a second, inside the limit.

So the shipped program is one self-contained file: read `n` and the pairs; return `0` immediately for
`n < 3`; dedupe and sort, then build the hull with the `<= 0` pop test and bail to `0` if `h < 3`; run
the per-apex-reset two-pointer with the `ln == i` guards; print `best`. The brute stays only as the
offline oracle.
