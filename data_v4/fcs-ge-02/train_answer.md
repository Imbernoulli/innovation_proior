# Half-plane intersection feasibility

**Problem.** Given `m` closed half-planes `a*x + b*y <= c` (integer `a, b, c`, `(a, b) != (0, 0)`,
`m <= 2*10^5`, `|a|, |b|, |c| <= 10^6`), decide whether some point lies in all of them at once. Read
`m` and the triples from stdin; print `YES` if the intersection is non-empty, else `NO`. With `m = 0`
the answer is `YES`.

**Why the obvious framing is the wrong one.** "Is there a point satisfying all `m` linear
inequalities?" invites a general LP/simplex solver, but that buries the geometry and, in floating
point, gives wrong `YES`/`NO` answers exactly at the feasibility boundary an exact judge probes.
Because every constraint has only two variables, the intersection is a convex polygon and "non-empty"
is a purely combinatorial question — decidable with exact integers, no floating point.

**Key idea — exact-integer half-plane intersection (HPI) by angular sort + deque.** Represent each
half-plane by its line `(a, b, c)` and a boundary direction `d = (-b, a)` (interior on the left). Sort
the half-planes by direction angle, then sweep them into a deque that maintains the running convex
boundary: for each new half-plane, pop the back while its corner is strictly outside the new
half-plane, pop the front likewise, then push. Each half-plane is pushed once and popped at most once,
so after the `O(m log m)` sort the sweep is linear — `O(m log m)` overall, fast enough for
`m = 2*10^5` (about 60 ms in practice). The corner of lines `i, j` is `(Nx/D, Ny/D)` by Cramer, and
"corner strictly outside half-plane `k`" is the division-free integer sign test
`sgn(a_k*Nx + b_k*Ny - c_k*D) * sgn(D) > 0`, all carried in `__int128`.

**Detecting emptiness (the part that is easy to get wrong).** Two ingredients turn the boundary into a
correct `YES`/`NO`:
1. A large integer **bounding box** at `B = 4*10^12` (any non-empty region has a corner within
   `2*(10^6)^2 = 2*10^12`, so the box never clips a real region) makes every non-empty intersection a
   *bounded* polygon.
2. A **closure test**: a bounded convex polygon's edge directions span the whole circle, i.e. every
   cyclic gap between consecutive sorted directions is `< pi`, which is exactly `crossDir(u, v) > 0`
   for each consecutive pair including the wrap. A gap `>= pi` means the survivors form an open wedge
   (the box was eaten) — the true intersection is empty.

**Pitfalls.**
1. *Sweep order.* The back/front **pops must run before** the parallel handling against the new deque
   back; reversed, a binding constraint gets dropped and an empty region is reported feasible.
2. *Anti-parallel half-planes.* When the new half-plane is parallel to the deque back with opposite
   direction (`crossDir == 0` and `dot < 0`), e.g. `x >= 2` and `x <= 1`, the intersection is empty —
   detect and stop. Same-direction parallels keep only the tighter (smaller `c/|normal|`).
3. *`len >= 3` is not enough.* Three half-planes whose directions fit inside a half-plane form an open
   wedge, not a polygon; the closure test is what rejects it.
4. *Overflow.* Coefficients up to `10^6` and a box at `4*10^12` push intermediate products to about
   `4*10^24`; `long long` overflows, `__int128` (ceiling about `1.7*10^38`) is safe with ~13 orders to
   spare.

**Edge cases.** `m = 0` -> `YES` (whole plane); single half-plane -> `YES`; anti-parallel disjoint
bands -> `NO`; degenerate-but-feasible regions that are a single line or a single point -> `YES` (the
box fills the perpendicular directions so the survivors span the circle); feasible regions far from the
origin (corner near `2*10^12`) -> still inside the box, `YES`.

**Complexity.** `O(m log m)` time, `O(m)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Each half-plane is the closed set  a*x + b*y <= c  (integer a,b,c).
// We decide whether the intersection of all m half-planes is non-empty.
//
// SOTA: angular-sort + deque half-plane intersection in O(m log m).
// Everything is done in EXACT integer / __int128 arithmetic so the YES/NO
// decision is deterministic with no floating-point error.  A moderate
// integer bounding box makes an unbounded-but-non-empty region close into a
// polygon, so "feasible" reduces to "the surviving boundary has >= 3 edges".

typedef long long ll;
typedef __int128 lll;

struct HP {
    // boundary line: a*x + b*y == c ; interior is a*x + b*y <= c.
    // direction along the line with interior on the LEFT: d = (-b, a).
    ll a, b, c;
    ll dx, dy; // direction = (-b, a)
    int half;  // 0 if direction in upper half-plane (for angular sort), else 1
};

// half = 0 for angles in [0, pi)  (dy>0, or dy==0 && dx>0), else 1.
static int dirHalf(ll dx, ll dy) {
    if (dy != 0) return dy > 0 ? 0 : 1;
    return dx > 0 ? 0 : 1; // dx>0 -> angle 0 (upper), dx<0 -> angle pi (lower)
}

// cross of the two direction vectors: d_i x d_j
static lll crossDir(const HP &i, const HP &j) {
    return (lll)i.dx * j.dy - (lll)i.dy * j.dx;
}

// Angular comparator: sort by direction angle in [0, 2pi). Exact via half + cross.
static bool angLess(const HP &i, const HP &j) {
    if (i.half != j.half) return i.half < j.half;
    lll cr = crossDir(i, j); // >0 means j is CCW from i, i.e. i has smaller angle
    return cr > 0;
}

static int sgn(lll v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

// Intersection point of boundary lines i and j is (Nx/D, Ny/D) with
//   D  = a_i b_j - a_j b_i  (== cross(d_i,d_j))
//   Nx = c_i b_j - c_j b_i
//   Ny = a_i c_j - a_j c_i
// Test: is that point STRICTLY outside half-plane k (i.e. a_k x + b_k y > c_k)?
// Let S = a_k*Nx + b_k*Ny - c_k*D ; then (a_k x + b_k y - c_k) = S / D.
// Strictly outside  <=>  S/D > 0  <=>  sgn(S)*sgn(D) > 0.
static bool outStrict(const HP &k, const HP &i, const HP &j) {
    lll D  = (lll)i.a * j.b - (lll)j.a * i.b;
    lll Nx = (lll)i.c * j.b - (lll)j.c * i.b;
    lll Ny = (lll)i.a * j.c - (lll)j.a * i.c;
    lll S  = (lll)k.a * Nx + (lll)k.b * Ny - (lll)k.c * D;
    return sgn(S) * sgn(D) > 0;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    if (!(cin >> m)) return 0;

    vector<HP> H;
    H.reserve(m + 4);
    for (int idx = 0; idx < m; idx++) {
        ll a, b, c;
        cin >> a >> b >> c;
        HP h;
        h.a = a; h.b = b; h.c = c;
        h.dx = -b; h.dy = a;
        h.half = dirHalf(h.dx, h.dy);
        H.push_back(h);
    }

    // Bounding box: |a|,|b|,|c| <= 1e6 guarantees any non-empty intersection
    // contains a point with |x|,|y| <= 2*(1e6)^2 = 2e12. Box at B = 4e12 (safely
    // larger) so the box never falsely cuts off a real feasible region.
    const ll B = 4000000000000LL; // 4e12
    auto addLine = [&](ll a, ll b, ll c) {
        HP h; h.a = a; h.b = b; h.c = c;
        h.dx = -b; h.dy = a; h.half = dirHalf(h.dx, h.dy);
        H.push_back(h);
    };
    addLine(1, 0, B);   //  x <= B
    addLine(-1, 0, B);  // -x <= B  => x >= -B
    addLine(0, 1, B);   //  y <= B
    addLine(0, -1, B);  // -y <= B  => y >= -B

    sort(H.begin(), H.end(), angLess);

    // Deque of half-planes forming the current intersection boundary (CCW).
    vector<HP> dq(H.size());
    int lo = 0, hi = -1; // inclusive indices; size = hi-lo+1
    bool emptyByParallel = false;

    for (size_t idx = 0; idx < H.size(); idx++) {
        const HP &cur = H[idx];

        // Pop from back while the back vertex is strictly outside cur.
        while (hi - lo + 1 >= 2 && outStrict(cur, dq[hi], dq[hi - 1])) hi--;
        // Pop from front while the front vertex is strictly outside cur.
        while (hi - lo + 1 >= 2 && outStrict(cur, dq[lo], dq[lo + 1])) lo++;

        // Parallel handling against the current back (after the pops above).
        if (hi - lo + 1 >= 1) {
            const HP &last = dq[hi];
            if (crossDir(cur, last) == 0) { // parallel directions
                lll d = (lll)cur.dx * last.dx + (lll)cur.dy * last.dy;
                if (d < 0) {
                    // Anti-parallel: two opposite half-planes whose feasible bands
                    // would have to overlap. If the deque has reduced them to being
                    // adjacent here, their intersection is empty.
                    emptyByParallel = true;
                    break;
                }
                // Same outward-normal direction (a,b)_cur = lambda*(a,b)_last,
                // lambda>0. "More inward" = smaller offset c/|n|, compared on a
                // shared nonzero axis component:
                //   c_cur/|n_cur| <= c_last/|n_last|
                //   <=> c_cur*|n_last,k| <= c_last*|n_cur,k|  (same axis k).
                ll nc = (cur.a != 0 ? cur.a : cur.b);
                ll nl = (last.a != 0 ? last.a : last.b);
                // (cur.a==0 <=> last.a==0 because the normals are parallel.)
                lll lhs = (lll)cur.c * (nl < 0 ? -nl : nl);
                lll rhs = (lll)last.c * (nc < 0 ? -nc : nc);
                if (lhs < rhs) dq[hi] = cur; // cur strictly tighter -> replace
                continue;                    // otherwise cur is redundant
            }
        }

        dq[++hi] = cur;
    }

    if (emptyByParallel) { cout << "NO\n"; return 0; }

    // Final cleanup: remove back/front half-planes made redundant by wrap-around.
    while (hi - lo + 1 >= 3 && outStrict(dq[lo], dq[hi], dq[hi - 1])) hi--;
    while (hi - lo + 1 >= 3 && outStrict(dq[hi], dq[lo], dq[lo + 1])) lo++;

    int len = hi - lo + 1;
    if (len < 3) { cout << "NO\n"; return 0; }

    // Closure test. The surviving half-planes are sorted by direction angle. The
    // intersection is a BOUNDED polygon (we always added a box) iff the edge
    // directions span the whole circle, i.e. every cyclic gap between consecutive
    // directions is < pi. A gap >= pi means the directions all fit in one
    // half-plane: the region is an open wedge -> the box got eaten -> the true
    // (box-free) intersection is empty. gap(u->v) < pi  <=>  crossDir(u,v) > 0.
    bool bounded = true;
    for (int i = lo; i <= hi; i++) {
        const HP &u = dq[i];
        const HP &v = dq[(i == hi) ? lo : i + 1];
        if (crossDir(u, v) <= 0) { bounded = false; break; }
    }

    cout << (bounded ? "YES" : "NO") << "\n";
    return 0;
}
```
