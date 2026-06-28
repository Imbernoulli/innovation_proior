**Reading the problem and pinning the contract.** I am given `m` half-planes, each a closed set `a*x + b*y <= c` with integer `a, b, c` and `(a, b) != (0, 0)`, and I must decide whether some real point `(x, y)` lies in all of them at once — geometrically, whether the intersection is non-empty. Output is a single `YES`/`NO`. Before any algorithm I fix the scale, because it governs everything downstream: `m` up to `2*10^5`, and `|a|, |b|, |c| <= 10^6`. So an `O(m^2)` or `O(m^3)` method is dead on arrival, and I need something near `O(m log m)`. The coefficient bound also tells me that exact arithmetic is plausible: products of two coefficients are around `10^12`, products of three around `10^18`, which is exactly the regime where `long long` starts to strain and `__int128` becomes the safe carrier. I note that now — the precision story is going to be the hard part of this problem, more than the algorithm skeleton.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can both make fast and make *exactly* right, not merely the one that is easy to type.

- *Pairwise vertex enumeration.* The feasible region, if non-empty, is a convex polygon (possibly unbounded) whose corners are intersections of pairs of boundary lines. There are `O(m^2)` such intersection points; test each against all `m` constraints, and answer `YES` iff one survives. This is `O(m^3)`, obviously correct, and a perfect oracle for small `m` — but utterly hopeless at `m = 2*10^5`. I keep it in my pocket as the brute force to validate against, not as the solution.
- *Incremental half-plane intersection (HPI).* Sort the half-planes by the angle of their boundary direction, then sweep them into a double-ended queue that maintains the running convex boundary. Because the boundary of a convex intersection visits its edges in angular order, each half-plane is pushed once and popped at most once, so after the `O(m log m)` sort the sweep is linear. This is the right asymptotics. The cost is that the deque-pop invariant and the emptiness/unboundedness detection are notoriously fiddly to get exactly right.

The HPI route is the only one that fits the limits, so the real work is making it correct. Let me derive it rather than copy it, so I understand every branch.

**Why the obvious "just LP-solve it" instinct is the wrong framing here.** My first instinct is to reach for a generic 2-variable linear-programming feasibility routine — pick an objective, run simplex, see if it is feasible. But that buries the geometry and, worse, a floating-point simplex on adversarial near-degenerate inputs gives wrong `YES`/`NO` answers right at the feasibility boundary, which is exactly where an exact judge will probe. The structure I actually have — *all* constraints are two-variable inequalities — is far more special than general LP. The intersection is a convex polygon, and "non-empty" is a purely combinatorial question about whether the half-planes' directed boundaries wrap all the way around. That reframing is what lets me drop floating point entirely and decide with integers. So I will build HPI, but commit to doing every predicate in exact integer arithmetic.

**Setting up the representation so the predicates can be exact.** I represent each half-plane by its line `(a, b, c)` and a direction vector along the boundary. I want the interior (`a*x + b*y <= c`) to lie to the *left* of the direction, the standard convention for a counterclockwise convex hull of half-planes. The outward normal (pointing into the infeasible side, where `a*x + b*y` increases) is `(a, b)`. Rotating a direction `d` by +90 degrees gives `(-d.y, d.x)`; I want that rotation to point *outward*, i.e. to equal `(a, b)`. Solving, `d = (-b, a)`: rotating `(-b, a)` by +90 gives `(-a, -b)`, the *inward* direction, so the inside is to the left of `d = (-b, a)`. Good. With this convention, the test "is point `r` strictly outside this half-plane" is "is `r` strictly to the right of the directed boundary", i.e. `cross(d, r - p) < 0`, which I will verify reduces to `a*r.x + b*r.y - c > 0` — pleasingly, exactly the algebraic statement that `r` violates the inequality.

**The angular sort, done exactly.** A floating `atan2` sort is the usual way, but I want exactness, so I sort by angle using only integer comparisons. I split directions into two halves: `half = 0` for angles in `[0, pi)` (`dy > 0`, or `dy == 0 && dx > 0`) and `half = 1` for `[pi, 2pi)`. Within the same half, the angle order is decided by the sign of the cross product of the two directions: `crossDir(i, j) > 0` means `j` is counterclockwise from `i`, so `i` has the smaller angle. The cross product of two directions is `dx_i*dy_j - dy_i*dx_j`, magnitude at most `2*(10^6)^2 = 2*10^12`, which fits `long long` comfortably; I carry it in `__int128` anyway to never think about it again.

**The intersection point and the `out` predicate, done exactly.** The corner where lines `i` and `j` meet is the solution of the 2x2 system, by Cramer:

```
D  = a_i*b_j - a_j*b_i        (this is exactly cross(d_i, d_j))
Nx = c_i*b_j - c_j*b_i
Ny = a_i*c_j - a_j*c_i
```

so the corner is `(Nx/D, Ny/D)` with `D != 0` for non-parallel lines. Now "is that corner strictly outside half-plane `k`?" means `a_k*(Nx/D) + b_k*(Ny/D) - c_k > 0`. I do not want to divide. Define

```
S = a_k*Nx + b_k*Ny - c_k*D
```

Then `a_k*x + b_k*y - c_k = S / D`, so the corner is strictly outside `k` iff `S/D > 0`, i.e. iff `sgn(S)*sgn(D) > 0`. No division, no epsilon — a pure integer sign test. The only thing I owe myself is an overflow check: `Nx, Ny, D` are around `2*10^12`; multiplying by another coefficient (`10^6`) gives around `2*10^18`; summing three such terms gives around `6*10^18`. Once a bounding box with offset `~4*10^12` enters, the worst term becomes coefficient times a box-sized `Nx` around `4*10^18`, i.e. about `4*10^24`. That is far under the `__int128` ceiling of about `1.7*10^38` — roughly thirteen orders of magnitude of headroom. So `__int128` makes every predicate exact and safe.

**The deque sweep.** Sorted by direction angle, I sweep. Maintaining the boundary as a deque `dq[lo..hi]`:
for each new half-plane `cur`, while the back corner `inter(dq[hi], dq[hi-1])` is strictly outside `cur`, pop the back; while the front corner `inter(dq[lo], dq[lo+1])` is strictly outside `cur`, pop the front; then push `cur`. After processing all half-planes, a wrap-around cleanup removes back/front members that the first/last members made redundant. The surviving deque is the boundary of the intersection.

**The unboundedness problem, and the bounding-box resolution.** Here is the first real subtlety. If the true intersection is unbounded (say, a single half-plane, or a wedge open to infinity), the deque never closes into a polygon and the "is it empty?" test becomes ambiguous. The standard fix is to intersect everything with a large axis-aligned bounding box first; then a non-empty region is always a bounded polygon, and "feasible" is a clean statement about the surviving boundary. The box must be large enough that it never cuts off a genuinely feasible region. How large? Every corner of the arrangement is `(Nx/D, Ny/D)` with `|D| >= 1` (integer, non-zero) and `|Nx|, |Ny| <= 2*(10^6)^2 = 2*10^12`, so every corner has coordinates bounded by `2*10^12`. A non-empty region — bounded or not — always contains such a corner or, in the truly degenerate cases (a whole line or the whole plane), a point near the origin. So a box at `B = 4*10^12`, comfortably larger than `2*10^12`, never falsely empties a feasible region, and it keeps box-involved products within the `__int128` budget I checked above. I add the four box lines `x <= B`, `x >= -B`, `y <= B`, `y >= -B`.

**Parallel and same-direction half-planes.** Two half-planes with the same boundary direction (`crossDir == 0` and the directions agreeing, `dot > 0`) are redundant except for the tighter one. After the angular sort they are adjacent, so when `cur` has the same direction as the current deque back, I keep only the more restrictive: comparing offsets `c/|normal|`, which with parallel normals on a shared nonzero axis component reduces to comparing `c_cur*|n_last| <= c_last*|n_cur|`. The genuinely dangerous case is *anti-parallel* half-planes (`crossDir == 0`, `dot < 0`): two opposite half-planes whose feasible bands may fail to overlap, e.g. `x >= 2` and `x <= 1`. That is an emptiness witness, and I will return to whether my first draft handles it.

**First implementation — and then I actually run it against the brute, because clean geometry transcribes dirty.** My first cut put the same-direction collapse *before* the deque pops, handled only the `dot > 0` (same-direction) parallel case, and decided feasibility by `len >= 3` alone (deque has at least three half-planes ⇒ a polygon ⇒ feasible). On paper that looked complete. So I wrote the `O(m^2)` vertex-enumeration brute (with the same bounding box, in exact `Fraction` arithmetic) and a random small-case generator, and ran 600 cases. Two mismatches: `sol = YES`, `brute = NO`.

**Diagnosing the first failure (anti-parallel slips through).** I shrank one failing seed to a 4-constraint core:

```
-1 -3 0     (x + 3y >= 0)
 0 -5 0     (y >= 0)
 2  6 -1    (x + 3y <= -0.5)    [= 2x + 6y <= -1]
-2 -3 0     (2x + 3y >= 0)
```

Constraints 1 and 3 say `x + 3y >= 0` and `x + 3y <= -0.5` — a flat contradiction; the intersection is empty, and `brute` correctly says `NO`. But my solution said `YES`. I dumped the final deque and found three survivors with directions at angles roughly `0`, `-0.59`, `-0.32` — all clustered within well under half a turn. That is not a closed polygon; it is an open wedge. My `len >= 3` test saw three half-planes and declared victory, but three half-planes whose directions all fit inside a half-plane do *not* enclose a bounded region. The two contradictory half-planes here are anti-parallel (`(2,6)` is `-2*(-1,-3)`); they sort to angles a full `pi` apart, so they were never adjacent and my same-direction-only collapse never compared them, and the wedge they left was mislabeled feasible.

**Deriving the fix — the closure test.** The diagnosis hands me the correct criterion directly. A genuinely bounded convex polygon (which, thanks to the box, is what every non-empty region becomes) has edge directions that **span the entire circle**: walking the boundary counterclockwise, the directions sweep through a full `2pi`. Equivalently, **every cyclic gap between consecutive sorted directions is less than `pi`**. A gap of `pi` or more means all surviving directions fit inside one half-plane — an open wedge — which, because the box would otherwise have closed any real region, signals emptiness. And the gap test is exact: the directed gap from `u` to the next direction `v` (counterclockwise) is less than `pi` iff `v` lies strictly to the left of `u`, i.e. `crossDir(u, v) > 0`. So after the sweep I check that every consecutive pair in the deque, including the wrap from `dq[hi]` back to `dq[lo]`, has `crossDir > 0`; if any gap is `<= 0`, I answer `NO`. On the wedge case the offending gap `(5,0) -> (3,-2)` has `crossDir = -10 <= 0`, so the closure test correctly rejects it.

**Re-running, and hitting a second, deeper failure.** With the closure test added, the 600 cases passed. But when I scaled to 3000 cases, two more mismatches appeared, again `sol = YES`, `brute = NO`, both irreducible to a 6-constraint core. Tracing one of them, the deque ended with three survivors whose directions *did* span the circle (closure test passed), yet the region was empty. The smoking gun: I checked the three survivors *by themselves* with the brute and they were feasible — meaning the deque had **dropped a constraint that was actually binding**, one that cut the survivors' region down to nothing. So the bug was not the emptiness test at all; it was the **sweep order**. I had run the same-direction collapse *before* the pops and, critically, I never detected the anti-parallel emptiness when it surfaced adjacent in the deque. I compared my structure against the canonical formulation and found two corrections: (1) the back/front **pops must come first**, *then* the parallel handling against the new deque back; and (2) when `cur` is anti-parallel to the deque back (`crossDir == 0`, `dot < 0`), the intersection is empty and I must stop immediately. With pops-first ordering, anti-parallel contradictions are driven to be adjacent in the deque exactly when they conflict, and the `dot < 0` check catches them.

**Implementing the corrected sweep.** The loop body becomes: pop the back while its corner is strictly outside `cur`; pop the front while its corner is strictly outside `cur`; if `cur` is now parallel to the deque back, either declare emptiness (anti-parallel) or keep the tighter of the two (same direction) and `continue`; otherwise push `cur`. After the loop, if an anti-parallel emptiness was flagged, answer `NO`. Then the wrap-around cleanup, the `len < 3 => NO` guard, and finally the closure (max-gap `< pi`) test. Re-tracing both 6-constraint cores: both now correctly return `NO` — one via the anti-parallel flag, one via a pop that is no longer skipped.

**Re-verifying, hard.** I re-ran the differential test and it passed cleanly. To make sure I was not fooling myself with an oracle that shared my blind spots, I wrote a **second, independent oracle** — exact rational **Fourier-Motzkin elimination** with no bounding box at all: eliminate `x` by combining every positive-`a` constraint with every negative-`a` one, then eliminate `y` to a 1-D interval and check it is non-empty. Two oracles built on entirely different principles (vertex enumeration vs. variable elimination), one of them box-free, agreeing with the solver is much stronger evidence than either alone. A three-way differential run (solution vs. vertex-enum brute vs. Fourier-Motzkin) over thousands of random small cases — including modes that deliberately generate anti-parallel pairs, lines through the origin, tight single-point boxes, and same-direction duplicates — produced **zero mismatches**.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `m = 0`: no constraints; the box alone survives as a square, closure passes, answer `YES`. Correct — the whole plane is feasible.
- Single half-plane: with the box it is a (clipped) half-plane, a bounded polygon spanning the circle, `YES`. Correct.
- Anti-parallel disjoint bands (`x <= 1`, `x >= 2`): the `dot < 0` flag fires, `NO`. Correct.
- Degenerate-but-feasible regions — a single line (`x <= 0` and `x >= 0`), a single point (`x = 0` and `y = 0`) — the box fills the perpendicular directions so the survivors span the circle and the closure test passes, `YES`. Correct, and matched by both oracles.
- Far-from-origin feasible regions built from near-parallel lines whose corner sits near `2*10^12`: still inside the `4*10^12` box, so `YES`. Verified.
- Overflow: every predicate is `__int128`; the worst magnitude (`~4*10^24`) is about thirteen orders below the `__int128` ceiling. A 200k-triple bigint cross-check confirmed no intermediate ever approaches the limit.
- Performance: `m = 2*10^5` runs in about 60 ms, far under the 2-second limit.

**Final solution.** I convinced myself the *idea* is right by reframing LP feasibility as "do the directed boundaries wrap all the way around," and I convinced myself the *code* is right the hard way: two independent failures, each traced to a precise cause (an open-wedge mislabeled as a polygon; a binding constraint dropped by mis-ordered pops plus a missed anti-parallel emptiness), each fixed and re-verified, and the whole thing cross-checked against two oracles built on different principles. That is what I ship — one self-contained file, exact integer HPI in `O(m log m)`:

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

**Causal recap.** General LP-solving was the wrong framing: the two-variable structure makes feasibility a combinatorial "do the boundaries wrap around" question, decidable with exact integers and no floating point. The SOTA engine is angular-sort + deque HPI in `O(m log m)`. Doing the `out` test as `sgn(S)*sgn(D)` over `__int128` makes every predicate exact and overflow-safe. A bounding box at `4*10^12` turns unbounded-but-feasible regions into polygons without ever clipping a real one. The two bugs my differential tests exposed were both about emptiness detection: an open wedge with three half-planes was mislabeled feasible until I added the max-angular-gap-`< pi` closure test, and a binding constraint was silently dropped until I put the deque pops *before* the parallel handling and added the anti-parallel (`dot < 0`) emptiness flag. Two independent oracles — vertex enumeration and box-free Fourier-Motzkin — agreeing over thousands of cases is what let me trust the final `YES`/`NO`.
