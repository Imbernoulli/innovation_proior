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
